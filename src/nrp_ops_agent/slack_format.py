"""Markdown -> Slack mrkdwn.

The model writes Markdown. Slack's ``text`` field is *not* Markdown -- it is
mrkdwn, a much smaller syntax that overlaps just enough to be misleading.
``**bold**`` renders as literal asterisks, ``## Heading`` as a literal hash,
``[docs](https://nrp.ai/x)`` as literal brackets. An answer full of
``**OOMKilled**`` and ``- item`` is still readable, but it reads as a bug, and
the doc links -- the part an operator is meant to click to check the finding --
arrive unclickable.

Asking the model to emit mrkdwn (see :mod:`nrp_ops_agent.prompts`) gets most of
the way there and nothing more: models drift back to Markdown, especially on
long answers, and a bulleted list is exactly where they drift. So the prompt
asks and this module enforces. It is idempotent on text that is already mrkdwn,
which is what makes running both safe.

What Slack accepts, and what we therefore translate to:

===================  ======================  ==========================
Markdown             Slack mrkdwn            note
===================  ======================  ==========================
``**bold**``         ``*bold*``
``__bold__``         ``*bold*``
``*text*``           ``*text*``              left alone; see
                                             :func:`_convert_emphasis`
``~~strike~~``       ``~strike~``
``# Heading``        ``*Heading*``           mrkdwn has no headings
``- item``           ``• item``              mrkdwn has no list markup
``[text](url)``      ``<url|text>``
``---``              a rule of box glyphs    mrkdwn has no rule
pipe table           a code block            mrkdwn has no tables
===================  ======================  ==========================

Code is the reason this is a segmenting parser rather than a pile of
:func:`re.sub` calls. Operators get YAML, PromQL and log lines back, and those
are full of ``*``, ``_`` and ``|``; rewriting emphasis inside a fenced block
would corrupt the evidence the answer rests on. Code spans are therefore lifted
out first and put back last, untouched apart from the entity escaping Slack
requires everywhere.
"""

from __future__ import annotations

import re
from typing import Final

#: Placeholders for text that must survive the prose rewrites verbatim. NUL
#: cannot appear in Slack message text, so it cannot collide with content.
_STASH_OPEN: Final = "\x00"
_STASH_CLOSE: Final = "\x00"

_FENCE_RE: Final = re.compile(r"[ \t]*```[^\n`]*\n?(?P<body>.*?)```", re.S)
_INLINE_CODE_RE: Final = re.compile(r"(?P<ticks>`+)(?P<body>.+?)(?P=ticks)", re.S)

_HEADING_RE: Final = re.compile(r"^[ \t]*#{1,6}[ \t]+(?P<text>.+?)[ \t]*#*[ \t]*$", re.M)
_RULE_RE: Final = re.compile(
    r"^[ \t]*(?:-[ \t]*-[ \t]*-|\*[ \t]*\*[ \t]*\*|_[ \t]*_[ \t]*_)[-*_ \t]*$", re.M
)
_BULLET_RE: Final = re.compile(r"^(?P<indent>[ \t]*)[-*+][ \t]+(?=\S)", re.M)
_LINK_RE: Final = re.compile(r"!?\[(?P<text>[^\]\n]*)\]\((?P<url>[^)\s]+)(?:[ \t]+\"[^\"]*\")?\)")
_AUTOLINK_RE: Final = re.compile(r"<(?P<url>(?:https?|mailto):[^>\s|]+)>")

_STRIKE_RE: Final = re.compile(r"~~(?=\S)(?P<text>.+?)(?<=\S)~~", re.S)
_BOLD_STAR_RE: Final = re.compile(r"\*\*(?=\S)(?P<text>.+?)(?<=\S)\*\*", re.S)
_BOLD_UNDER_RE: Final = re.compile(r"(?<![\w\\])__(?=\S)(?P<text>.+?)(?<=\S)__(?!\w)", re.S)
#: A single-asterisk run. Deliberately *not* rewritten in prose -- see
#: :func:`_convert_emphasis` -- but stripped where emphasis cannot render.
_EMPH_STAR_RE: Final = re.compile(r"(?<![\w*])\*(?=\S)(?P<text>[^*\n]+?)(?<=\S)\*(?![\w*])")

#: A blockquote marker, after entity escaping has turned its ``>`` into
#: ``&gt;``. mrkdwn has blockquotes and they are written exactly as Markdown
#: writes them, so this is the one escape that has to be undone.
_QUOTED_ESCAPE_RE: Final = re.compile(r"^(?P<indent>[ \t]*)(?P<marks>(?:&gt;[ \t]*)+)", re.M)

#: Slack entities the model may echo back from an earlier turn (``<@U123>``,
#: ``<#C0FOO|ops>``, ``<!here>``, or a link we ourselves emitted). These are
#: already mrkdwn; escaping their angle brackets would turn a working mention
#: into literal text, so they are stashed before escaping like code is.
_SLACK_ENTITY_RE: Final = re.compile(r"<[@#!][^<>\n]*>|<(?:https?|mailto):[^<>\n]*\|[^<>\n]*>")

#: Rendered in place of a ``---`` rule. mrkdwn has no rule, and box-drawing
#: characters read as one at any font size.
_RULE_TEXT: Final = "─" * 24

_BULLETS: Final = ("•", "◦", "▪")
#: Columns of one tab, or of two spaces, are both a level of nesting.
_INDENT_WIDTH: Final = 2

_TABLE_DIVIDER_RE: Final = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)+\|?[ \t]*$"
)


def escape_entities(text: str) -> str:
    """Escape the three characters Slack reserves in message text.

    Slack asks for this everywhere, code blocks included: an unescaped ``<`` in
    a log line starts what Slack tries to read as a link and eats the text after
    it. ``&`` goes first or it would double-escape the two that follow.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _Stash:
    """Holds chunks that must not be rewritten, keyed by an unusable character."""

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def put(self, chunk: str) -> str:
        self._chunks.append(chunk)
        return f"{_STASH_OPEN}{len(self._chunks) - 1}{_STASH_CLOSE}"

    def restore(self, text: str) -> str:
        # Reversed: a stashed chunk may itself contain an earlier placeholder
        # (a table cell holding an inline code span), and the outer chunk has to
        # be put back before the inner one can be found.
        for index in reversed(range(len(self._chunks))):
            text = text.replace(f"{_STASH_OPEN}{index}{_STASH_CLOSE}", self._chunks[index])
        return text


def to_mrkdwn(text: str) -> str:
    """Convert Markdown to Slack mrkdwn, leaving code and existing mrkdwn alone."""
    if not text:
        return text

    stash = _Stash()
    out = _stash_code(text, stash)
    out = _SLACK_ENTITY_RE.sub(lambda m: stash.put(m.group(0)), out)
    out = _stash_tables(out, stash)

    # Autolink brackets go before escaping, or `<https://x>` escapes into
    # `&lt;https://x&gt;` and there is no bracket left to recognise.
    out = _AUTOLINK_RE.sub(lambda m: m.group("url"), out)
    out = escape_entities(out)
    out = _QUOTED_ESCAPE_RE.sub(_unescape_quote, out)

    out = _convert_links(out)
    out = _convert_blocks(out)
    out = _convert_emphasis(out)

    return stash.restore(out)


# ------------------------------------------------------------------- code --


def _stash_code(text: str, stash: _Stash) -> str:
    """Lift fenced blocks and inline spans out, normalised for Slack."""

    def fence(match: re.Match[str]) -> str:
        # The info string is Markdown's, not Slack's: mrkdwn has no syntax
        # highlighting and renders ```yaml as a first line reading "yaml".
        body = match.group("body").lstrip("\n")
        if not body.endswith("\n"):
            body += "\n"
        return stash.put("```\n" + escape_entities(body) + "```")

    text = _FENCE_RE.sub(fence, text)
    return _INLINE_CODE_RE.sub(lambda m: stash.put(f"`{escape_entities(m.group('body'))}`"), text)


# ----------------------------------------------------------------- tables --


def _stash_tables(text: str, stash: _Stash) -> str:
    """Rewrite pipe tables as code blocks, which is the only aligned text Slack has.

    A table is the one construct with no mrkdwn equivalent at all. Dropping the
    pipes would run the columns together, so the columns are padded and the
    whole thing goes into a fence -- monospace is what makes a column a column.
    """
    lines = text.split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        rows, consumed = _read_table(lines, index)
        if rows is None:
            out.append(lines[index])
            index += 1
            continue
        out.append(stash.put(_render_table(rows)))
        index += consumed
    return "\n".join(out)


def _read_table(lines: list[str], start: int) -> tuple[list[list[str]] | None, int]:
    """Read a header/divider/body table at ``start``; ``(None, 0)`` if there is none."""
    if start + 1 >= len(lines):
        return None, 0
    if "|" not in lines[start] or not _TABLE_DIVIDER_RE.match(lines[start + 1]):
        return None, 0

    rows = [_split_row(lines[start])]
    index = start + 2
    while index < len(lines) and "|" in lines[index] and lines[index].strip():
        rows.append(_split_row(lines[index]))
        index += 1
    return rows, index - start


def _split_row(line: str) -> list[str]:
    cells = line.strip().split("|")
    # A row may or may not be wrapped in leading/trailing pipes; both are legal
    # Markdown and both produce an empty edge cell when split.
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return [_strip_cell(cell) for cell in cells]


def _strip_cell(cell: str) -> str:
    """A cell is prose, so its own markup has to go -- a fence renders none of it."""
    cell = _LINK_RE.sub(lambda m: m.group("text") or m.group("url"), cell.strip())
    return _strip_emphasis(cell)


def _strip_emphasis(text: str) -> str:
    """Drop emphasis markers where emphasis cannot render: fences and headings."""
    text = _BOLD_STAR_RE.sub(lambda m: m.group("text"), text)
    text = _BOLD_UNDER_RE.sub(lambda m: m.group("text"), text)
    return _EMPH_STAR_RE.sub(lambda m: m.group("text"), text)


def _render_table(rows: list[list[str]]) -> str:
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    columns = [max(len(row[col]) for row in padded) for col in range(width)]
    body = "\n".join(
        "  ".join(cell.ljust(columns[col]) for col, cell in enumerate(row)).rstrip()
        for row in padded
    )
    return f"```\n{escape_entities(body)}\n```"


# ------------------------------------------------------------ prose rules --


def _unescape_quote(match: re.Match[str]) -> str:
    """Put back the ``>`` of a blockquote marker that escaping just took away."""
    return match.group("indent") + "> " * match.group("marks").count("&gt;")


def _convert_links(text: str) -> str:
    """``[text](url)`` -> ``<url|text>``; a bare autolink loses its brackets.

    Slack auto-links a naked URL, so ``<url|url>`` would only add noise; an
    image is flattened to its link because mrkdwn cannot inline one.
    """

    def link(match: re.Match[str]) -> str:
        url = match.group("url")
        label = match.group("text").strip()
        if not label or label == url:
            return url
        return f"<{url}|{label}>"

    text = _AUTOLINK_RE.sub(lambda m: m.group("url"), text)
    return _LINK_RE.sub(link, text)


def _convert_blocks(text: str) -> str:
    """Line-level structure: headings, rules and bullets."""
    text = _RULE_RE.sub(_RULE_TEXT, text)
    # After the rule, or `- - -` would be read as a bullet holding `- -`.
    # A heading's own emphasis is dropped: `## **Evidence**` bolded again would
    # come out as `***Evidence***`, which renders as literal asterisks.
    text = _HEADING_RE.sub(lambda m: f"*{_strip_emphasis(m.group('text'))}*", text)
    return _BULLET_RE.sub(_bullet, text)


def _bullet(match: re.Match[str]) -> str:
    indent = match.group("indent").expandtabs(_INDENT_WIDTH)
    depth = len(indent) // _INDENT_WIDTH
    return f"{indent}{_BULLETS[min(depth, len(_BULLETS) - 1)]} "


def _convert_emphasis(text: str) -> str:
    """``**b**`` -> ``*b*``, ``__b__`` -> ``*b*``, ``~~s~~`` -> ``~s~``.

    A single ``*x*`` is left exactly as it is, and that is the one deliberate
    infidelity here. Markdown reads it as italic and Slack reads it as bold, so
    there is no rendering that satisfies both. Rewriting it to ``_x_`` would be
    the faithful-to-Markdown choice and the wrong one: the system prompt asks
    the model for mrkdwn, so most single-asterisk runs reaching this function
    were *meant* as Slack bold, and rewriting them would demote exactly the
    words the model chose to emphasise. Leaving them alone also makes the whole
    conversion idempotent -- output that is already mrkdwn passes through
    unchanged -- which is what lets the prompt and this module both run without
    fighting each other. The cost is a Markdown italic rendering as bold.
    """
    text = _STRIKE_RE.sub(lambda m: f"~{m.group('text')}~", text)
    text = _BOLD_STAR_RE.sub(lambda m: f"*{m.group('text')}*", text)
    return _BOLD_UNDER_RE.sub(lambda m: f"*{m.group('text')}*", text)


# -------------------------------------------------------------- truncation --


def fit(text: str, limit: int) -> str:
    """Truncate to ``limit`` characters without leaving a code fence open.

    Slack renders an unterminated ``` as one long code block swallowing the rest
    of the message, so a naive slice can turn a merely-truncated answer into an
    unreadable one. Cutting at the last line boundary that keeps the fences
    balanced costs a line and keeps the rest legible.
    """
    if len(text) <= limit:
        return text

    marker = "\n_[truncated]_"
    cut = text[: max(0, limit - len(marker))]
    if cut.count("```") % 2:
        # Inside a block: close it, giving back room for the fence itself.
        cut = cut[: len(cut) - len("\n```")].rstrip()
        return f"{cut}\n```{marker}"
    return cut.rstrip() + marker

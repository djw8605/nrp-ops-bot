"""Markdown -> mrkdwn conversion.

The bug these cover: the model writes Markdown, Slack renders mrkdwn, and the
two agree on so little that an answer arrived full of literal `**`, `##` and
`[label](url)` brackets -- with the doc links, the part an operator clicks to
check the finding, not clickable.
"""

from __future__ import annotations

import pytest

from nrp_ops_agent.slack_format import escape_entities, fit, to_mrkdwn


class TestEmphasis:
    def test_double_asterisk_bold_becomes_single(self) -> None:
        assert to_mrkdwn("**coder-0** is down") == "*coder-0* is down"

    def test_underscore_bold_becomes_asterisk(self) -> None:
        assert to_mrkdwn("__coder-0__ is down") == "*coder-0* is down"

    def test_a_single_asterisk_run_is_left_alone(self) -> None:
        """Markdown reads `*x*` as italic, Slack as bold, and the prompt asks the
        model for Slack. Rewriting it to `_x_` would demote the words the model
        chose to emphasise, and would stop the conversion being idempotent."""
        assert to_mrkdwn("this is *probably* the cause") == "this is *probably* the cause"

    def test_converted_bold_is_not_rewritten_again(self) -> None:
        assert to_mrkdwn("**a** and **b**") == "*a* and *b*"

    def test_underscore_italic_is_already_mrkdwn(self) -> None:
        assert to_mrkdwn("this is _probably_ the cause") == "this is _probably_ the cause"

    def test_strikethrough_loses_one_tilde(self) -> None:
        assert to_mrkdwn("~~not this~~") == "~not this~"

    def test_existing_mrkdwn_survives_unchanged(self) -> None:
        """Prompt and converter both run, so the converter must be idempotent."""
        already = "*bold* and _italic_ and `code`"
        assert to_mrkdwn(already) == already
        once = to_mrkdwn("**bold** and _italic_ and [x](https://nrp.ai/x)\n- item\n## H")
        assert to_mrkdwn(once) == once

    def test_snake_case_identifiers_are_left_alone(self) -> None:
        assert to_mrkdwn("check nrp_ops_agent and max_pods") == "check nrp_ops_agent and max_pods"

    def test_a_lone_asterisk_is_not_emphasis(self) -> None:
        assert to_mrkdwn("2 * 3 replicas") == "2 * 3 replicas"


class TestBlocks:
    @pytest.mark.parametrize("hashes", ["#", "##", "######"])
    def test_headings_become_bold_lines(self, hashes: str) -> None:
        assert to_mrkdwn(f"{hashes} Evidence") == "*Evidence*"

    def test_bullets_become_glyphs(self) -> None:
        assert to_mrkdwn("- one\n- two") == "• one\n• two"

    @pytest.mark.parametrize("marker", ["-", "*", "+"])
    def test_every_bullet_marker_is_recognised(self, marker: str) -> None:
        assert to_mrkdwn(f"{marker} item") == "• item"

    def test_nested_bullets_get_a_distinct_glyph(self) -> None:
        assert to_mrkdwn("- top\n  - nested") == "• top\n  ◦ nested"

    def test_numbered_lists_are_left_as_written(self) -> None:
        """mrkdwn has no ordered list, and `1.` already reads as one."""
        assert to_mrkdwn("1. first\n2. second") == "1. first\n2. second"

    def test_a_rule_becomes_something_that_renders(self) -> None:
        out = to_mrkdwn("above\n\n---\n\nbelow")
        assert "---" not in out
        assert "─" in out

    def test_blockquotes_are_already_mrkdwn(self) -> None:
        assert to_mrkdwn("> quoted") == "> quoted"


class TestLinks:
    def test_inline_link_becomes_slack_syntax(self) -> None:
        out = to_mrkdwn("see [the runbook](https://nrp.ai/docs/coder)")
        assert out == "see <https://nrp.ai/docs/coder|the runbook>"

    def test_a_bare_url_is_left_for_slack_to_autolink(self) -> None:
        assert to_mrkdwn("see https://nrp.ai/docs") == "see https://nrp.ai/docs"

    def test_an_autolink_loses_its_brackets(self) -> None:
        assert to_mrkdwn("see <https://nrp.ai/docs>") == "see https://nrp.ai/docs"

    def test_a_link_labelled_with_its_own_url_is_flattened(self) -> None:
        assert to_mrkdwn("[https://nrp.ai/x](https://nrp.ai/x)") == "https://nrp.ai/x"

    def test_a_url_query_string_is_escaped_but_intact(self) -> None:
        out = to_mrkdwn("[metrics](https://nrp.ai/g?a=1&b=2)")
        assert out == "<https://nrp.ai/g?a=1&amp;b=2|metrics>"

    def test_an_existing_slack_link_is_not_double_escaped(self) -> None:
        already = "see <https://nrp.ai/docs|the runbook>"
        assert to_mrkdwn(already) == already

    def test_a_user_mention_survives(self) -> None:
        assert to_mrkdwn("ping <@U0OPERATOR1>") == "ping <@U0OPERATOR1>"


class TestCode:
    def test_a_fence_loses_its_language_tag(self) -> None:
        """Slack has no highlighting and renders ```yaml as a line saying yaml."""
        assert to_mrkdwn("```yaml\nkey: value\n```") == "```\nkey: value\n```"

    def test_emphasis_inside_a_fence_is_not_rewritten(self) -> None:
        """The fence holds evidence. Rewriting a PromQL selector corrupts it."""
        query = '```\nsum(rate(x[5m])) * 100 / ignoring(a) sum(y{b="c"})\n```'
        assert to_mrkdwn(query) == query

    def test_markdown_inside_a_fence_is_not_converted(self) -> None:
        source = "```\n**literal** and [a](b)\n```"
        assert to_mrkdwn(source) == source

    def test_inline_code_is_left_alone(self) -> None:
        assert to_mrkdwn("run `kubectl get pods -l app=*`") == "run `kubectl get pods -l app=*`"

    def test_angle_brackets_in_code_are_escaped(self) -> None:
        """Unescaped, Slack reads `<none>` as a link and swallows what follows."""
        out = to_mrkdwn("`RESTARTS <none>`")
        assert out == "`RESTARTS &lt;none&gt;`"

    def test_prose_around_a_fence_is_still_converted(self) -> None:
        out = to_mrkdwn("**before**\n```\nx = *y*\n```\n**after**")
        assert out == "*before*\n```\nx = *y*\n```\n*after*"


class TestTables:
    def test_a_pipe_table_becomes_an_aligned_code_block(self) -> None:
        table = "| pod | restarts |\n| --- | --- |\n| coder-0 | 47 |\n| coder-1 | 0 |"
        out = to_mrkdwn(table)
        assert out.startswith("```\n") and out.endswith("\n```")
        assert "|" not in out
        lines = out.splitlines()
        assert lines[1] == "pod      restarts"
        assert lines[2] == "coder-0  47"

    def test_table_cells_lose_their_markdown(self) -> None:
        table = "| pod | doc |\n| --- | --- |\n| **coder-0** | [runbook](https://nrp.ai/c) |"
        out = to_mrkdwn(table)
        assert "**" not in out
        assert "coder-0" in out and "runbook" in out

    def test_prose_after_a_table_is_still_converted(self) -> None:
        out = to_mrkdwn("| a | b |\n| --- | --- |\n| 1 | 2 |\n\n**after**")
        assert out.endswith("*after*")

    def test_a_pipe_in_prose_is_not_a_table(self) -> None:
        assert to_mrkdwn("grep foo | wc -l") == "grep foo | wc -l"


class TestEscaping:
    def test_the_three_reserved_characters_are_escaped(self) -> None:
        assert escape_entities("a & b < c > d") == "a &amp; b &lt; c &gt; d"

    def test_ampersand_is_escaped_first(self) -> None:
        """`&` last would re-escape the `&` of `&lt;` into `&amp;lt;`."""
        assert escape_entities("<") == "&lt;"

    def test_prose_angle_brackets_are_escaped(self) -> None:
        assert to_mrkdwn("exit code <137>") == "exit code &lt;137&gt;"


class TestFit:
    def test_short_text_is_untouched(self) -> None:
        assert fit("short", 100) == "short"

    def test_over_the_limit_is_cut_and_marked(self) -> None:
        out = fit("x" * 500, 100)
        assert len(out) <= 100
        assert "truncated" in out

    def test_a_cut_inside_a_fence_closes_it(self) -> None:
        """An unterminated ``` makes Slack render the rest of the message as one
        code block, so a truncated answer becomes an unreadable one."""
        out = fit("intro\n```\n" + "log line\n" * 200, 120)
        assert len(out) <= 120
        assert out.count("```") % 2 == 0

    def test_a_balanced_fence_is_left_balanced(self) -> None:
        out = fit("```\na\n```\n" + "tail " * 200, 200)
        assert out.count("```") % 2 == 0


class TestDegenerateInput:
    @pytest.mark.parametrize("text", ["", "\n", "   ", "**", "```", "[](", "|", "- "])
    def test_malformed_markdown_does_not_raise(self, text: str) -> None:
        assert isinstance(to_mrkdwn(text), str)

    def test_an_unclosed_fence_is_not_swallowed(self) -> None:
        out = to_mrkdwn("```\nlog line without a close")
        assert "log line without a close" in out

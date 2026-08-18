"""Chart rendering, and the per-turn collector that carries images to Slack.

An accounting answer is mostly a shape: "GPU hours climbed for three weeks and
then fell off a cliff" is one glance at a line and three sentences of prose. The
prose still goes in the reply -- a chart nobody can grep is not evidence -- but
the picture is what makes the trend arguable.

Security role: small but real, and it is the *ordering* that carries it.

* Nothing here reads the cluster. Charts are drawn from accounting aggregates
  returned by :mod:`nrp_ops_agent.tools.accounting` -- dates, numbers, and the
  names of namespaces, institutions and nodes. Pod logs and event messages, the
  two genuinely attacker-authored sources, never reach a label.
* Labels are redacted *before* they are drawn, by
  :func:`~nrp_ops_agent.tools.accounting` calling :func:`clean_label`. Once text
  is rasterised it cannot be scrubbed, so the outbound chokepoint in
  :mod:`nrp_ops_agent.slack_app` cannot help a PNG -- redaction has to happen on
  this side of the renderer.
* Image bytes never enter the model's context. A tool returns the *numbers* it
  plotted; the PNG travels out of band through :func:`collecting`.

The collector is a context variable rather than a return value because the tool
dispatcher's contract is "handler returns JSON". Widening that to carry binary
payloads would touch every tool; a context-local list touches one.

Style follows a single validated palette (see ``PALETTE``) so that two charts in
one thread read as one system. Matplotlib is imported lazily: it costs about a
second and most turns never draw anything.
"""

from __future__ import annotations

import contextvars
import io
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final

from nrp_ops_agent.redact import redact

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

#: Categorical hues, assigned by slot and never cycled. Validated as a set on
#: the light surface below: worst adjacent pair is 9.1 ΔE simulated for
#: colour-vision deficiency (target >= 8) and 19.6 unsimulated (floor 15). Three
#: of them sit under 3:1 contrast against the surface, which is why every chart
#: here carries a legend or a direct label -- identity is never colour alone.
PALETTE: Final[tuple[str, ...]] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
)

#: Everything that is not one of the top series. Deliberately grey: "Other" is
#: an absence of identity, and giving it a hue implies it is a thing.
OTHER_COLOUR: Final = "#a8a69d"
OTHER_LABEL: Final = "Other"

SURFACE: Final = "#fcfcfb"
INK: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
MUTED: Final = "#898781"
GRID: Final = "#e1e0d9"
BASELINE: Final = "#c3c2b7"

#: 1400x730 at this dpi -- wide enough for 90 daily points, small enough that
#: Slack shows it inline rather than as a download.
FIGURE_INCHES: Final = (10.0, 5.2)
DPI: Final = 140

#: Bars past this many are unreadable at the figure height above, and a ranking
#: nobody can read is worse than a list.
MAX_BARS: Final = 15

#: Series past this fold into ``Other``. Six is the palette length.
MAX_SERIES: Final = len(PALETTE)


# --------------------------------------------------------------------------- #
# The unit of output
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chart:
    """One rendered image, on its way to Slack.

    ``title`` and ``alt_text`` are posted as text alongside the file, so they
    are redacted like any other outbound string. ``png`` is opaque.
    """

    filename: str
    title: str
    alt_text: str
    png: bytes

    @property
    def size_bytes(self) -> int:
        return len(self.png)


@dataclass
class Collector:
    """Charts drawn during one investigation, plus the budget that bounds them."""

    limit: int
    charts: list[Chart] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - len(self.charts))

    def add(self, chart: Chart) -> bool:
        if self.remaining <= 0:
            return False
        self.charts.append(chart)
        return True


_collector: contextvars.ContextVar[Collector | None] = contextvars.ContextVar(
    "nrp_ops_charts", default=None
)


@contextmanager
def collecting(limit: int) -> Iterator[Collector]:
    """Collect charts drawn inside this block.

    Tool handlers run in tasks spawned by ``asyncio.gather``, which copy the
    context but share the :class:`Collector` object itself -- so appends made
    inside a tool are visible here.
    """
    collector = Collector(limit=limit)
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)


def emit(chart: Chart) -> bool:
    """Attach a chart to the running investigation.

    Returns ``False`` when there is no collector (an MCP client, a unit test) or
    when the per-turn budget is spent, so the caller can tell the model its
    picture will not appear rather than letting it promise one that never does.
    """
    collector = _collector.get()
    if collector is None:
        return False
    return collector.add(chart)


def budget_remaining() -> int:
    collector = _collector.get()
    return 0 if collector is None else collector.remaining


def clean_label(text: object) -> str:
    """Redact a string that is about to be rasterised.

    Namespace and node names are user-chosen, and this is the last point at
    which a credential pasted into one is still text.
    """
    scrubbed, _ = redact(str(text))
    return scrubbed


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


#: Scale thresholds, largest first. GPU hours run to the millions and token
#: counts to the billions, so an unscaled axis is a wall of digits.
_SCALES: Final[tuple[tuple[float, str], ...]] = (
    (1e12, "T"),
    (1e9, "B"),
    (1e6, "M"),
    (1e4, "k"),
)


def formatter(peak: float) -> Callable[[float], str]:
    """One number format for a whole chart, chosen from its largest value.

    Per-value scaling reads badly in a column -- "15.8k" above "9,120" makes the
    smaller number look bigger at a glance. Picking the scale once from the peak
    means every label on one chart is directly comparable, and the axis ticks
    match the tip labels because both come through here.
    """
    divisor, suffix = 1.0, ""
    for threshold, symbol in _SCALES:
        if abs(peak) >= threshold:
            divisor, suffix = threshold if symbol != "k" else 1e3, symbol
            break

    def render(value: float) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "-"
        if value == 0:
            return "0"  # "0k" is not a number anyone writes
        if not suffix:
            return f"{value:,.0f}" if abs(value) >= 1 or value == 0 else f"{value:.3g}"
        # ".0" carries no information and makes a tick row noisy; the digit
        # after the point does, whenever a value has one.
        return f"{value / divisor:.1f}{suffix}".replace(f".0{suffix}", suffix)

    return render


def compact(value: float) -> str:
    """A single number formatted on its own scale. For prose, not for axes."""
    return formatter(value)(value)


def _tick_labels(x_labels: Sequence[str], count: int = 8) -> tuple[list[int], list[str]]:
    """Evenly spaced tick positions, always including the last point.

    The right-hand end is where the answer usually is ("is it still climbing?"),
    so it is the one tick that is never dropped to make the spacing tidy.
    """
    total = len(x_labels)
    if total == 0:
        return [], []
    if total <= count:
        positions = list(range(total))
    else:
        step = max(1, round((total - 1) / (count - 1)))
        positions = list(range(total - 1, -1, -step))[::-1]

    shown = [str(x_labels[i]) for i in positions]
    # ISO dates spanning one year need no year on every tick; longer ones do.
    if all(len(s) == 10 and s[4] == "-" and s[7] == "-" for s in shown):
        years = {s[:4] for s in shown}
        if len(years) == 1:
            shown = [s[5:] for s in shown]
    return positions, shown


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Series:
    """One line or one stacked band. ``values`` aligns with the shared x axis."""

    label: str
    values: tuple[float, ...]

    @property
    def total(self) -> float:
        return float(sum(self.values))


def _pyplot() -> Any:
    """Import matplotlib on first use, with the headless backend forced.

    ``Agg`` before ``pyplot``: importing pyplot first lets it pick an
    interactive backend, which in a container without a display is an import
    error rather than a fallback.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # DejaVu ships with matplotlib, so this resolves in the container
            # without a font package. The others are for a developer's laptop.
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "axes.titlesize": 15,
            "font.size": 11,
        }
    )
    return plt


def _frame(plt: Any, *, title: str, subtitle: str, right: float = 0.965) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=FIGURE_INCHES, dpi=DPI)
    # Wide enough on the left for a five-digit thousands-separated tick plus
    # the rotated axis label; at 0.085 the label was clipped by the figure edge.
    fig.subplots_adjust(left=0.105, right=right, top=0.845, bottom=0.145)

    fig.text(0.105, 0.945, title, fontsize=15, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.105, 0.895, subtitle, fontsize=10.5, color=INK_SECONDARY, va="top")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=0, pad=6)
    return fig, ax


def _finish(plt: Any, fig: Any, *, footer: str) -> bytes:
    if footer:
        fig.text(0.024, 0.028, footer, fontsize=9, color=MUTED, va="bottom")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return buffer.getvalue()


def _y_axis(ax: Any, label: str, render: Callable[[float], str]) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: render(v)))
    ax.grid(axis="y", color=GRID, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    if label:
        ax.set_ylabel(label, fontsize=10.5, color=INK_SECONDARY, labelpad=8)
    ax.set_ylim(bottom=0)


def _legend_order(count: int, columns: int) -> list[int]:
    """Permute legend entries so a multi-row legend reads left to right.

    Matplotlib fills a legend column by column, which puts the second-largest
    series under the largest instead of beside it -- the reading order stops
    matching the stacking order, and the legend becomes a lookup table rather
    than a key.
    """
    rows = math.ceil(count / columns)
    order = [(index % rows) * columns + (index // rows) for index in range(rows * columns)]
    return [index for index in order if index < count]


def _end_labels(ax: Any, points: list[tuple[float, float, str]]) -> None:
    """Label series ends, dropping any that would collide with one already placed.

    Nudging colliding labels apart detaches them from their lines, so the rule
    is to drop instead: the legend still carries every series, and a chart with
    two readable labels beats one with five overlapping ones.
    """
    bottom, top = ax.get_ylim()
    span = max(top - bottom, 1e-9)
    minimum_gap = span * 0.075

    placed: list[float] = []
    for x, y, text in sorted(points, key=lambda point: -point[1]):
        if any(abs(y - other) < minimum_gap for other in placed):
            continue
        placed.append(y)
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            # Text wears ink, never the series colour -- the marker beside it
            # carries identity, and three of these hues are illegible as text.
            color=INK,
            clip_on=False,
        )


def render_trend(
    *,
    title: str,
    subtitle: str,
    footer: str,
    x_labels: Sequence[str],
    series: Sequence[Series],
    y_label: str,
    stacked: bool = False,
) -> bytes:
    """A daily trend: one line with a wash under it, or a stack of bands.

    ``stacked`` is for composition over time ("which namespaces make up the
    total"). Bands are solid and separated by a 2px gap in the surface colour,
    which is what keeps adjacent hues apart without drawing a border.
    """
    plt = _pyplot()
    stack = stacked and len(series) > 1
    # End labels sit outside the data area, so the axes stop short of the
    # figure edge when there are any. A stacked chart has none.
    fig, ax = _frame(plt, title=title, subtitle=subtitle, right=0.965 if stack else 0.885)

    x = list(range(len(x_labels)))
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(series))]
    for index, item in enumerate(series):
        if item.label == OTHER_LABEL:
            colours[index] = OTHER_COLOUR

    if stack:
        peak = max(
            (sum(column) for column in zip(*(item.values for item in series), strict=True)),
            default=0.0,
        )
    else:
        peak = max((max(item.values, default=0.0) for item in series), default=0.0)
    render = formatter(peak)

    if stack:
        cumulative = [0.0] * len(x)
        for item, colour in zip(series, colours, strict=True):
            top = [base + value for base, value in zip(cumulative, item.values, strict=True)]
            ax.fill_between(x, cumulative, top, color=colour, linewidth=0, zorder=2)
            # The surface gap: a line in the background colour along the seam.
            ax.plot(x, top, color=SURFACE, linewidth=2.0, zorder=3)
            cumulative = top
        _y_axis(ax, y_label, render)
    else:
        ends: list[tuple[float, float, str]] = []
        for item, colour in zip(series, colours, strict=True):
            ax.plot(
                x,
                item.values,
                color=colour,
                linewidth=2.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )
            if len(series) == 1:
                ax.fill_between(x, 0, item.values, color=colour, alpha=0.10, linewidth=0, zorder=2)
            if x:
                ax.plot(
                    [x[-1]],
                    [item.values[-1]],
                    marker="o",
                    markersize=8,
                    color=colour,
                    # A 2px ring in the surface colour, so end dots stay legible
                    # where two series converge.
                    markeredgecolor=SURFACE,
                    markeredgewidth=2.0,
                    zorder=4,
                )
                ends.append((x[-1], float(item.values[-1]), render(item.values[-1])))
        _y_axis(ax, y_label, render)
        _end_labels(ax, ends)

    positions, labels = _tick_labels(x_labels)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(0, max(len(x) - 1, 1))
    ax.margins(x=0.02)

    if len(series) > 1:
        columns = min(len(series), 4)
        handles = [
            plt.Line2D(
                [], [], marker="s", markersize=9, linestyle="none", color=colour, label=item.label
            )
            for item, colour in zip(series, colours, strict=True)
        ]
        ax.legend(
            handles=[handles[i] for i in _legend_order(len(handles), columns)],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.11),
            ncol=columns,
            frameon=False,
            fontsize=10,
            labelcolor=INK_SECONDARY,
            handletextpad=0.5,
            columnspacing=1.8,
        )
        fig.subplots_adjust(bottom=0.235 if len(series) > 4 else 0.19)

    return _finish(plt, fig, footer=footer)


def render_ranking(
    *,
    title: str,
    subtitle: str,
    footer: str,
    labels: Sequence[str],
    values: Sequence[float],
    x_label: str = "",
) -> bytes:
    """Horizontal bars, largest at the top.

    One hue for every bar on purpose: a ranking encodes magnitude, and length
    already carries it. Colouring by rank would also mean the colours move when
    the window changes, which makes two charts in one thread contradict each
    other.
    """
    plt = _pyplot()
    from matplotlib.ticker import FuncFormatter

    rows = len(labels)
    height = max(3.4, 1.55 + 0.42 * rows)
    fig, ax = plt.subplots(figsize=(FIGURE_INCHES[0], height), dpi=DPI)
    fig.subplots_adjust(left=0.30, right=0.955, top=1.0 - (0.92 / height), bottom=0.62 / height)

    fig.text(
        0.024, 1.0 - (0.26 / height), title, fontsize=15, fontweight="bold", color=INK, va="top"
    )
    if subtitle:
        fig.text(
            0.024, 1.0 - (0.58 / height), subtitle, fontsize=10.5, color=INK_SECONDARY, va="top"
        )

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0, pad=6)

    numbers = [float(value) for value in values]
    render = formatter(max(numbers, default=0.0))

    y = list(range(rows))
    # 0.4 of a 0.42in row at this dpi lands the bar near 24px thick and leaves
    # the rest of the slot as air.
    ax.barh(y, numbers, height=0.4, color=PALETTE[0], zorder=3)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([str(item) for item in labels], fontsize=10.5, color=INK)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: render(v)))
    ax.grid(axis="x", color=GRID, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10.5, color=INK_SECONDARY, labelpad=10)

    largest = max(numbers, default=0.0) or 1.0
    ax.set_xlim(0, largest * 1.12)
    for index, value in enumerate(numbers):
        # Measured, not assumed: a label only goes inside the bar when the bar
        # is long enough to hold it with padding, otherwise it sits past the tip.
        inside = value > largest * 0.86
        ax.annotate(
            render(value),
            xy=(value, index),
            xytext=(-8 if inside else 8, 0),
            textcoords="offset points",
            va="center",
            ha="right" if inside else "left",
            fontsize=10,
            fontweight="bold",
            color=SURFACE if inside else INK,
            zorder=4,
        )

    return _finish(plt, fig, footer=footer)

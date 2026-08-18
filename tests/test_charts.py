"""Chart rendering and the per-turn collector.

Rendering is checked for the things that can silently go wrong -- an empty
series, a scale that flips mid-chart, a label that carries a secret -- not for
pixels. What the picture looks like is a review question; what it *contains* is
a test question.
"""

from __future__ import annotations

import struct

import pytest

from nrp_ops_agent import charts

DATES = ("2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04")


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height straight out of the IHDR chunk."""
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


class TestFormatter:
    def test_one_scale_for_the_whole_chart(self) -> None:
        # Per-value scaling put "15.8k" above "9,120", which reads as the
        # smaller number being larger.
        render = charts.formatter(184_320)
        assert render(184_320) == "184.3k"
        assert render(9_120) == "9.1k"
        assert render(0) == "0"

    def test_small_values_stay_unscaled(self) -> None:
        render = charts.formatter(870)
        assert render(870) == "870"
        assert render(1) == "1"

    @pytest.mark.parametrize(
        ("peak", "value", "expected"),
        [
            (5_000_000, 4_210_000, "4.2M"),
            (5_000_000_000, 1_500_000_000, "1.5B"),
            (2e12, 2e12, "2T"),
            (20_000, 21_000, "21k"),
        ],
    )
    def test_scales(self, peak: float, value: float, expected: str) -> None:
        assert charts.formatter(peak)(value) == expected

    def test_non_finite_is_a_dash_not_a_traceback(self) -> None:
        assert charts.formatter(10)(float("nan")) == "-"


class TestTickLabels:
    def test_the_last_point_is_always_a_tick(self) -> None:
        labels = [f"2026-07-{day:02d}" for day in range(1, 31)]
        positions, shown = charts._tick_labels(labels)
        assert positions[-1] == len(labels) - 1
        assert shown[-1] == "07-30"

    def test_the_year_is_dropped_when_it_never_changes(self) -> None:
        _, shown = charts._tick_labels(["2026-07-01", "2026-07-02"])
        assert shown == ["07-01", "07-02"]

    def test_the_year_is_kept_when_the_window_crosses_one(self) -> None:
        _, shown = charts._tick_labels(["2025-12-30", "2026-01-02"])
        assert shown == ["2025-12-30", "2026-01-02"]

    def test_no_points_is_not_an_error(self) -> None:
        assert charts._tick_labels([]) == ([], [])


class TestLegendOrder:
    def test_a_two_row_legend_reads_left_to_right(self) -> None:
        # Matplotlib fills columns first, so without this the second-largest
        # series sits under the largest instead of beside it.
        order = charts._legend_order(7, 4)
        assert order == [0, 4, 1, 5, 2, 6, 3]

    def test_a_full_grid_is_a_permutation(self) -> None:
        assert sorted(charts._legend_order(8, 4)) == list(range(8))


class TestRendering:
    def test_single_series_trend_is_a_png_of_the_expected_size(self) -> None:
        png = charts.render_trend(
            title="GPU hours for coder",
            subtitle="2026-07-01 to 2026-07-04",
            footer="NRP accounting",
            x_labels=DATES,
            series=[charts.Series(label="GPU hours", values=(1.0, 2.0, 3.0, 4.0))],
            y_label="GPU hours",
        )
        assert png_size(png) == (1400, 728)

    def test_stacked_trend_renders(self) -> None:
        series = [
            charts.Series(label=f"ns-{i}", values=tuple(float(i + j) for j in range(4)))
            for i in range(3)
        ]
        png = charts.render_trend(
            title="GPU hours by namespace",
            subtitle="",
            footer="",
            x_labels=DATES,
            series=series,
            y_label="GPU hours",
            stacked=True,
        )
        assert png.startswith(b"\x89PNG")

    def test_more_series_than_the_palette_does_not_cycle_a_colour(self) -> None:
        # A ninth series repeating slot one is two entities in one colour; the
        # tool folds the tail into "Other" before it gets here.
        assert charts.MAX_SERIES == len(charts.PALETTE)

    def test_ranking_height_grows_with_the_row_count(self) -> None:
        few = charts.render_ranking(
            title="Top namespaces", subtitle="", footer="", labels=["a", "b"], values=[2.0, 1.0]
        )
        many = charts.render_ranking(
            title="Top namespaces",
            subtitle="",
            footer="",
            labels=[f"ns-{i}" for i in range(12)],
            values=[float(12 - i) for i in range(12)],
        )
        assert png_size(many)[1] > png_size(few)[1]

    def test_flat_zero_series_does_not_divide_by_zero(self) -> None:
        png = charts.render_trend(
            title="GPU hours",
            subtitle="",
            footer="",
            x_labels=DATES,
            series=[charts.Series(label="GPU hours", values=(0.0, 0.0, 0.0, 0.0))],
            y_label="GPU hours",
        )
        assert png.startswith(b"\x89PNG")

    def test_a_single_day_renders(self) -> None:
        png = charts.render_trend(
            title="GPU hours",
            subtitle="",
            footer="",
            x_labels=("2026-07-01",),
            series=[charts.Series(label="GPU hours", values=(12.0,))],
            y_label="GPU hours",
        )
        assert png.startswith(b"\x89PNG")


class TestCollector:
    def test_charts_reach_the_collector(self) -> None:
        chart = charts.Chart(filename="a.png", title="t", alt_text="a", png=b"x")
        with charts.collecting(2) as collector:
            assert charts.emit(chart) is True
            assert charts.budget_remaining() == 1
        assert collector.charts == [chart]

    def test_the_budget_refuses_rather_than_dropping_silently(self) -> None:
        chart = charts.Chart(filename="a.png", title="t", alt_text="a", png=b"x")
        with charts.collecting(1) as collector:
            assert charts.emit(chart) is True
            assert charts.emit(chart) is False
        assert len(collector.charts) == 1

    def test_no_collector_is_not_an_error(self) -> None:
        # An MCP client calling the same dispatcher has nowhere to put an image.
        chart = charts.Chart(filename="a.png", title="t", alt_text="a", png=b"x")
        assert charts.emit(chart) is False
        assert charts.budget_remaining() == 0

    def test_collectors_do_not_leak_between_turns(self) -> None:
        chart = charts.Chart(filename="a.png", title="t", alt_text="a", png=b"x")
        with charts.collecting(2) as first:
            charts.emit(chart)
        with charts.collecting(2) as second:
            pass
        assert len(first.charts) == 1
        assert second.charts == []

    async def test_a_chart_drawn_in_a_gathered_task_is_collected(self) -> None:
        # Tool handlers run under asyncio.gather, which copies the context. The
        # collector object is shared, so appends still land.
        import asyncio

        chart = charts.Chart(filename="a.png", title="t", alt_text="a", png=b"x")

        async def draw() -> bool:
            return charts.emit(chart)

        with charts.collecting(2) as collector:
            results = await asyncio.gather(draw(), draw())
        assert results == [True, True]
        assert len(collector.charts) == 2


class TestLabelRedaction:
    def test_a_credential_in_a_label_never_reaches_the_renderer(self) -> None:
        # Rasterised text cannot be scrubbed on the way out of Slack, so the
        # scrub has to happen before the label is drawn.
        cleaned = charts.clean_label("ns-AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in cleaned
        assert "REDACTED" in cleaned

    def test_an_ordinary_namespace_is_untouched(self) -> None:
        assert charts.clean_label("jhub-datasci") == "jhub-datasci"

    def test_non_strings_are_coerced(self) -> None:
        assert charts.clean_label(2026) == "2026"

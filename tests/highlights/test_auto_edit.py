from buzz.highlights.auto_edit import (
    ALGORITHM_VERSION,
    select_auto_candidates,
    selection_summary,
)
from buzz.highlights.models import Candidate, HighlightConfig, VideoInfo
from buzz.highlights.subtitles import associate_subtitles, parse_srt
from buzz.highlights.windows import generate_candidates


def test_selection_summary_reports_transcript_signal_counts():
    cues, _ = parse_srt("1\n00:00:01,000 --> 00:00:02,000\ndialogue here")
    scored = Candidate("scored", 1_000, 3_000, 0, 3_000)
    rescued = Candidate("rescued", 1_000, 3_000, 0, 3_000, status="ignore", reasons=["static_scene"])
    silent = Candidate("silent", 30_000, 32_000, 0, 32_000)
    associate_subtitles(scored, cues)
    associate_subtitles(rescued, cues)

    summary = selection_summary([scored, rescued, silent], HighlightConfig(), 60_000)
    assert summary["algorithm"] == ALGORITHM_VERSION
    assert summary["transcript_candidate_count"] == 2
    assert summary["transcript_rescued_count"] == 1


def test_selection_summary_transcript_counts_are_zero_without_cues():
    summary = selection_summary(
        [Candidate("a", 0, 1_000, 0, 1_000)], HighlightConfig(), 60_000
    )
    assert summary["transcript_candidate_count"] == 0
    assert summary["transcript_rescued_count"] == 0


def test_subtitles_raise_a_candidates_score():
    with_text = Candidate("with", 1_000, 3_000, 0, 3_000, score=0.4)
    without_text = Candidate("without", 1_000, 3_000, 0, 3_000, score=0.4)
    cues, _ = parse_srt("1\n00:00:01,000 --> 00:00:03,000\n" + "あ" * 200)
    associate_subtitles(with_text, cues)
    assert with_text.score > without_text.score
    assert "transcript_density" in with_text.reasons
    assert "transcript_density" not in without_text.reasons


def test_budget_below_the_shortest_candidate_still_selects_a_clip():
    """A 10s clip cannot fit a 9s budget; returning nothing would be useless."""
    candidates = [Candidate("a", 0, 10_000, 0, 10_000, score=0.5)]
    selection = select_auto_candidates(
        candidates, HighlightConfig(target_duration_ratio=0.15), source_duration_ms=60_000
    )
    assert len(selection.selected) == 1
    assert selection.selected[0].id == "a"


def test_widened_budget_is_reported_and_not_exceeded():
    candidates = [Candidate("a", 0, 10_000, 0, 10_000, score=0.5)]
    selection = select_auto_candidates(
        candidates, HighlightConfig(target_duration_ratio=0.15), source_duration_ms=60_000
    )
    assert selection.budget_ms == 10_000
    assert selection.total_duration_ms == 10_000
    assert selection.total_duration_ms <= selection.budget_ms


def test_budget_widening_picks_the_highest_scoring_candidate():
    candidates = [
        Candidate("low", 0, 10_000, 0, 10_000, score=0.1),
        Candidate("high", 20_000, 30_000, 0, 30_000, score=0.9),
    ]
    selection = select_auto_candidates(
        candidates, HighlightConfig(target_duration_ratio=0.15), source_duration_ms=60_000
    )
    assert [candidate.id for candidate in selection.selected] == ["high"]


def test_adequate_budget_is_left_untouched():
    candidates = [
        Candidate("a", 0, 10_000, 0, 10_000, score=0.5),
        Candidate("b", 20_000, 30_000, 0, 30_000, score=0.5),
    ]
    selection = select_auto_candidates(
        candidates, HighlightConfig(target_duration_ratio=0.5), source_duration_ms=60_000
    )
    assert selection.budget_ms == 30_000
    assert len(selection.selected) == 2


def test_long_source_selection_is_unaffected_by_widening():
    """The widening must never engage when the budget already fits."""
    candidates = [
        Candidate(f"c{index}", index * 20_000, index * 20_000 + 20_000, 0,
                  index * 20_000 + 20_000, score=0.5)
        for index in range(30)
    ]
    selection = select_auto_candidates(
        candidates, HighlightConfig(target_duration_ratio=0.1), source_duration_ms=3_600_000
    )
    assert selection.budget_ms == 360_000
    assert selection.total_duration_ms <= selection.budget_ms


def test_short_sources_produce_a_reel_at_the_default_ratio():
    """Anything under ~35s used to fail outright with an empty selection."""
    for seconds in (6, 10, 20, 30):
        video = VideoInfo(
            duration_ms=seconds * 1000, width=320, height=240, fps=30,
            video_codec="h264", audio_codec="aac",
        )
        config = HighlightConfig(target_duration_ratio=0.3)
        candidates = generate_candidates(video, config, None, [], [])
        if not candidates:
            continue
        selection = select_auto_candidates(
            candidates, config, source_duration_ms=video.duration_ms
        )
        assert selection.selected, f"{seconds}s source produced an empty reel"


def test_no_candidates_still_returns_an_empty_selection():
    """Widening must not invent a clip when nothing is selectable."""
    selection = select_auto_candidates(
        [Candidate("a", 0, 10_000, 0, 10_000, score=0.5, status="ignore")],
        HighlightConfig(target_duration_ratio=0.15),
        source_duration_ms=60_000,
    )
    assert selection.selected == []

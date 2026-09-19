from buzz.highlights.auto_edit import ALGORITHM_VERSION, selection_summary
from buzz.highlights.models import Candidate, HighlightConfig
from buzz.highlights.subtitles import associate_subtitles, parse_srt


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

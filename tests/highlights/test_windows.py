from buzz.highlights.auto_edit import resolve_target_duration_seconds, select_auto_candidates
from buzz.highlights.models import Candidate, HighlightConfig, VideoInfo
from buzz.highlights.windows import (
    apply_motion_scores,
    combine_candidates,
    fixed_window_candidates,
    generate_candidates,
    parse_motion_samples,
    parse_scene_timestamps,
    score_candidate,
)


def test_fixed_windows_cover_short_tail():
    candidates = fixed_window_candidates(VideoInfo(61_000), HighlightConfig())
    assert candidates
    assert all(c.end_ms <= 61_000 for c in candidates)
    assert candidates[-1].end_ms == 61_000


def test_default_two_hour_candidate_count_is_bounded():
    candidates = generate_candidates(VideoInfo(2 * 60 * 60 * 1000), HighlightConfig(no_scene_detection=True))
    assert len(candidates) <= 200
    assert candidates[0].id == "candidate-0001"


def test_auto_candidates_include_short_scale_for_brief_moments():
    candidates = generate_candidates(
        VideoInfo(60_000),
        HighlightConfig(auto_edit=True, no_scene_detection=True, max_candidates=0),
    )
    assert any("short_window" in candidate.reasons for candidate in candidates)
    assert any(candidate.duration_ms <= 10_000 for candidate in candidates)


def test_scene_parser_deduplicates():
    assert parse_scene_timestamps("pts_time:1.5 pts_time:1.5 pts_time:-1 scene_time=3") == [1500, 3000]


def test_motion_parser_reads_metadata_samples():
    output = """frame:0 pts_time:1\n[Parsed_metadata] lavfi.signalstats.YAVG=0.5\nframe:1 pts_time:2\n[Parsed_metadata] lavfi.signalstats.YAVG=8.0\n"""
    assert parse_motion_samples(output) == [(1000, 0.5), (2000, 8.0)]


def test_motion_scores_rescue_brief_high_peak():
    config = HighlightConfig(ignore_static_scenes=True, static_motion_threshold=1.5)
    candidate = Candidate("reaction", 0, 5_000, 0, 5_000)
    samples = [(index * 500, 0.2) for index in range(10)] + [(4_500, 8.0)]
    apply_motion_scores([candidate], samples, config)
    assert candidate.status == "unprocessed"
    assert "static_scene" not in candidate.reasons


def test_motion_scores_flag_static_candidates():
    config = HighlightConfig(ignore_static_scenes=True, static_motion_threshold=1.5)
    candidates = fixed_window_candidates(VideoInfo(40_000), HighlightConfig(
        ignore_static_scenes=True, static_motion_threshold=1.5, stride_seconds=20.0
    ))
    result = apply_motion_scores(candidates, [(0, 0.2), (10_000, 0.2), (20_000, 8.0), (30_000, 8.0)], config)
    assert result[0].status == "ignore"
    assert "static_scene" in result[0].reasons
    assert result[1].status == "unprocessed"
    assert result[1].score > result[0].score


def test_motion_scores_are_optional_when_analysis_has_no_samples():
    config = HighlightConfig(ignore_static_scenes=True)
    candidates = generate_candidates(VideoInfo(20_000), config, motion_samples=[])
    assert candidates
    assert all(candidate.status == "unprocessed" for candidate in candidates)


def test_auto_duration_uses_source_ratio():
    assert resolve_target_duration_seconds(3_600_000) == 1080
    assert resolve_target_duration_seconds(7_200_000) == 2160
    assert resolve_target_duration_seconds(3_600_000, configured_ratio=0.5) == 1800
    assert resolve_target_duration_seconds(3_600_000, 0.5) == 0.5
    assert resolve_target_duration_seconds(3_600_000, 1.5) == 1.5
    assert resolve_target_duration_seconds(3_600_000, configured_seconds=300) == 300
    assert resolve_target_duration_seconds(3_600_000, configured_seconds=0) == 1080
    assert resolve_target_duration_seconds(3_600_000, configured_ratio=0.3, configured_seconds=300) == 300


def test_auto_selection_uses_full_source_duration_for_ratio():
    candidates = [Candidate("early", 0, 10_000, 0, 10_000, score=1.0)]
    result = select_auto_candidates(
        candidates,
        HighlightConfig(target_duration_ratio=0.5),
        source_duration_ms=100_000,
    )
    assert result.total_duration_ms == 50_000
    assert result.budget_ms == 10_000


def test_auto_selection_prefers_total_score_over_longer_low_score_clip():
    candidates = [
        Candidate("long", 0, 10_000, 0, 10_000, score=0.9),
        Candidate("first", 0, 5_000, 0, 5_000, score=0.6),
        Candidate("second", 5_000, 10_000, 5_000, 10_000, score=0.6),
    ]
    result = select_auto_candidates(
        candidates,
        HighlightConfig(target_duration_ratio=1.0, target_duration_seconds=10),
    )
    assert [candidate.id for candidate in result.selected] == ["first", "second"]


def test_auto_selection_respects_budget_and_avoids_overlap():
    candidates = [
        Candidate("long", 0, 20_000, 0, 20_000, score=1.0),
        Candidate("short-a", 0, 5_000, 0, 5_000, score=0.7),
        Candidate("short-b", 5_000, 10_000, 5_000, 10_000, score=0.7),
    ]
    result = select_auto_candidates(
        candidates,
        HighlightConfig(target_duration_ratio=1.0, target_duration_seconds=10, max_auto_clips=2),
    )
    assert [candidate.id for candidate in result.selected] == ["short-a", "short-b"]
    assert result.total_duration_ms == 10_000


def test_auto_selection_allows_adjacent_clips():
    candidates = [
        Candidate("a", 0, 5_000, 0, 5_000, score=0.5),
        Candidate("b", 5_000, 10_000, 5_000, 10_000, score=0.5),
    ]
    result = select_auto_candidates(candidates, HighlightConfig(target_duration_ratio=1.0, target_duration_seconds=10))
    assert [candidate.id for candidate in result.selected] == ["a", "b"]


def test_auto_selection_has_no_default_clip_count_limit():
    candidates = [
        Candidate(str(index), index * 5_000, (index + 1) * 5_000, index * 5_000, (index + 1) * 5_000, score=0.5)
        for index in range(4)
    ]
    result = select_auto_candidates(candidates, HighlightConfig(target_duration_ratio=1.0, target_duration_seconds=20))
    assert len(result.selected) == 4
    assert result.total_duration_ms == 20_000


def test_auto_selection_skips_ignored_candidates():
    candidates = [Candidate("ignored", 0, 5_000, 0, 5_000, score=1.0, status="ignore")]
    result = select_auto_candidates(candidates, HighlightConfig(target_duration_ratio=1.0, target_duration_seconds=10))
    assert result.selected == []


def test_score_weights():
    assert score_candidate(1, 1, 1, 1) == 1


def test_combine_reasons_and_ids():
    video = VideoInfo(30_000)
    candidates = fixed_window_candidates(video, HighlightConfig(window_seconds=10, stride_seconds=10))
    result = combine_candidates(candidates, video)
    assert [c.id for c in result] == [f"candidate-{i:04d}" for i in range(1, len(result) + 1)]

from buzz.highlights.models import HighlightConfig, VideoInfo
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


def test_scene_parser_deduplicates():
    assert parse_scene_timestamps("pts_time:1.5 pts_time:1.5 pts_time:-1 scene_time=3") == [1500, 3000]


def test_motion_parser_reads_metadata_samples():
    output = """frame:0 pts_time:1\n[Parsed_metadata] lavfi.signalstats.YAVG=0.5\nframe:1 pts_time:2\n[Parsed_metadata] lavfi.signalstats.YAVG=8.0\n"""
    assert parse_motion_samples(output) == [(1000, 0.5), (2000, 8.0)]


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


def test_score_weights():
    assert score_candidate(1, 1, 1, 1) == 1


def test_combine_reasons_and_ids():
    video = VideoInfo(30_000)
    candidates = fixed_window_candidates(video, HighlightConfig(window_seconds=10, stride_seconds=10))
    result = combine_candidates(candidates, video)
    assert [c.id for c in result] == [f"candidate-{i:04d}" for i in range(1, len(result) + 1)]

from buzz.highlights.models import HighlightConfig, VideoInfo
from buzz.highlights.windows import (
    combine_candidates,
    fixed_window_candidates,
    generate_candidates,
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


def test_score_weights():
    assert score_candidate(1, 1, 1, 1) == 1


def test_combine_reasons_and_ids():
    video = VideoInfo(30_000)
    candidates = fixed_window_candidates(video, HighlightConfig(window_seconds=10, stride_seconds=10))
    result = combine_candidates(candidates, video)
    assert [c.id for c in result] == [f"candidate-{i:04d}" for i in range(1, len(result) + 1)]

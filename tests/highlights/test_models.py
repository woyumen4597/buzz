import pytest

from buzz.highlights.models import Candidate, HighlightConfig, VideoInfo, format_timestamp, interval_iou


def test_timestamp_formatting():
    assert format_timestamp(-1) == "00:00:00.000"
    assert format_timestamp(3_600_001) == "01:00:00.001"
    assert format_timestamp(86_400_001) == "24:00:00.001"
    assert format_timestamp(1234, ",") == "00:00:01,234"


def test_config_validation():
    assert HighlightConfig().max_candidates == 200
    assert HighlightConfig().target_duration_ratio == 0.3
    with pytest.raises(ValueError):
        HighlightConfig(target_duration_ratio=1.1)
    assert HighlightConfig(target_duration_seconds=300).target_duration_seconds == 300
    with pytest.raises(ValueError):
        HighlightConfig(scene_threshold=1.1)
    with pytest.raises(ValueError):
        HighlightConfig(max_duration_seconds=2, min_duration_seconds=3)


def test_candidate_round_trip():
    candidate = Candidate("x", 0, 5000, 0, 5000, reasons=["fixed_window"])
    assert Candidate.from_dict(candidate.to_dict()).to_dict() == candidate.to_dict()
    with pytest.raises(ValueError):
        Candidate("x", 10, 10, 0, 10)


def test_iou_touching_intervals_is_zero():
    assert interval_iou(0, 10, 10, 20) == 0


def test_video_info_rejects_negative_duration():
    with pytest.raises(ValueError):
        VideoInfo(-1)

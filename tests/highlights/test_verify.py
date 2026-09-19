"""Tests for the fast highlight-reel validator.

These favour pure parsing and synthetic fixtures over real encodes: the point
of the validator is to be cheap, so its tests should be cheap too.
"""

import subprocess

import pytest

from buzz.highlights.verify import (
    VerifyReport,
    _parse_srt_text,
    check_container,
    check_durations,
    check_start_time,
    probe_streams,
    sample_windows,
    summarise,
    verify_reel,
)


# --------------------------------------------------------------------------
# Raw cue parsing: validation must SEE zero-length cues, not drop them
# --------------------------------------------------------------------------


def test_parse_srt_text_keeps_zero_length_cues():
    content = (
        "1\n00:00:01,000 --> 00:00:03,000\nVISIBLE\n\n"
        "2\n00:00:05,000 --> 00:00:05,000\nZERO LENGTH\n"
    )
    cues = _parse_srt_text(content)
    assert len(cues) == 2
    zero = [cue for cue in cues if cue.end_ms <= cue.start_ms]
    assert len(zero) == 1
    assert zero[0].text == "ZERO LENGTH"


def test_parse_srt_text_handles_missing_sequence_numbers():
    content = "00:00:01,000 --> 00:00:02,000\nNO INDEX\n"
    cues = _parse_srt_text(content)
    assert len(cues) == 1
    assert cues[0].text == "NO INDEX"


def test_parse_srt_text_ignores_garbage_blocks():
    content = "not a cue\n\n1\n00:00:01,000 --> 00:00:02,000\nOK\n"
    cues = _parse_srt_text(content)
    assert [cue.text for cue in cues] == ["OK"]


# --------------------------------------------------------------------------
# Container checks
# --------------------------------------------------------------------------


def _probe(**overrides):
    probe = {
        "format": {"duration": "60.0", "start_time": "0.0"},
        "streams": [
            {
                "codec_type": "video", "codec_name": "h264",
                "width": 1920, "height": 1080, "duration": "60.0",
            },
            {
                "codec_type": "audio", "codec_name": "aac",
                "sample_rate": "48000", "channels": 2, "duration": "60.0",
            },
        ],
    }
    probe.update(overrides)
    return probe


def test_container_accepts_a_normal_reel():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    check_container(report, _probe())
    assert report.find("video_stream").severity == "ok"
    assert report.find("audio_stream").severity == "ok"


def test_container_fails_without_video():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    probe = _probe()
    probe["streams"] = [s for s in probe["streams"] if s["codec_type"] != "video"]
    check_container(report, probe)
    assert report.find("video_stream").severity == "error"
    assert not report.ok


def test_container_warns_when_audio_is_missing():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    probe = _probe()
    probe["streams"] = [s for s in probe["streams"] if s["codec_type"] != "audio"]
    check_container(report, probe)
    # A silent source legitimately yields a silent reel, so this is not fatal.
    assert report.find("audio_stream").severity == "warning"
    assert report.ok


def test_container_detects_text_subtitle_track():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    probe = _probe()
    probe["streams"].append(
        {"codec_type": "subtitle", "codec_name": "mov_text", "nb_frames": "42"}
    )
    check_container(report, probe)
    assert report.find("subtitle_stream").severity == "ok"


def test_container_warns_without_subtitles():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    check_container(report, _probe())
    assert report.find("subtitle_stream").severity == "warning"


def test_container_ignores_non_text_subtitle_tracks():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    probe = _probe()
    probe["streams"].append(
        {"codec_type": "subtitle", "codec_name": "dvd_subtitle"}
    )
    check_container(report, probe)
    assert report.find("subtitle_stream").severity == "warning"


# --------------------------------------------------------------------------
# Duration / drift checks
# --------------------------------------------------------------------------


def test_durations_flags_large_av_drift():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    probe = _probe()
    probe["streams"][1]["duration"] = "75.0"
    check_durations(report, probe)
    check = report.find("av_drift")
    assert check.severity == "error"
    assert "audio" in check.detail


def test_durations_accepts_small_drift():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    probe = _probe()
    probe["streams"][1]["duration"] = "60.4"
    check_durations(report, probe)
    assert report.find("av_drift").severity == "ok"


def test_durations_fails_without_container_duration():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    check_durations(report, _probe(format={"duration": "0"}))
    assert report.find("duration").severity == "error"


def test_durations_warns_about_a_long_container_tail():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    probe = _probe()
    probe["format"]["duration"] = "90.0"
    check_durations(report, probe)
    assert report.find("tail").severity == "warning"


def test_start_time_warns_when_non_zero():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    check_start_time(report, _probe(format={"duration": "60", "start_time": "2.5"}))
    assert report.find("start_time").severity == "warning"


def test_start_time_ok_when_zero():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    check_start_time(report, _probe())
    assert report.find("start_time").severity == "ok"


# --------------------------------------------------------------------------
# Sampling plan
# --------------------------------------------------------------------------


def test_sample_windows_are_inside_the_reel():
    windows = sample_windows(600.0, count=8, window_s=6.0)
    assert len(windows) == 8
    for start, length in windows:
        assert start >= 0
        assert start + length <= 600.0 + 1e-6


def test_sample_windows_collapse_for_a_short_reel():
    windows = sample_windows(5.0, count=8, window_s=6.0)
    assert windows == [(0.0, 5.0)]


def test_sample_windows_empty_for_zero_duration():
    assert sample_windows(0.0) == []


# --------------------------------------------------------------------------
# Report helpers
# --------------------------------------------------------------------------


def test_report_ok_ignores_warnings_but_not_errors():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    report.add("a", "warning", "w")
    assert report.ok
    report.add("b", "error", "e")
    assert not report.ok


def test_report_to_dict_round_trips_checks():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    report.add("thing", "ok", "fine", count=3)
    payload = report.to_dict()
    assert payload["ok"] is True
    assert payload["checks"][0]["data"] == {"count": 3}


def test_summarise_marks_each_severity():
    report = VerifyReport(path=None)  # type: ignore[arg-type]
    report.add("a", "error", "bad")
    report.add("b", "warning", "meh")
    report.add("c", "ok", "good")
    text = summarise(report)
    assert "[FAIL]" in text and "[WARN]" in text and "[ok" in text


# --------------------------------------------------------------------------
# End-to-end against real FFmpeg on a tiny synthetic reel
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_reel(tmp_path_factory):
    """Build a 6s reel with audio and a subtitle track using FFmpeg."""
    from buzz.highlights.media import find_tools

    ffmpeg, _ = find_tools()
    target = tmp_path_factory.mktemp("reel") / "reel.mp4"
    subtitle = target.parent / "in.srt"
    subtitle.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHELLO\n", encoding="utf-8"
    )
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=s=160x120:d=6:r=15",
            "-f", "lavfi", "-i", "sine=frequency=440:d=6",
            "-f", "srt", "-i", str(subtitle),
            "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            "-c:s", "mov_text", "-t", "6", str(target),
        ],
        check=True, capture_output=True,
    )
    return target


def test_verify_reel_passes_a_good_reel(small_reel):
    report = verify_reel(small_reel, sample_count=2, sample_window_s=2.0)
    assert report.find("video_stream").severity == "ok"
    assert report.find("audio_stream").severity == "ok"
    assert report.find("subtitle_stream").severity == "ok"
    # The cue must survive the round trip and stay inside the reel.
    assert report.find("subtitle_bounds").severity == "ok"
    assert not report.errors


def test_verify_reel_reports_a_missing_file(tmp_path):
    report = verify_reel(tmp_path / "nope.mp4")
    assert report.find("exists").severity == "error"
    assert not report.ok


def test_verify_reel_reports_an_empty_file(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    report = verify_reel(empty)
    assert report.find("exists").severity == "error"


def test_verify_reel_reports_an_unreadable_file(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is definitely not a video")
    report = verify_reel(junk)
    assert report.find("readable").severity == "error"
    assert not report.ok


def test_probe_streams_raises_on_unreadable_input(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"nope")
    from buzz.highlights.media import find_tools

    _, ffprobe = find_tools()
    with pytest.raises(RuntimeError):
        probe_streams(ffprobe, str(junk))


def test_verify_reel_can_skip_video_sampling(small_reel):
    report = verify_reel(small_reel, check_video=False)
    assert report.find("video_samples") is None
    assert report.find("subtitles") is not None

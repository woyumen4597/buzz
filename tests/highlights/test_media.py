import threading
from pathlib import Path

from buzz.highlights.media import (
    clip_command,
    concat_command,
    extract_subtitles_command,
    render_selected_video,
    video_encoder_args,
)
from buzz.highlights.models import Candidate
from buzz.highlights.output import automatic_output_path
from buzz.highlights.subtitles import Subtitle


def test_automatic_output_path_uses_source_directory_and_suffix():
    assert automatic_output_path("/tmp/a video.mp4") == Path("/tmp/a video_highlight.mp4")
    assert automatic_output_path("/tmp/source.mov") == Path("/tmp/source_highlight.mp4")


def test_commands_keep_paths_as_single_args():
    path = "/tmp/a video 中文.mp4"
    candidate = Candidate("only", 1_000, 3_000, 1_000, 3_000)
    command = clip_command("ffmpeg", path, candidate, "/tmp/out x.mp4")
    assert path in command


def test_clip_command_uses_high_quality_encoding_and_preserves_subtitles(monkeypatch):
    monkeypatch.setattr("buzz.highlights.media.video_encoder_args", lambda _ffmpeg: ["-c:v", "libx264"])
    candidate = Candidate("only", 1_000, 3_000, 1_000, 3_000)
    command = clip_command("ffmpeg", "in.mp4", candidate, "out.mp4")
    assert command[command.index("-map"):command.index("-c:v")] == [
        "-map", "0:v:0", "-map", "0:a:0?", "-map", "0:s?",
    ]
    assert command[command.index("-c:v"):command.index("-c:a")] == [
        "-c:v", "libx264",
    ]
    assert command[command.index("-c:a"):command.index("-c:s")] == [
        "-c:a", "aac", "-b:a", "192k",
    ]
    assert command[command.index("-c:s"):command.index("out.mp4")] == [
        "-c:s", "mov_text",
    ]


def test_clip_command_without_audio_does_not_map_audio():
    candidate = Candidate("only", 0, 1_000, 0, 1_000)
    command = clip_command("ffmpeg", "in.mp4", candidate, "out.mp4", has_audio=False)
    assert "0:a:0?" not in command
    assert "-an" not in command
    assert "-map" in command and "0:s?" in command


def test_clip_command_with_retimed_srt_muxes_the_srt_input():
    candidate = Candidate("only", 1_000, 3_000, 1_000, 3_000)
    command = clip_command(
        "ffmpeg", "in.mp4", candidate, "out.mp4", subtitle_path="clip.srt"
    )
    # The retimed SRT is a second input, so its stream is mapped as 1:0.
    assert "clip.srt" in command
    assert command[command.index("clip.srt") - 3:command.index("clip.srt")] == ["-f", "srt", "-i"]
    assert "1:0" in command
    assert "0:s?" not in command
    # -t must be an output option, i.e. after the subtitle input.
    assert command.index("-t") > command.index("clip.srt")
    assert command[-1] == "out.mp4"


def test_clip_command_without_retimed_srt_still_maps_source_subtitles():
    candidate = Candidate("only", 0, 1_000, 0, 1_000)
    command = clip_command("ffmpeg", "in.mp4", candidate, "out.mp4")
    assert "0:s?" in command
    assert "1:0" not in command


def test_video_encoder_args_prefers_hardware_on_macos(monkeypatch):
    monkeypatch.setattr("buzz.highlights.media.sys.platform", "darwin")
    monkeypatch.setattr("buzz.highlights.media._encoder_available", lambda *_: True)
    monkeypatch.setattr("buzz.highlights.media._ENCODER_CACHE", {})
    args = video_encoder_args("/usr/bin/ffmpeg")
    assert args[:2] == ["-c:v", "h264_videotoolbox"]


def test_video_encoder_args_falls_back_to_software(monkeypatch):
    monkeypatch.setattr("buzz.highlights.media.sys.platform", "darwin")
    monkeypatch.setattr("buzz.highlights.media._encoder_available", lambda *_: False)
    monkeypatch.setattr("buzz.highlights.media._ENCODER_CACHE", {})
    args = video_encoder_args("/usr/bin/ffmpeg")
    assert args[:2] == ["-c:v", "libx264"]
    assert "-preset" in args and "veryfast" in args
    # libx264 regresses past the performance-core count on Apple silicon.
    assert args[args.index("-threads") + 1] == "4"


def test_video_encoder_args_are_memoized_per_binary(monkeypatch):
    calls = []

    def counting_probe(*_args):
        calls.append(True)
        return True

    monkeypatch.setattr("buzz.highlights.media.sys.platform", "darwin")
    monkeypatch.setattr("buzz.highlights.media._encoder_available", counting_probe)
    monkeypatch.setattr("buzz.highlights.media._ENCODER_CACHE", {})
    video_encoder_args("/usr/bin/ffmpeg")
    video_encoder_args("/usr/bin/ffmpeg")
    assert len(calls) == 1


def test_render_selected_video_parallelizes_and_keeps_timeline_order(tmp_path, monkeypatch):
    """Concurrent encodes must still concatenate in timeline order."""
    candidates = [
        Candidate(str(i), i * 2_000, i * 2_000 + 1_000, i * 2_000, i * 2_000 + 1_000,
                  selected=True, status="keep")
        for i in range(4)
    ]
    seen = []
    lock = threading.Lock()

    def fake_run(command, output_path):
        if output_path.suffix == ".mp4" and output_path.name.startswith("clip_"):
            with lock:
                seen.append(output_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ok")

    monkeypatch.setattr("buzz.highlights.media._run_atomic", fake_run)
    render_selected_video("ffmpeg", "input.mp4", candidates, tmp_path)

    # The four clips are named by sorted index regardless of encode order.
    assert sorted(seen) == [f"clip_{i:04d}.mp4" for i in range(1, 5)]


def test_extract_subtitles_command_maps_first_subtitle_track():
    command = extract_subtitles_command("ffmpeg", "in.mp4", "out.srt")
    assert command == [
        "ffmpeg", "-y", "-i", "in.mp4", "-map", "0:s:0", "-f", "srt", "out.srt",
    ]


def test_concat_command_uses_concat_demuxer():
    command = concat_command("ffmpeg", "/tmp/concat list.txt", "/tmp/highlights.mp4")
    assert command == [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "/tmp/concat list.txt",
        "-map", "0:v:0", "-map", "0:a:0?", "-map", "0:s?",
        "-c", "copy", "-movflags", "+faststart", "/tmp/highlights.mp4",
    ]


def test_render_selected_video_rejects_overlapping_selection(tmp_path):
    candidates = [
        Candidate("a", 0, 2_000, 0, 2_000, selected=True, status="keep"),
        Candidate("b", 1_000, 3_000, 1_000, 3_000, selected=True, status="keep"),
    ]
    try:
        render_selected_video("ffmpeg", "input.mp4", candidates, tmp_path)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping candidates should be rejected")


def test_render_selected_video_sorts_by_timeline(tmp_path, monkeypatch):
    candidates = [
        Candidate("late", 2_000, 3_000, 2_000, 3_000, selected=True, status="keep"),
        Candidate("early", 0, 1_000, 0, 1_000, selected=True, status="keep"),
    ]
    commands = []

    def fake_run(command, output_path):
        commands.append(command)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ok")

    monkeypatch.setattr("buzz.highlights.media._run_atomic", fake_run)
    output = render_selected_video("ffmpeg", "input.mp4", candidates, tmp_path)
    assert output == tmp_path / "highlights.mp4"
    assert commands[0][3] == "0.000"
    assert commands[1][3] == "2.000"
    assert commands[2][2:6] == ["-f", "concat", "-safe", "0"]


def test_render_selected_video_supports_final_output_path(tmp_path, monkeypatch):
    candidate = Candidate("only", 0, 1_000, 0, 1_000, selected=True, status="keep")
    commands = []

    def fake_run(command, output_path):
        commands.append(command)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ok")

    monkeypatch.setattr("buzz.highlights.media._run_atomic", fake_run)
    final_path = tmp_path / "source_highlight.mp4"
    output = render_selected_video(
        "ffmpeg", "input.mp4", [candidate], tmp_path / "work", output_path=final_path
    )
    assert output == final_path
    assert commands[-1][-1] == str(final_path)


def test_render_selected_video_reuses_caller_supplied_subtitles(tmp_path, monkeypatch):
    """Ranking already extracted the track, so rendering must not extract again."""
    candidate = Candidate("only", 0, 5_000, 0, 5_000, selected=True, status="keep")
    commands = []

    def fake_run(command, output_path):
        commands.append(command)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ok")

    def explode(*_args, **_kwargs):
        raise AssertionError("render must not re-extract subtitles")

    monkeypatch.setattr("buzz.highlights.media._run_atomic", fake_run)
    monkeypatch.setattr("buzz.highlights.media._load_source_subtitles", explode)
    render_selected_video(
        "ffmpeg",
        "input.mp4",
        [candidate],
        tmp_path,
        source_subtitles=[Subtitle(1_000, 2_000, "cue")],
    )
    clip_command = commands[0]
    assert "-f" in clip_command and "srt" in clip_command


def test_render_selected_video_extracts_subtitles_when_not_supplied(tmp_path, monkeypatch):
    candidate = Candidate("only", 0, 5_000, 0, 5_000, selected=True, status="keep")
    commands = []
    calls = []

    def fake_run(command, output_path):
        commands.append(command)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ok")

    def fake_load(_ffmpeg, _video, _output_dir):
        calls.append(True)
        return [Subtitle(1_000, 2_000, "cue")]

    monkeypatch.setattr("buzz.highlights.media._run_atomic", fake_run)
    monkeypatch.setattr("buzz.highlights.media._load_source_subtitles", fake_load)
    render_selected_video("ffmpeg", "input.mp4", [candidate], tmp_path)
    assert calls == [True]

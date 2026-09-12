from buzz.highlights.media import concat_command, gif_command, preview_command, render_selected_video, thumbnail_command
from buzz.highlights.models import Candidate


def test_commands_keep_paths_as_single_args():
    path = "/tmp/a video 中文.mp4"
    assert path in thumbnail_command("ffmpeg", path, 1500, "/tmp/out x.jpg")
    command = preview_command("ffmpeg", path, 1000, 5000, "/tmp/out.mp4", has_audio=False)
    assert "-an" in command
    assert path in command
    assert "-t" in command and "4.000" in command


def test_preview_command_without_audio_uses_an():
    command = preview_command("ffmpeg", "in.mp4", 0, 1000, "out.mp4", has_audio=False)
    assert "-an" in command
    assert "-c:a" not in command


def test_gif_command_limits_duration():
    command = gif_command("ffmpeg", "in.mp4", 0, 20_000, "out.gif")
    assert "12.000" in command


def test_concat_command_uses_concat_demuxer():
    command = concat_command("ffmpeg", "/tmp/concat list.txt", "/tmp/highlights.mp4")
    assert command == [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "/tmp/concat list.txt",
        "-c", "copy", "-movflags", "+faststart", "/tmp/highlights.mp4",
    ]


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

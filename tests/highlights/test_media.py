from buzz.highlights.media import gif_command, preview_command, thumbnail_command


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

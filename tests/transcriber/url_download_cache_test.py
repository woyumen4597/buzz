import os
import time

import pytest

from buzz.transcriber.file_transcriber import (
    cleanup_download_cache,
    download_video_to_cache,
    find_cached_video,
)


class TestUrlDownloadCache:
    def test_cleanup_removes_old_entries_keeps_fresh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "buzz.transcriber.file_transcriber._download_cache_dir",
            lambda: str(tmp_path),
        )
        old = tmp_path / "old"
        old.mkdir()
        (old / "audio.wav").write_bytes(b"x")
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        (fresh / "audio.wav").write_bytes(b"x")
        old_time = time.time() - 31 * 24 * 60 * 60
        os.utime(old, (old_time, old_time))
        os.utime(old / "audio.wav", (old_time, old_time))

        assert cleanup_download_cache(max_age_days=30) == 1
        assert not old.exists()
        assert fresh.exists()

    def test_find_cached_video(self, tmp_path):
        wav = tmp_path / "title.wav"
        wav.write_bytes(b"x")
        assert find_cached_video(str(wav)) is None

        video = tmp_path / "title.mp4"
        video.write_bytes(b"x")
        assert find_cached_video(str(wav)) == str(video)

    def test_download_video_to_cache_uses_existing(self, tmp_path):
        wav = tmp_path / "title.wav"
        wav.write_bytes(b"x")
        video = tmp_path / "title.mp4"
        video.write_bytes(b"x")

        assert (
            download_video_to_cache("https://example.com/v", str(wav))
            == str(video)
        )

    def test_download_video_to_cache_downloads(self, tmp_path, monkeypatch):
        wav = tmp_path / "title.wav"
        wav.write_bytes(b"x")

        class FakeYDL:
            def __init__(self, options):
                assert options["format"] == "bestvideo+bestaudio/best"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def download(self, urls):
                assert urls == ["https://example.com/v"]
                # yt-dlp writes the merged mp4 next to the audio file
                (tmp_path / "title.mp4").write_bytes(b"video")

        monkeypatch.setattr("buzz.transcriber.file_transcriber.YoutubeDL", FakeYDL)

        assert download_video_to_cache("https://example.com/v", str(wav)) == str(
            tmp_path / "title.mp4"
        )

    def test_download_video_to_cache_raises_when_no_file(self, tmp_path, monkeypatch):
        wav = tmp_path / "title.wav"
        wav.write_bytes(b"x")

        class FakeYDL:
            def __init__(self, options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def download(self, urls):
                pass

        monkeypatch.setattr("buzz.transcriber.file_transcriber.YoutubeDL", FakeYDL)

        with pytest.raises(Exception, match="no playable file"):
            download_video_to_cache("https://example.com/v", str(wav))

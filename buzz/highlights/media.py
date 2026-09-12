"""FFmpeg/FFprobe discovery, command construction and media artifact generation."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from buzz.ffmpeg_video_player import _find_ffmpeg, _find_ffprobe, probe_video

from .models import Candidate, VideoInfo, format_timestamp

LOG = logging.getLogger(__name__)


def find_tools() -> tuple[str, str]:
    ffmpeg = _find_ffmpeg()
    ffprobe = _find_ffprobe()
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).exists():
        raise FileNotFoundError("FFmpeg not found; install ffmpeg and ensure it is on PATH")
    if shutil.which(ffprobe) is None and not Path(ffprobe).exists():
        raise FileNotFoundError("FFprobe not found; install ffmpeg and ensure it is on PATH")
    return ffmpeg, ffprobe


def probe_media(video_path: str) -> VideoInfo:
    info = probe_video(video_path)
    result = VideoInfo.from_probe(info)
    if not result.has_video or result.duration_ms <= 0:
        raise ValueError(f"input is not a readable video: {video_path}")
    return result


def _seconds(ms: int) -> str:
    return f"{max(0, ms) / 1000:.3f}"


def thumbnail_command(ffmpeg: str, video_path: str, timestamp_ms: int, output_path: str) -> list[str]:
    return [
        ffmpeg, "-y", "-ss", _seconds(timestamp_ms), "-i", video_path,
        "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "3", output_path,
    ]


def preview_command(
    ffmpeg: str,
    video_path: str,
    start_ms: int,
    end_ms: int,
    output_path: str,
    has_audio: bool = True,
) -> list[str]:
    return [
        ffmpeg, "-y", "-ss", _seconds(start_ms), "-i", video_path,
        "-t", _seconds(max(0, end_ms - start_ms)),
        "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "30", *( ["-c:a", "aac", "-b:a", "96k"] if has_audio else ["-an"] ),
        "-movflags", "+faststart", output_path,
    ]


def gif_command(
    ffmpeg: str, video_path: str, start_ms: int, end_ms: int, output_path: str
) -> list[str]:
    duration = min(12_000, max(0, end_ms - start_ms))
    return [
        ffmpeg, "-y", "-ss", _seconds(start_ms), "-i", video_path, "-t", _seconds(duration),
        "-vf", "fps=8,scale=480:-2:flags=lanczos", output_path,
    ]


def _run_atomic(command: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-", suffix=output_path.suffix, dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        # FFmpeg needs a path it can create; remove the empty mkstemp placeholder.
        temp_path.unlink()
        subprocess.run(command[:-1] + [str(temp_path)], check=True, capture_output=True, text=True)
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _error_summary(exc: subprocess.CalledProcessError) -> str:
    output = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
    return output[-1][-500:] if output else str(exc)


def generate_thumbnail(
    candidate: Candidate, video_path: str, output_path: Path, ffmpeg: str, keep_existing: bool = False
) -> None:
    if keep_existing and output_path.exists():
        candidate.thumbnail = output_path.as_posix()
        return
    timestamp = candidate.start_ms + round(candidate.duration_ms * 0.5)
    try:
        _run_atomic(thumbnail_command(ffmpeg, video_path, timestamp, str(output_path)), output_path)
        candidate.thumbnail = output_path.as_posix()
    except subprocess.CalledProcessError as exc:
        candidate.errors.append(f"thumbnail: {_error_summary(exc)}")
        LOG.warning("Thumbnail failed for %s: %s", candidate.id, candidate.errors[-1])


def generate_preview(
    candidate: Candidate,
    video_path: str,
    output_path: Path,
    ffmpeg: str,
    keep_existing: bool = False,
    has_audio: bool = True,
) -> None:
    if keep_existing and output_path.exists():
        candidate.preview = output_path.as_posix()
        return
    try:
        _run_atomic(
            preview_command(
                ffmpeg,
                video_path,
                candidate.preview_start_ms,
                candidate.preview_end_ms,
                str(output_path),
                has_audio=has_audio,
            ),
            output_path,
        )
        candidate.preview = output_path.as_posix()
    except subprocess.CalledProcessError as exc:
        candidate.errors.append(f"preview: {_error_summary(exc)}")
        LOG.warning("Preview failed for %s: %s", candidate.id, candidate.errors[-1])


def generate_gif(
    candidate: Candidate, video_path: str, output_path: Path, ffmpeg: str, keep_existing: bool = False
) -> None:
    if keep_existing and output_path.exists():
        candidate.gif = output_path.as_posix()
        return
    try:
        _run_atomic(
            gif_command(ffmpeg, video_path, candidate.preview_start_ms, candidate.preview_end_ms, str(output_path)),
            output_path,
        )
        candidate.gif = output_path.as_posix()
    except subprocess.CalledProcessError as exc:
        candidate.errors.append(f"gif: {_error_summary(exc)}")
        LOG.warning("GIF failed for %s: %s", candidate.id, candidate.errors[-1])


def clip_command(
    ffmpeg: str,
    video_path: str,
    candidate: Candidate,
    output_path: str,
) -> list[str]:
    return [
        ffmpeg, "-y", "-ss", _seconds(candidate.start_ms), "-i", video_path,
        "-t", _seconds(candidate.duration_ms), "-c:v", "libx264", "-c:a", "aac", output_path,
    ]


def candidate_timestamp_label(candidate: Candidate) -> str:
    return f"{format_timestamp(candidate.start_ms)} --> {format_timestamp(candidate.end_ms)}"

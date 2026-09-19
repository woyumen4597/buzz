"""FFmpeg/FFprobe discovery, command construction and media artifact generation."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from buzz.ffmpeg_video_player import _find_ffmpeg, _find_ffprobe, probe_video

from .models import Candidate, VideoInfo, format_timestamp
from .subtitles import Subtitle, clip_subtitles, parse_srt_file, write_srt_file

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
    has_audio: bool = True,
    subtitle_path: str | None = None,
) -> list[str]:
    """Build a clip command.

    When ``subtitle_path`` is given the clip is muxed from that pre-retimed SRT
    instead of copying the source track. Input seeking does not rebase text
    subtitles, so copying the source track makes every clip's cues drift by its
    start offset. An empty SRT still yields a ``mov_text`` stream, which keeps
    the concatenation stream layout identical across clips.
    """
    if subtitle_path is not None:
        return [
            ffmpeg, "-y", "-ss", _seconds(candidate.start_ms), "-i", video_path,
            "-f", "srt", "-i", subtitle_path,
            "-map", "0:v:0", *( ["-map", "0:a:0?"] if has_audio else [] ),
            "-map", "1:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            *( ["-c:a", "aac", "-b:a", "192k"] if has_audio else [] ),
            "-c:s", "mov_text",
            # ``-t`` must be an output option: before the subtitle input it
            # would instead cap that input and produce an over-long clip.
            "-t", _seconds(candidate.duration_ms), output_path,
        ]
    return [
        ffmpeg, "-y", "-ss", _seconds(candidate.start_ms), "-i", video_path,
        "-t", _seconds(candidate.duration_ms),
        "-map", "0:v:0", *( ["-map", "0:a:0?"] if has_audio else [] ),
        "-map", "0:s?", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        *( ["-c:a", "aac", "-b:a", "192k"] if has_audio else [] ),
        "-c:s", "mov_text", output_path,
    ]


def extract_subtitles_command(ffmpeg: str, video_path: str, output_path: str) -> list[str]:
    """Extract the first text subtitle track of a source as SRT."""
    return [
        ffmpeg, "-y", "-i", video_path, "-map", "0:s:0", "-f", "srt", output_path,
    ]


def concat_command(ffmpeg: str, concat_file: str, output_path: str) -> list[str]:
    return [
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-map", "0:v:0", "-map", "0:a:0?", "-map", "0:s?",
        "-c", "copy", "-movflags", "+faststart", output_path,
    ]


def source_has_text_subtitles(video_path: str) -> bool:
    """Return whether the source carries a text subtitle track we can retime."""
    try:
        ffprobe = _find_ffprobe()
        result = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "s",
                "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path,
            ],
            capture_output=True, check=True, text=True,
        )
    except Exception as exc:  # probing is best-effort; no subtitles means no retiming
        LOG.debug("subtitle probe failed for %s: %s", video_path, exc)
        return False
    text_codecs = {"subrip", "srt", "ass", "ssa", "mov_text", "text", "webvtt", "ttml"}
    return any(
        line.strip().lower() in text_codecs
        for line in result.stdout.splitlines()
        if line.strip()
    )


def render_selected_video(
    ffmpeg: str,
    video_path: str,
    candidates: Sequence[Candidate],
    output_dir: Path,
    has_audio: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create individual selected clips and concatenate them in timeline order."""
    selected = sorted(
        (candidate for candidate in candidates if candidate.selected),
        key=lambda item: (item.start_ms, item.end_ms, item.id),
    )
    if not selected:
        raise ValueError("select at least one candidate before rendering")
    if any(
        earlier.end_ms > later.start_ms
        for earlier, later in zip(selected, selected[1:])
    ):
        raise ValueError("selected candidates must not overlap")

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    total = len(selected) + 1
    concat_file = output_dir / ".highlight-concat.txt"
    clip_paths: list[Path] = []
    source_subtitles = _load_source_subtitles(ffmpeg, video_path, output_dir)
    if source_subtitles is None:
        LOG.info("no retimable subtitle track; rendering %d clips without subtitles", len(selected))
    else:
        LOG.info(
            "retiming %d subtitle cues across %d clips", len(source_subtitles), len(selected)
        )
    try:
        for index, candidate in enumerate(selected, 1):
            clip_path = clips_dir / f"clip_{index:04d}.mp4"
            subtitle_path: Path | None = None
            if source_subtitles is not None:
                # Rebase the cue timeline onto this clip so the concatenated
                # reel keeps every subtitle aligned with its own video segment.
                subtitle_path = clips_dir / f"clip_{index:04d}.srt"
                write_srt_file(
                    subtitle_path,
                    clip_subtitles(source_subtitles, candidate.start_ms, candidate.end_ms),
                )
            try:
                _run_atomic(
                    clip_command(
                        ffmpeg,
                        video_path,
                        candidate,
                        str(clip_path),
                        has_audio=has_audio,
                        subtitle_path=str(subtitle_path) if subtitle_path else None,
                    ),
                    clip_path,
                )
            finally:
                if subtitle_path is not None:
                    subtitle_path.unlink(missing_ok=True)
            clip_paths.append(clip_path)
            if progress_callback:
                progress_callback(index, total, f"剪辑 {candidate.id}")

        concat_lines = []
        for clip_path in clip_paths:
            escaped = str(clip_path.absolute()).replace("'", "'\\''")
            concat_lines.append(f"file '{escaped}'")
        concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        final_path = output_path or output_dir / "highlights.mp4"
        _run_atomic(concat_command(ffmpeg, str(concat_file), str(final_path)), final_path)
        if progress_callback:
            progress_callback(total, total, "拼接完成")
        return final_path
    finally:
        concat_file.unlink(missing_ok=True)


def _load_source_subtitles(
    ffmpeg: str, video_path: str, output_dir: Path
) -> list[Subtitle] | None:
    """Extract the source subtitle track once, or ``None`` if it has none.

    Extracting a single full track and slicing it per clip is what keeps the
    reel's subtitles aligned; copying the source track per clip drifts.
    """
    if not source_has_text_subtitles(video_path):
        return None
    srt_path = output_dir / ".highlight-source.srt"
    try:
        subprocess.run(
            extract_subtitles_command(ffmpeg, video_path, str(srt_path)),
            check=True,
            capture_output=True,
            text=True,
        )
        subtitles, warnings = parse_srt_file(srt_path)
        for warning in warnings:
            LOG.debug("source subtitle: %s", warning)
        if not subtitles:
            LOG.warning("source subtitles could not be parsed; rendering without subtitles")
            return None
        return subtitles
    except subprocess.CalledProcessError as exc:
        LOG.warning("subtitle extraction failed; rendering without subtitles: %s", _error_summary(exc))
        return None
    finally:
        srt_path.unlink(missing_ok=True)


def candidate_timestamp_label(candidate: Candidate) -> str:
    return f"{format_timestamp(candidate.start_ms)} --> {format_timestamp(candidate.end_ms)}"

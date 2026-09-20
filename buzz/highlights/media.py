"""FFmpeg/FFprobe discovery, command construction and media artifact generation."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Sequence

from buzz.ffmpeg_video_player import _find_ffmpeg, _find_ffprobe, probe_video

from .models import Candidate, VideoInfo, format_timestamp
from .subtitles import (
    Subtitle,
    clip_subtitles,
    extract_source_subtitles,
    write_srt_file,
)

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


# Rendering a highlight reel is dominated by the H.264 encode of each clip, and
# the clips are independent, so the two levers that matter are per-encode CPU
# cost and running several encodes at once. Apple silicon exposes a hardware
# encoder (VideoToolbox) that is roughly as fast as ``libx264 -preset medium``
# while using a fraction of the CPU, which leaves the performance cores free
# for the parallel workers below.
_HARDWARE_ENCODER = "h264_videotoolbox"
_SOFTWARE_ENCODER = "libx264"

#: Encodes run concurrently. Apple silicon has 4 performance cores; beyond that
#: the extra workers land on efficiency cores and slow the batch down.
_PARALLEL_ENCODES = 4

#: libx264 beyond the performance-core count regresses (measured 8 > 4 on M1).
_SOFTWARE_THREADS = "4"

# VideoToolbox is bitrate-driven and cannot take a CRF, so the bitrate has to be
# chosen up front. A fixed value is wrong for sources that are already heavily
# compressed: re-encoding a 720p source that averages 1.4 Mbit/s at a flat
# 6 Mbit/s produced a reel only 6% smaller than a 4-hour original despite being
# 3.4x shorter. Follow the source instead, with a floor so a very low-bitrate
# source is not re-encoded into visible artifacts, and a ceiling so an
# unusually high-bitrate source cannot balloon the output.
_MIN_VIDEO_BITRATE = 1_500_000
_MAX_VIDEO_BITRATE = 8_000_000

#: Re-encoding is not free: the output needs headroom over the source or the
#: extra generation loss shows. 1.15 keeps the reel visually equivalent while
#: still tracking a low-bitrate source downwards.
_SOURCE_BITRATE_FACTOR = 1.15

_ENCODER_CACHE: dict[tuple[str, int | None], list[str]] = {}

#: Hardware-encoder availability per ffmpeg binary. Probing spawns a process, so
#: the answer is cached separately from the per-bitrate command cache: the
#: bitrate varies by source but availability does not.
_ENCODER_PROBE_CACHE: dict[str, bool] = {}


def _encoder_available(ffmpeg: str, encoder: str) -> bool:
    cached = _ENCODER_PROBE_CACHE.get(ffmpeg)
    if cached is not None:
        return cached
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-h", f"encoder={encoder}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # probing is best-effort; fall back to software
        LOG.debug("encoder probe for %s failed: %s", encoder, exc)
        return False
    available = result.returncode == 0 and "not recognized" not in result.stdout.lower()
    _ENCODER_PROBE_CACHE[ffmpeg] = available
    return available


def source_video_bitrate(video_path: str, duration_ms: int) -> int | None:
    """Approximate the source's overall bitrate in bits per second.

    The container's own ``bit_rate`` is absent for some files, so this falls
    back to size over duration. The total (audio included) is the right basis
    because the clip encoder re-encodes audio too.
    """
    try:
        size = os.path.getsize(video_path)
    except OSError as exc:
        LOG.debug("could not stat %s for a bitrate estimate: %s", video_path, exc)
        return None
    if size <= 0 or duration_ms <= 0:
        return None
    return int(size * 8 / (duration_ms / 1000))


def target_video_bitrate(source_bitrate: int | None) -> int | None:
    """Pick the clip bitrate, tracking the source where it is known.

    Returns ``None`` when the source bitrate is unknown, which makes the caller
    fall back to a fixed rate rather than guessing from a bad measurement.
    """
    if not source_bitrate or source_bitrate <= 0:
        return None
    scaled = int(source_bitrate * _SOURCE_BITRATE_FACTOR)
    return max(_MIN_VIDEO_BITRATE, min(_MAX_VIDEO_BITRATE, scaled))


def video_encoder_args(
    ffmpeg: str, source_bitrate: int | None = None
) -> list[str]:
    """Video-encoding options for clip renders, preferring hardware when usable.

    Results are memoized per ``(ffmpeg, bitrate)`` because the bitrate depends on
    the source; probing spawns a process and every clip would otherwise pay for
    it. The software path is CRF-based and therefore bitrate-independent, so it
    memoizes under a single key.
    """
    bitrate = target_video_bitrate(source_bitrate)
    # Check both possible cache keys before probing so a repeated call with a
    # known bitrate never spawns ffmpeg again.
    for key in ((ffmpeg, bitrate), (ffmpeg, None)):
        cached = _ENCODER_CACHE.get(key)
        if cached is not None:
            return list(cached)

    hardware = sys.platform == "darwin" and _encoder_available(ffmpeg, _HARDWARE_ENCODER)
    key = (ffmpeg, bitrate) if hardware else (ffmpeg, None)

    if hardware:
        if bitrate is None:
            # No usable measurement; keep the previous 1080p-oriented default.
            args = ["-c:v", _HARDWARE_ENCODER, "-b:v", "6M"]
            detail = "6.0 Mbit/s (source bitrate unknown)"
        else:
            args = ["-c:v", _HARDWARE_ENCODER, "-b:v", str(bitrate)]
            detail = f"{bitrate / 1e6:.1f} Mbit/s"
        LOG.info("clip encoding: %s (hardware, %s)", _HARDWARE_ENCODER, detail)
    else:
        # ``veryfast``/``crf 20`` is ~1.8x faster than ``medium``/``crf 18`` at a
        # difference that is invisible in a highlight preview. CRF adapts to
        # picture complexity on its own, so no source bitrate is needed.
        args = [
            "-c:v", _SOFTWARE_ENCODER,
            "-preset", "veryfast", "-crf", "20",
            "-threads", _SOFTWARE_THREADS,
        ]
        LOG.info("clip encoding: %s (software)", _SOFTWARE_ENCODER)
    _ENCODER_CACHE[key] = args
    return list(args)


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
    except subprocess.CalledProcessError as exc:
        temp_path.unlink(missing_ok=True)
        # ``capture_output`` hides FFmpeg's own message, which is the only place
        # the reason for a failure appears (e.g. the subrip decoder rejecting a
        # cue). Surface it on the exception or the CLI/GUI shows just a command.
        raise RuntimeError(
            f"{exc.cmd[0]} failed with exit code {exc.returncode}: {_error_summary(exc)}"
        ) from exc
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


#: Substrings that mark the informative line in FFmpeg's stderr. The final line
#: is usually a generic "Conversion failed!", so picking only the last line
#: hides the actual cause.
_ERROR_HINTS = (
    "invalid",
    "error",
    "unable",
    "failed",
    "no such",
    "not found",
    "unknown",
    "cannot",
    "denied",
)


def _error_summary(exc: subprocess.CalledProcessError) -> str:
    """Return the most useful line(s) from a failed FFmpeg invocation."""
    raw = (exc.stderr or exc.stdout or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return str(exc)

    def readable(line: str) -> str:
        # Drop the "[component @ 0x...]" prefix FFmpeg puts in front of messages.
        if line.startswith("[") and "] " in line:
            line = line.split("] ", 1)[1]
        return line.strip()

    useful: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(hint in lowered for hint in _ERROR_HINTS):
            text = readable(line)
            if text and text not in useful:
                useful.append(text)
        if len(useful) == 3:
            break
    if not useful:
        useful = [readable(lines[-1])]
    return "; ".join(useful)[-500:]


def clip_command(
    ffmpeg: str,
    video_path: str,
    candidate: Candidate,
    output_path: str,
    has_audio: bool = True,
    subtitle_path: str | None = None,
    source_bitrate: int | None = None,
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
            *video_encoder_args(ffmpeg, source_bitrate),
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
        "-map", "0:s?", *video_encoder_args(ffmpeg, source_bitrate),
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
    source_subtitles: Sequence[Subtitle] | None = None,
    source_duration_ms: int = 0,
) -> Path:
    """Create individual selected clips and concatenate them in timeline order.

    When ``source_subtitles`` is given the caller has already extracted them
    (the ranking stage does this so cues can influence scores); otherwise the
    track is extracted here.

    ``source_duration_ms`` lets the clip bitrate track the source's own; without
    it a heavily compressed source is re-encoded at a fixed rate and the reel
    can end up nearly as large as the original despite being far shorter.
    """
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
    source_bitrate = None
    if source_duration_ms:
        source_bitrate = source_video_bitrate(video_path, source_duration_ms)
    if source_bitrate:
        target = target_video_bitrate(source_bitrate)
        LOG.info(
            "source bitrate ~%.1f Mbit/s; clip target %s",
            source_bitrate / 1e6,
            f"{target / 1e6:.1f} Mbit/s" if target else "default",
        )
    if source_subtitles is None:
        source_subtitles = _load_source_subtitles(ffmpeg, video_path, output_dir)
    if source_subtitles is None:
        LOG.info("no retimable subtitle track; rendering %d clips without subtitles", len(selected))
    else:
        LOG.info(
            "retiming %d subtitle cues across %d clips", len(source_subtitles), len(selected)
        )
    try:
        def render_clip(index: int, candidate: Candidate) -> Path:
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
                        source_bitrate=source_bitrate,
                    ),
                    clip_path,
                )
            finally:
                if subtitle_path is not None:
                    subtitle_path.unlink(missing_ok=True)
            return clip_path

        # Each clip is an independent ffmpeg process reading the source and
        # writing its own file, so they encode concurrently. The pool is capped
        # at the performance-core count: a single libx264 encode cannot saturate
        # the machine (measured ~3.3 of 8 cores), and running four at once cut a
        # four-clip batch from 5.9s to 3.2s.
        workers = min(_PARALLEL_ENCODES, len(selected))
        if workers > 1:
            LOG.info("rendering %d clips with %d parallel encodes", len(selected), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(render_clip, index, candidate): candidate
                for index, candidate in enumerate(selected, 1)
            }
            # Report progress in completion order, not submission order: a
            # failing clip should surface as soon as it fails.
            for done, future in enumerate(as_completed(futures), 1):
                candidate = futures[future]
                clip_paths.append(future.result())
                LOG.info(
                    "clip %d/%d rendered (%.1fs from %s)",
                    done,
                    len(selected),
                    candidate.duration_ms / 1000,
                    format_timestamp(candidate.start_ms),
                )
                if progress_callback:
                    progress_callback(done, total, f"剪辑 {candidate.id}")

        # Concatenation order is the timeline order, which is independent of the
        # order the encodes finished in.
        clip_paths.sort()

        concat_lines = []
        for clip_path in clip_paths:
            escaped = str(clip_path.absolute()).replace("'", "'\\''")
            concat_lines.append(f"file '{escaped}'")
        concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        final_path = output_path or output_dir / "highlights.mp4"
        LOG.info("concatenating %d clips", len(clip_paths))
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
    return extract_source_subtitles(
        ffmpeg,
        video_path,
        has_text_subtitles=source_has_text_subtitles,
        build_command=extract_subtitles_command,
        work_dir=output_dir,
    )


def candidate_timestamp_label(candidate: Candidate) -> str:
    return f"{format_timestamp(candidate.start_ms)} --> {format_timestamp(candidate.end_ms)}"

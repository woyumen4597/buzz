"""Candidate generation from fixed windows and FFmpeg scene changes."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import replace
from typing import Iterable, Sequence

from .models import (
    Candidate,
    HighlightConfig,
    VideoInfo,
    clamp_interval,
    interval_iou,
    merge_reasons,
)

LOG = logging.getLogger(__name__)


def _candidate(
    start_ms: int,
    end_ms: int,
    video: VideoInfo,
    config: HighlightConfig,
    reasons: Iterable[str],
    scene_signal: float = 0.0,
) -> Candidate | None:
    start_ms, end_ms = clamp_interval(start_ms, end_ms, video.duration_ms)
    duration = end_ms - start_ms
    if duration < round(config.min_duration_seconds * 1000):
        return None
    padding = round(config.padding_seconds * 1000)
    preview_start, preview_end = clamp_interval(
        start_ms - padding, end_ms + padding, video.duration_ms
    )
    return Candidate(
        id="",
        start_ms=start_ms,
        end_ms=end_ms,
        preview_start_ms=preview_start,
        preview_end_ms=preview_end,
        score=score_candidate(scene_signal, 0.0, 0.0, duration_quality(duration, config)),
        reasons=merge_reasons(reasons),
    )


def score_candidate(
    scene_signal: float = 0.0,
    transcript_density_signal: float = 0.0,
    audio_activity_signal: float = 0.0,
    duration_quality_signal: float = 0.0,
    motion_signal: float | None = None,
) -> float:
    """Combine heuristic signals into a review-priority score.

    The four-argument form is kept compatible with the original heuristic.
    When measured visual motion is supplied, it replaces the unused audio
    placeholder with a real signal and makes visually active windows rank
    above otherwise identical static windows.
    """
    if motion_signal is None:
        value = (
            0.50 * scene_signal
            + 0.25 * transcript_density_signal
            + 0.15 * audio_activity_signal
            + 0.10 * duration_quality_signal
        )
    else:
        value = (
            0.30 * scene_signal
            + 0.20 * transcript_density_signal
            + 0.20 * motion_signal
            + 0.20 * audio_activity_signal
            + 0.10 * duration_quality_signal
        )
    return max(0.0, min(1.0, value))


def duration_quality(duration_ms: int, config: HighlightConfig) -> float:
    target = config.window_seconds * 1000
    maximum = config.max_duration_seconds * 1000
    if duration_ms <= 0 or maximum <= 0:
        return 0.0
    if duration_ms <= target:
        return duration_ms / target if target else 0.0
    return max(0.0, 1.0 - (duration_ms - target) / max(1, maximum - target))


def fixed_window_candidates(video: VideoInfo, config: HighlightConfig | None = None) -> list[Candidate]:
    config = config or HighlightConfig()
    window = max(1, round(config.window_seconds * 1000))
    stride = max(1, round(config.stride_seconds * 1000))
    minimum = round(config.min_duration_seconds * 1000)
    maximum = round(config.max_duration_seconds * 1000)
    if video.duration_ms < minimum:
        return []
    result: list[Candidate] = []
    start = 0
    while start < video.duration_ms:
        end = min(video.duration_ms, start + window)
        if end - start >= minimum:
            if end - start > maximum:
                end = start + maximum
            candidate = _candidate(start, end, video, config, ["fixed_window"])
            if candidate:
                result.append(candidate)
        if end >= video.duration_ms:
            break
        start += stride
    # If the tail is too short, extend the previous window to cover it.
    if result and video.duration_ms - result[-1].end_ms >= minimum:
        tail = _candidate(
            max(0, video.duration_ms - window), video.duration_ms, video, config, ["fixed_window"]
        )
        if tail and (tail.start_ms, tail.end_ms) != (result[-1].start_ms, result[-1].end_ms):
            result.append(tail)
    return result


def parse_scene_timestamps(output: str) -> list[int]:
    """Parse FFmpeg showinfo pts_time values from stderr/stdout."""
    values: list[int] = []
    for match in re.finditer(r"(?:pts_time|scene_time)[:=]\s*(-?\d+(?:\.\d+)?)", output):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value >= 0:
            values.append(round(value * 1000))
    return sorted(set(values))


def motion_analysis_command(ffmpeg: str, video_path: str) -> list[str]:
    """Build a low-rate frame-difference scan for visual motion."""
    filter_expr = "fps=1,scale=160:-2,tblend=all_mode=difference,signalstats,metadata=print"
    return [
        ffmpeg, "-hide_banner", "-nostats", "-i", video_path,
        "-vf", filter_expr, "-an", "-f", "null", "-",
    ]


def parse_motion_samples(output: str) -> list[tuple[int, float]]:
    """Parse ``metadata=print`` YAVG samples into (timestamp_ms, value)."""
    samples: list[tuple[int, float]] = []
    timestamp: int | None = None
    for line in output.splitlines():
        pts_time_match = re.search(r"pts_time[:=]\s*(-?\d+(?:\.\d+)?)", line)
        pts_match = re.search(r"(?:^|\s)pts[:=]\s*(-?\d+(?:\.\d+)?)", line)
        if pts_time_match:
            try:
                timestamp = round(float(pts_time_match.group(1)) * 1000)
            except ValueError:
                timestamp = None
        elif pts_match:
            # ``metadata=print`` normally includes pts_time; pts is only a
            # fallback for synthetic/test output where it is already ms.
            try:
                timestamp = round(float(pts_match.group(1)))
            except ValueError:
                timestamp = None
        value_match = re.search(r"lavfi\.signalstats\.YAVG=([0-9]+(?:\.[0-9]+)?)", line)
        if value_match and timestamp is not None:
            try:
                samples.append((timestamp, float(value_match.group(1))))
            except ValueError:
                pass
            timestamp = None
    return samples


def scan_motion(
    ffmpeg: str, video_path: str, timeout: float | None = None
) -> list[tuple[int, float]]:
    process = subprocess.run(
        motion_analysis_command(ffmpeg, video_path),
        capture_output=True, text=True, check=True, timeout=timeout,
    )
    return parse_motion_samples(process.stderr + "\\n" + process.stdout)


def audio_activity_command(ffmpeg: str, video_path: str) -> list[str]:
    """Build a one-second RMS loudness scan for speech/music activity."""
    filter_expr = (
        "aresample=16000,asetnsamples=n=16000:p=1,"
        "astats=metadata=1:reset=1,"
        "ametadata=print:key=lavfi.astats.Overall.RMS_level"
    )
    return [
        ffmpeg, "-hide_banner", "-nostats", "-i", video_path,
        "-vn", "-af", filter_expr, "-f", "null", "-",
    ]


def parse_audio_samples(output: str) -> list[tuple[int, float]]:
    """Parse FFmpeg RMS dB samples into ``(timestamp_ms, loudness_db)``."""
    samples: list[tuple[int, float]] = []
    timestamp: int | None = None
    for line in output.splitlines():
        pts_match = re.search(r"pts_time[:=]\s*(-?\d+(?:\.\d+)?)", line)
        if pts_match:
            try:
                timestamp = round(float(pts_match.group(1)) * 1000)
            except ValueError:
                timestamp = None
        value_match = re.search(
            r"(?:lavfi\.astats\.Overall\.RMS_level|RMS_level)=(-?\d+(?:\.\d+)?)",
            line,
        )
        if value_match and timestamp is not None:
            try:
                samples.append((timestamp, float(value_match.group(1))))
            except ValueError:
                pass
            timestamp = None
    return samples


def scan_audio_activity(
    ffmpeg: str, video_path: str, timeout: float | None = None
) -> list[tuple[int, float]]:
    process = subprocess.run(
        audio_activity_command(ffmpeg, video_path),
        capture_output=True, text=True, check=True, timeout=timeout,
    )
    return parse_audio_samples(process.stderr + "\\n" + process.stdout)


def scene_detection_command(
    ffmpeg: str, video_path: str, threshold: float = 0.35
) -> list[str]:
    """Build a low-cost scene scan command; output is parsed from showinfo."""
    filter_expr = f"fps=2,scale=320:-2,select='gt(scene\\,{threshold:g})',showinfo"
    return [
        ffmpeg, "-hide_banner", "-nostats", "-i", video_path,
        "-vf", filter_expr, "-an", "-f", "null", "-",
    ]


def scan_scene_changes(
    ffmpeg: str, video_path: str, threshold: float = 0.35, timeout: float | None = None
) -> list[int]:
    process = subprocess.run(
        scene_detection_command(ffmpeg, video_path, threshold),
        capture_output=True, text=True, check=True, timeout=timeout,
    )
    return parse_scene_timestamps(process.stderr + "\n" + process.stdout)


def scene_candidates(
    video: VideoInfo,
    scene_changes_ms: Sequence[int],
    config: HighlightConfig | None = None,
) -> list[Candidate]:
    config = config or HighlightConfig()
    points = [0] + sorted({max(0, min(video.duration_ms, int(point))) for point in scene_changes_ms})
    if points[-1] < video.duration_ms:
        points.append(video.duration_ms)
    result: list[Candidate] = []
    for start, end in zip(points, points[1:]):
        if end - start < round(config.min_scene_duration_seconds * 1000):
            continue
        if end - start <= round(config.max_duration_seconds * 1000):
            candidate = _candidate(start, end, video, config, ["scene_change"], 1.0)
            if candidate:
                result.append(candidate)
        else:
            sub_config = replace(config, window_seconds=min(config.window_seconds, config.max_duration_seconds))
            local_video = VideoInfo(end - start, video.width, video.height, video.fps)
            local_candidates = fixed_window_candidates(local_video, sub_config)
            for local_candidate in local_candidates:
                shifted = _candidate(
                    local_candidate.start_ms + start,
                    local_candidate.end_ms + start,
                    video,
                    config,
                    [*local_candidate.reasons, "scene_change"],
                    1.0,
                )
                if shifted:
                    result.append(shifted)
    return result


def combine_candidates(
    candidates: Iterable[Candidate], video: VideoInfo, config: HighlightConfig | None = None
) -> list[Candidate]:
    """NMS candidates by IoU, retain strongest score, and assign stable IDs."""
    config = config or HighlightConfig()
    ordered = sorted(candidates, key=lambda item: (-item.score, item.start_ms, item.end_ms))
    kept: list[Candidate] = []
    for original in ordered:
        candidate = replace(
            original,
            reasons=list(original.reasons),
            errors=list(original.errors),
        )
        duplicate = next(
            (existing for existing in kept if interval_iou(
                existing.start_ms, existing.end_ms, candidate.start_ms, candidate.end_ms
            ) > 0.6),
            None,
        )
        if duplicate is not None:
            duplicate.reasons = merge_reasons([*duplicate.reasons, *candidate.reasons])
            duplicate.score = max(duplicate.score, candidate.score)
            if candidate.status == "ignore":
                duplicate.status = "ignore"
                duplicate.selected = False
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: (item.start_ms, item.end_ms))
    if config.max_candidates > 0:
        kept = sorted(kept, key=lambda item: (-item.score, item.start_ms))[: config.max_candidates]
        kept.sort(key=lambda item: (item.start_ms, item.end_ms))
    for index, candidate in enumerate(kept, 1):
        candidate.id = f"candidate-{index:04d}"
        candidate.start_ms, candidate.end_ms = clamp_interval(
            candidate.start_ms, candidate.end_ms, video.duration_ms
        )
        candidate.preview_start_ms, candidate.preview_end_ms = clamp_interval(
            candidate.preview_start_ms, candidate.preview_end_ms, video.duration_ms
        )
    return kept


def apply_motion_scores(
    candidates: Iterable[Candidate],
    motion_samples: Sequence[tuple[int, float]],
    config: HighlightConfig,
    audio_samples: Sequence[tuple[int, float]] | None = None,
) -> list[Candidate]:
    """Score candidates from visual motion and optional audio activity."""
    samples = sorted(motion_samples)
    audio = sorted(audio_samples or [])
    audio_values = sorted(value for _, value in audio)
    audio_floor = audio_values[max(0, len(audio_values) // 10 - 1)] if audio_values else -60.0
    audio_ceiling = audio_values[-1] if audio_values else audio_floor
    audio_range = max(12.0, audio_ceiling - audio_floor)
    for candidate in candidates:
        values = [value for timestamp, value in samples if candidate.start_ms <= timestamp < candidate.end_ms]
        average = sum(values) / len(values) if values else 0.0
        peak = max(values, default=0.0)
        loudness = [value for timestamp, value in audio if candidate.start_ms <= timestamp < candidate.end_ms]
        audio_average = sum(loudness) / len(loudness) if loudness else audio_floor
        audio_peak = max(loudness, default=audio_floor)
        audio_signal = max(0.0, min(1.0, (0.65 * audio_average + 0.35 * audio_peak - audio_floor) / audio_range))
        sorted_values = sorted(values)
        upper_quartile = sorted_values[(len(sorted_values) - 1) * 3 // 4] if sorted_values else 0.0
        # Average motion is stable but can dilute a short reaction. Blend in
        # the upper quartile and peak so brief high-energy moments remain
        # discoverable without letting a single value dominate completely.
        average_signal = max(0.0, min(1.0, average / 32.0))
        quartile_signal = max(0.0, min(1.0, upper_quartile / 32.0))
        peak_signal = max(0.0, min(1.0, peak / 32.0))
        motion_score = 0.50 * average_signal + 0.30 * quartile_signal + 0.20 * peak_signal
        candidate.score = score_candidate(
            scene_signal=1.0 if "scene_change" in candidate.reasons else 0.0,
            duration_quality_signal=duration_quality(candidate.duration_ms, config),
            audio_activity_signal=audio_signal,
            motion_signal=motion_score,
        )
        if "short_window" in candidate.reasons:
            candidate.score = min(1.0, candidate.score + 0.08)
        # A genuinely strong peak is enough to keep a short reaction from
        # being discarded as static, while empty analysis remains neutral.
        peak_rescue = peak >= config.static_motion_threshold * 4
        if values and config.ignore_static_scenes and average < config.static_motion_threshold and not peak_rescue:
            candidate.status = "ignore"
            candidate.reasons = merge_reasons([*candidate.reasons, "static_scene"])
        elif "static_scene" in candidate.reasons:
            candidate.reasons = [reason for reason in candidate.reasons if reason != "static_scene"]
    return list(candidates)


def generate_candidates(
    video: VideoInfo,
    config: HighlightConfig | None = None,
    scene_changes_ms: Sequence[int] | None = None,
    motion_samples: Sequence[tuple[int, float]] | None = None,
    audio_samples: Sequence[tuple[int, float]] | None = None,
) -> list[Candidate]:
    config = config or HighlightConfig()
    candidates = fixed_window_candidates(video, config)
    if config.auto_edit and config.window_seconds > config.min_duration_seconds * 2:
        # A second, shorter scale catches brief reactions, punchlines, and
        # other high-value moments diluted by the default 20-second window.
        short_config = replace(
            config,
            window_seconds=max(config.min_duration_seconds * 2, config.window_seconds / 2),
            stride_seconds=max(config.min_duration_seconds, config.stride_seconds / 2),
            max_candidates=0,
        )
        candidates.extend(
            replace(
                candidate,
                score=min(1.0, candidate.score + 0.08),
                reasons=merge_reasons([*candidate.reasons, "short_window"]),
            )
            for candidate in fixed_window_candidates(video, short_config)
        )
    if not config.no_scene_detection and scene_changes_ms is not None:
        candidates.extend(scene_candidates(video, scene_changes_ms, config))
    if motion_samples or audio_samples:
        apply_motion_scores(candidates, motion_samples or [], config, audio_samples=audio_samples)
    return combine_candidates(candidates, video, config)

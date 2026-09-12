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
) -> float:
    return max(0.0, min(1.0, (
        0.50 * scene_signal
        + 0.25 * transcript_density_signal
        + 0.15 * audio_activity_signal
        + 0.10 * duration_quality_signal
    )))


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


def generate_candidates(
    video: VideoInfo,
    config: HighlightConfig | None = None,
    scene_changes_ms: Sequence[int] | None = None,
) -> list[Candidate]:
    config = config or HighlightConfig()
    candidates = fixed_window_candidates(video, config)
    if not config.no_scene_detection and scene_changes_ms is not None:
        candidates.extend(scene_candidates(video, scene_changes_ms, config))
    return combine_candidates(candidates, video, config)

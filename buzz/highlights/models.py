"""Data models and time/overlap helpers for highlight candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable


def clamp_interval(start_ms: int, end_ms: int, duration_ms: int) -> tuple[int, int]:
    """Clamp an interval to a media duration and reject inverted ranges."""
    start = max(0, min(int(start_ms), int(duration_ms)))
    end = max(0, min(int(end_ms), int(duration_ms)))
    if end < start:
        start, end = end, start
    return start, end


def format_timestamp(ms: int | float, separator: str = ".") -> str:
    """Format milliseconds as HH:MM:SS.mmm (or SRT's comma separator)."""
    value = max(0, int(round(ms)))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


@dataclass
class VideoInfo:
    duration_ms: int

    def __post_init__(self) -> None:
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        if self.width < 0 or self.height < 0 or self.fps < 0:
            raise ValueError("video dimensions and fps must be non-negative")
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = True
    video_codec: str | None = None
    audio_codec: str | None = None

    @classmethod
    def from_probe(cls, info: dict[str, Any]) -> "VideoInfo":
        return cls(
            duration_ms=int(info.get("duration_ms", 0)),
            width=int(info.get("width", 0)),
            height=int(info.get("height", 0)),
            fps=float(info.get("fps", 0)),
            has_video=bool(info.get("has_video", True)),
            video_codec=info.get("video_codec"),
            audio_codec=info.get("audio_codec"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HighlightConfig:
    window_seconds: float = 20.0
    stride_seconds: float = 10.0
    min_duration_seconds: float = 5.0
    max_duration_seconds: float = 45.0
    padding_seconds: float = 1.5
    scene_threshold: float = 0.35
    min_scene_duration_seconds: float = 2.0
    max_candidates: int = 200
    no_scene_detection: bool = False
    no_previews: bool = False
    gif: bool = False
    gif_limit: int = 20
    keep_existing: bool = False

    def __post_init__(self) -> None:
        numeric = (
            self.window_seconds,
            self.stride_seconds,
            self.min_duration_seconds,
            self.max_duration_seconds,
            self.padding_seconds,
            self.scene_threshold,
            self.min_scene_duration_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("highlight durations and thresholds must be finite and non-negative")
        if self.window_seconds == 0 or self.stride_seconds == 0:
            raise ValueError("window_seconds and stride_seconds must be positive")
        if self.max_duration_seconds < self.min_duration_seconds:
            raise ValueError("max_duration_seconds must be >= min_duration_seconds")
        if not 0 <= self.scene_threshold <= 1:
            raise ValueError("scene_threshold must be between 0 and 1")
        if self.max_candidates < 0 or self.gif_limit < 0:
            raise ValueError("candidate limits must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    id: str
    start_ms: int
    end_ms: int
    preview_start_ms: int
    preview_end_ms: int
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    thumbnail: str | None = None
    preview: str | None = None
    gif: str | None = None
    transcript: str = ""
    selected: bool = False
    status: str = "unprocessed"
    errors: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("candidate must have a positive interval")
        if self.preview_start_ms > self.start_ms or self.preview_end_ms < self.end_ms:
            raise ValueError("preview interval must contain candidate interval")
        if not 0 <= self.score <= 1:
            raise ValueError("candidate score must be between 0 and 1")
        if self.status not in {"unprocessed", "keep", "ignore"}:
            raise ValueError("candidate status must be unprocessed, keep, or ignore")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        fields = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**fields)


def interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def interval_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    intersection = interval_overlap(a_start, a_end, b_start, b_end)
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def merge_reasons(reasons: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(reason for reason in reasons if reason))

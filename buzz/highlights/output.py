"""Output path helpers for automatic highlight reels."""

from __future__ import annotations

from pathlib import Path


def automatic_output_path(video_path: str | Path) -> Path:
    """Return the final reel path beside the source video."""
    source = Path(video_path).expanduser()
    return source.with_name(f"{source.stem}_highlight.mp4")

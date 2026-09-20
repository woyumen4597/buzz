"""Output path helpers for automatic highlight reels."""

from __future__ import annotations

from pathlib import Path


def automatic_output_path(video_path: str | Path) -> Path:
    """Return the final reel path beside the source video."""
    source = Path(video_path).expanduser()
    return source.with_name(f"{source.stem}_highlight.mp4")


def work_dir_for(video_path: str | Path) -> Path:
    """Return the scratch directory used when the caller names no output dir."""
    source = Path(video_path).expanduser()
    return source.with_name(f"{source.stem}_highlights")


def verification_report_path(reel_path: str | Path) -> Path:
    """Return where a reel's verification report is written.

    Reports live in the reel's work directory rather than beside the reel: the
    reel sits next to the source video, and dropping a stray text file into the
    user's library is not a cleanup step this tool owns. The directory is
    derived from the reel name, mirroring :func:`work_dir_for`.
    """
    reel = Path(reel_path).expanduser()
    stem = reel.stem
    if stem.endswith("_highlight"):
        stem = stem[: -len("_highlight")]
    return reel.with_name(f"{stem}_highlights") / "verification.txt"

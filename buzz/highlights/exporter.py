"""Resumable generation state for the highlight pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .models import Candidate, HighlightConfig, VideoInfo


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_checkpoint(
    output_dir: Path,
    video_path: str,
    video: VideoInfo,
    config: HighlightConfig,
    candidates: Iterable[Candidate],
) -> None:
    """Persist resumable generation state.

    Selection is deterministic from the cached analysis, so the checkpoint only
    needs the input identity, the config and the candidates themselves.
    """
    source = Path(video_path)
    stat = source.stat()
    candidate_list = list(candidates)
    _json(output_dir / ".highlight-progress.json", {
        "version": 1,
        "input": {
            "path": str(source.absolute()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "video": video.to_dict(),
        "config": config.to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidate_list],
    })


def load_checkpoint(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / ".highlight-progress.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

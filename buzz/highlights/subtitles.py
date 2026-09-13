"""Small dependency-free SRT parser and candidate text association."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Candidate, interval_overlap

LOG = logging.getLogger(__name__)
_TIME_RE = re.compile(
    r"^(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)


@dataclass(frozen=True)
class Subtitle:
    start_ms: int
    end_ms: int
    text: str


def parse_timestamp(value: str) -> int:
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid SRT timestamp: {value}")
    hours, minutes, seconds = parts
    sec, millis = (seconds.split(".", 1) + ["0"])[:2]
    if len(millis) > 3:
        millis = millis[:3]
    millis = millis.ljust(3, "0")
    parsed = (int(hours) * 3600 + int(minutes) * 60 + int(sec)) * 1000 + int(millis)
    if parsed < 0 or int(minutes) >= 60 or int(sec) >= 60:
        raise ValueError(f"invalid SRT timestamp: {value}")
    return parsed


def parse_srt(content: str) -> tuple[list[Subtitle], list[str]]:
    """Return valid subtitles and warnings for skipped malformed records."""
    subtitles: list[Subtitle] = []
    warnings: list[str] = []
    blocks = re.split(r"\r?\n\s*\r?\n", content.lstrip("\ufeff").strip())
    for block_number, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if not lines:
            continue
        time_index = 1 if lines[0].strip().isdigit() else 0
        if len(lines) <= time_index:
            warnings.append(f"block {block_number}: missing time range")
            continue
        match = _TIME_RE.match(lines[time_index].strip())
        if not match:
            warnings.append(f"block {block_number}: invalid time range")
            continue
        text = " ".join(line.strip() for line in lines[time_index + 1 :] if line.strip()).strip()
        if not text:
            warnings.append(f"block {block_number}: empty text")
            continue
        try:
            start_ms = parse_timestamp(match.group("start"))
            end_ms = parse_timestamp(match.group("end"))
        except ValueError as exc:
            warnings.append(f"block {block_number}: {exc}")
            continue
        if end_ms <= start_ms:
            warnings.append(f"block {block_number}: end is not after start")
            continue
        subtitles.append(Subtitle(start_ms, end_ms, text))
    subtitles.sort(key=lambda item: (item.start_ms, item.end_ms))
    return subtitles, warnings


def parse_srt_file(path: str | Path) -> tuple[list[Subtitle], list[str]]:
    return parse_srt(Path(path).read_text(encoding="utf-8-sig"))


def associate_subtitles(candidate: Candidate, subtitles: Iterable[Subtitle]) -> str:
    texts: list[str] = []
    for subtitle in sorted(subtitles, key=lambda item: (item.start_ms, item.end_ms)):
        if interval_overlap(candidate.start_ms, candidate.end_ms, subtitle.start_ms, subtitle.end_ms):
            if subtitle.text and subtitle.text not in texts:
                texts.append(subtitle.text)
    candidate.transcript = " ".join(texts)
    if candidate.transcript:
        density = len(candidate.transcript) / max(1, candidate.duration_ms)
        candidate.score = min(1.0, candidate.score + min(0.25, density * 1000 * 0.25))
        candidate.reasons = list(dict.fromkeys([*candidate.reasons, "transcript_density"]))
        # Speech can be highly valuable even when the camera is static. Do
        # not let the motion filter discard dialogue-rich candidates before
        # automatic selection gets a chance to rank them.
        if candidate.status == "ignore" and "static_scene" in candidate.reasons:
            candidate.status = "unprocessed"
            candidate.reasons = list(dict.fromkeys([*candidate.reasons, "dialogue_rescue"]))
    return candidate.transcript

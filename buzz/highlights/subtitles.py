"""Small dependency-free SRT parser and candidate text association."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .models import Candidate, format_timestamp, interval_overlap

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


def extract_source_subtitles(
    ffmpeg: str,
    video_path: str | Path,
    *,
    has_text_subtitles: Callable[[str], bool],
    build_command: Callable[[str, str, str], list[str]],
    work_dir: Path | None = None,
) -> list[Subtitle] | None:
    """Extract a source's embedded text subtitle track as cues.

    Both the ranking stage and the render stage need the same track: ranking
    uses cue density as a score signal, rendering re-times the cues per clip.
    Extracting once and sharing the result keeps the two consistent and avoids
    a second full-track decode.

    Returns ``None`` when the source carries no retimable text track or when
    extraction fails; both are non-fatal and callers proceed without
    subtitles. ``has_text_subtitles``/``build_command`` are injected so this
    module stays free of ffmpeg discovery.
    """
    if not has_text_subtitles(str(video_path)):
        return None
    temp_dir = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp())
    owned = work_dir is None
    srt_path = temp_dir / ".highlight-source.srt"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            build_command(ffmpeg, str(video_path), str(srt_path)),
            check=True,
            capture_output=True,
            text=True,
        )
        subtitles, warnings = parse_srt_file(srt_path)
        for warning in warnings:
            LOG.debug("source subtitle: %s", warning)
        if not subtitles:
            LOG.warning("source subtitles could not be parsed")
            return None
        return subtitles
    except (subprocess.CalledProcessError, OSError) as exc:
        LOG.warning("subtitle extraction failed: %s", exc)
        return None
    finally:
        srt_path.unlink(missing_ok=True)
        if owned:
            try:
                temp_dir.rmdir()
            except OSError:
                pass


#: Longest single line written into a cue. FFmpeg's subrip decoder fails on
#: extremely long lines: one cue in a real source held a ~20000-character
#: sound-effect run and every clip containing it died with "Invalid UTF-8"
#: (exit 69) even though the SRT was valid UTF-8, aborting the whole render.
#: Wrapping into short lines decodes cleanly. Ordinary cues are far shorter
#: than this, so the cap only ever affects pathological input.
_MAX_SRT_LINE_CHARS = 400


def wrap_cue_text(text: str, width: int = _MAX_SRT_LINE_CHARS) -> str:
    """Split over-long lines so FFmpeg's subrip decoder can read the cue.

    Existing line breaks are preserved; only lines above ``width`` are split,
    and the split is by character so a multi-byte character is never cut in
    half. The rendered subtitle is unchanged for normal-length cues.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    lines = text.splitlines() or [""]
    wrapped: list[str] = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
            continue
        wrapped.extend(line[index : index + width] for index in range(0, len(line), width))
    return "\n".join(wrapped)


def format_srt(subtitles: Iterable[Subtitle]) -> str:
    """Serialize subtitles as an SRT document with sequential cue numbers."""
    blocks: list[str] = []
    for index, subtitle in enumerate(
        sorted(subtitles, key=lambda item: (item.start_ms, item.end_ms, item.text)), 1
    ):
        start = format_timestamp(subtitle.start_ms, ",")
        end = format_timestamp(subtitle.end_ms, ",")
        blocks.append(f"{index}\n{start} --> {end}\n{wrap_cue_text(subtitle.text)}\n")
    return "\n".join(blocks)


def write_srt_file(path: str | Path, subtitles: Iterable[Subtitle]) -> None:
    Path(path).write_text(format_srt(subtitles), encoding="utf-8")


def clip_subtitles(
    subtitles: Iterable[Subtitle], start_ms: int, end_ms: int
) -> list[Subtitle]:
    """Clip cues to ``[start_ms, end_ms]`` and rebase them to a zero origin.

    Input seeking in FFmpeg does not reliably rebase a text subtitle track, so
    rendering uses this instead: the full track is extracted once, then each
    clip receives its own SRT whose cues are trimmed and shifted. Cues that do
    not overlap the window are dropped rather than carried over.
    """
    clipped: list[Subtitle] = []
    for subtitle in subtitles:
        cue_start = max(subtitle.start_ms, start_ms)
        cue_end = min(subtitle.end_ms, end_ms)
        if cue_end <= cue_start:
            continue
        clipped.append(
            Subtitle(cue_start - start_ms, cue_end - start_ms, subtitle.text)
        )
    clipped.sort(key=lambda item: (item.start_ms, item.end_ms, item.text))
    return clipped


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

"""Fast pre-flight validation of a rendered highlight reel.

A full visual review of an hour-long reel is expensive, so this module spends
effort where it is cheap and buys the most confidence per second of CPU:

* **Container/stream checks cost nothing.** Probing reports duration, stream
  layout, and per-stream durations, which catches the failure modes that
  actually ruin a reel -- a missing audio or subtitle track, a truncated tail,
  and A/V drift from a bad concatenation.
* **Audio checks are nearly free.** Decoding audio runs hundreds of times
  faster than video, so a whole-reel silence/discontinuity scan costs seconds.
* **Video checks are sampled, never exhaustive.** The reel is decoded for a
  few short windows spread across its length rather than end to end, because
  decoding is the one genuinely expensive step.

Every check returns a :class:`Check` with a severity so a caller can decide
whether a problem is fatal, worth a look, or merely informational.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .media import find_tools
from .models import format_timestamp
from .subtitles import Subtitle, parse_timestamp

LOG = logging.getLogger(__name__)

#: Severity ordering, worst first. Used to summarise a run.
SEVERITY_ORDER = ("error", "warning", "info", "ok")

#: Text subtitle codecs that carry readable cues worth validating.
_TEXT_SUBTITLE_CODECS = {
    "subrip", "srt", "ass", "ssa", "mov_text", "text", "webvtt", "ttml",
}


@dataclass
class Check:
    """One validation result.

    ``name`` is a stable identifier so callers can assert on it; ``detail`` is
    the human-readable explanation shown to the user.
    """

    name: str
    severity: str
    detail: str
    data: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.severity in ("ok", "info")

    @property
    def failed(self) -> bool:
        return self.severity == "error"


@dataclass
class VerifyReport:
    """The complete set of checks for one file."""

    path: Path
    checks: list[Check] = field(default_factory=list)

    @property
    def errors(self) -> list[Check]:
        return [check for check in self.checks if check.severity == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [check for check in self.checks if check.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(
        self, name: str, severity: str, detail: str, **data: object
    ) -> Check:
        check = Check(name=name, severity=severity, detail=detail, data=dict(data))
        self.checks.append(check)
        return check

    def find(self, name: str) -> Check | None:
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "checks": [
                {
                    "name": check.name,
                    "severity": check.severity,
                    "detail": check.detail,
                    "data": check.data,
                }
                for check in self.checks
            ],
        }


def _run(command: Sequence[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command), capture_output=True, text=True, timeout=timeout
    )


# --------------------------------------------------------------------------
# Container and stream checks (no decoding, effectively instant)
# --------------------------------------------------------------------------


def probe_streams(ffprobe: str, path: str, timeout: float | None = 60) -> dict:
    """Return parsed stream and format information for ``path``.

    Raises ``RuntimeError`` when the file cannot be probed at all, which is
    itself the most severe validation failure possible.
    """
    process = _run(
        [
            ffprobe, "-v", "error", "-show_streams", "-show_format",
            "-of", "json", path,
        ],
        timeout=timeout,
    )
    if process.returncode != 0:
        raise RuntimeError((process.stderr or "ffprobe failed").strip())

    try:
        return json.loads(process.stdout or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"ffprobe produced unreadable output: {exc}") from exc


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def check_container(report: VerifyReport, probe: dict) -> None:
    """Validate the stream layout a playable reel must have."""
    streams = probe.get("streams") or []
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    subtitle = [
        s for s in streams
        if s.get("codec_type") == "subtitle"
        and str(s.get("codec_name", "")).lower() in _TEXT_SUBTITLE_CODECS
    ]

    if video:
        codec = video[0].get("codec_name")
        report.add(
            "video_stream",
            "ok",
            f"video: {codec} {video[0].get('width')}x{video[0].get('height')}",
            codec=codec,
        )
    else:
        report.add("video_stream", "error", "no video stream in the output")

    if audio:
        report.add(
            "audio_stream",
            "ok",
            f"audio: {audio[0].get('codec_name')} "
            f"{audio[0].get('sample_rate')}Hz "
            f"{audio[0].get('channels')}ch",
            codec=audio[0].get("codec_name"),
        )
    else:
        # A silent reel is a legitimate outcome for a source without audio, so
        # this is only worth flagging, not failing.
        report.add("audio_stream", "warning", "no audio stream in the output")

    if subtitle:
        report.add(
            "subtitle_stream",
            "ok",
            f"subtitles: {subtitle[0].get('codec_name')}, "
            f"{subtitle[0].get('nb_frames', '?')} frames",
            codec=subtitle[0].get("codec_name"),
            frames=_float(subtitle[0].get("nb_frames")),
        )
    else:
        report.add(
            "subtitle_stream",
            "warning",
            "no text subtitle track in the output",
        )


def check_durations(report: VerifyReport, probe: dict, tolerance_s: float = 1.5) -> None:
    """Catch a truncated tail and audio/video drift from a bad concat.

    A reel whose audio outruns its video (or vice versa) by more than the
    tolerance usually means a clip boundary was stitched incorrectly.
    """
    streams = probe.get("streams") or []
    container = _float((probe.get("format") or {}).get("duration"))
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video_duration = _float((video or {}).get("duration"))
    audio_duration = _float((audio or {}).get("duration"))

    if container <= 0:
        report.add("duration", "error", "container reports no duration")
        return

    report.add(
        "duration",
        "ok",
        f"duration {format_timestamp(container * 1000)}",
        container_s=container,
    )

    # A/V drift is only meaningful when ffprobe actually reported both.
    if video_duration > 0 and audio_duration > 0:
        drift = abs(video_duration - audio_duration)
        if drift > tolerance_s:
            longer = "audio" if audio_duration > video_duration else "video"
            report.add(
                "av_drift",
                "error",
                f"{longer} runs {drift:.2f}s longer than the other stream "
                f"(video {video_duration:.2f}s, audio {audio_duration:.2f}s)",
                drift_s=drift,
            )
        else:
            report.add(
                "av_drift", "ok", f"audio/video drift {drift:.2f}s", drift_s=drift
            )

    if video_duration > 0 and container - video_duration > tolerance_s:
        report.add(
            "tail",
            "warning",
            f"container is {container - video_duration:.2f}s longer than the "
            f"video stream; the tail may be blank or truncated",
            gap_s=container - video_duration,
        )


def check_start_time(report: VerifyReport, probe: dict, tolerance_s: float = 0.5) -> None:
    """A non-zero start time shifts every downstream timestamp."""
    start = _float((probe.get("format") or {}).get("start_time"))
    if abs(start) > tolerance_s:
        report.add(
            "start_time",
            "warning",
            f"container start_time is {start:+.3f}s, not zero",
            start_s=start,
        )
    else:
        report.add("start_time", "ok", "start_time is zero")


# --------------------------------------------------------------------------
# Subtitle checks (extraction only, no video decode)
# --------------------------------------------------------------------------


def _parse_srt_text(content: str) -> list[Subtitle]:
    """Parse SRT cues without dropping malformed ones.

    ``subtitles.parse_srt`` intentionally rejects zero-length cues because it
    feeds the *source* track. Validation needs the opposite: a zero-length cue
    in the output is exactly the defect being looked for, so cues are parsed
    directly here and kept even when ``end <= start``.
    """
    cues: list[Subtitle] = []
    pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
        r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
    )
    blocks = re.split(r"\r?\n\s*\r?\n", content.lstrip("\ufeff").strip())
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        match = pattern.search(lines[1] if lines[0].strip().isdigit() else lines[0])
        if not match:
            continue
        text_index = 2 if lines[0].strip().isdigit() else 1
        text = " ".join(line.strip() for line in lines[text_index:]).strip()
        try:
            start_ms = parse_timestamp(match.group(1))
            end_ms = parse_timestamp(match.group(2))
        except ValueError:
            continue
        cues.append(Subtitle(start_ms, end_ms, text))
    cues.sort(key=lambda item: (item.start_ms, item.end_ms))
    return cues


def check_subtitles(
    report: VerifyReport,
    path: str,
    ffmpeg: str,
    duration_s: float,
    timeout: float | None = 300,
) -> None:
    """Validate cue timing in the muxed subtitle track.

    Subtitles are the most fragile part of a retimed reel: the muxer can emit
    zero-length cues, and mis-clipped cues land past the end of the file. Both
    are silent failures that a spot-check of the video would miss.
    """
    has_text_track = any(
        check.name == "subtitle_stream" and check.severity == "ok"
        for check in report.checks
    )
    if not has_text_track:
        return

    # ``-f srt`` to a pipe is rejected by the muxer, so extract to a temp file.
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "subs.srt"
        process = _run(
            [
                ffmpeg, "-v", "error", "-y", "-i", path, "-map", "0:s:0",
                "-c:s", "srt", str(target),
            ],
            timeout=timeout,
        )
        if process.returncode != 0 or not target.is_file():
            report.add(
                "subtitles",
                "warning",
                "subtitle track present but could not be extracted",
            )
            return
        content = target.read_text(encoding="utf-8", errors="replace")

    cues = _parse_srt_text(content)
    if not cues:
        report.add(
            "subtitles", "warning", "subtitle track contains no parseable cues"
        )
        return

    limit_ms = int(round(duration_s * 1000))
    # Concatenating many clips rounds each clip to a frame boundary, so the
    # final cue can overrun the container slightly. The misalignment bug this
    # guards against put cues many seconds past the end, so a small fixed
    # tolerance separates rounding from a real defect without scaling so far
    # that a long reel would hide one.
    tolerance_ms = 3_000
    zero_length = [c for c in cues if c.end_ms <= c.start_ms]
    past_end = [c for c in cues if c.end_ms > limit_ms + tolerance_ms]
    overlapping = sum(
        1 for earlier, later in zip(cues, cues[1:]) if earlier.end_ms > later.start_ms
    )
    live_texts = {c.text for c in cues if c.end_ms > c.start_ms}
    unreadable = [c for c in zero_length if c.text not in live_texts]

    detail = f"{len(cues)} cues"
    if zero_length:
        detail += f", {len(zero_length)} zero-length"
    if past_end:
        detail += f", {len(past_end)} past end of file"
    if overlapping:
        detail += f", {overlapping} overlapping"
    report.add(
        "subtitles",
        "ok",
        detail,
        cues=len(cues),
        zero_length=len(zero_length),
        past_end=len(past_end),
        overlapping=overlapping,
        unreadable=len(unreadable),
    )

    if past_end:
        overrun = max(c.end_ms for c in past_end) - limit_ms
        report.add(
            "subtitle_bounds",
            "error",
            f"{len(past_end)} cue(s) end past the reel duration by up to "
            f"{overrun / 1000:.1f}s (last at "
            f"{format_timestamp(max(c.end_ms for c in past_end))}); "
            "subtitles are misaligned",
            count=len(past_end),
            overrun_s=overrun / 1000,
        )
    else:
        report.add("subtitle_bounds", "ok", "all cues fall inside the reel")

    # Zero-length cues never display. They are cosmetic when the same text is
    # also shown by a live cue, but they lose dialogue when it is not.
    if unreadable:
        report.add(
            "subtitle_zero_length",
            "warning",
            f"{len(unreadable)} cue(s) have no duration and no visible "
            f"duplicate, so their text never appears "
            f"(e.g. \u201c{unreadable[0].text[:24]}\u201d)",
            unreadable=len(unreadable),
        )
    elif zero_length:
        report.add(
            "subtitle_zero_length",
            "info",
            f"{len(zero_length)} zero-length cue(s), all duplicated by a "
            "visible cue",
            zero_length=len(zero_length),
        )
    else:
        report.add("subtitle_zero_length", "ok", "no zero-length cues")


# --------------------------------------------------------------------------
# Audio checks (decode is ~500-1000x realtime, so the whole reel is affordable)
# --------------------------------------------------------------------------


_SILENCE_RE = re.compile(r"silence_start:\s*([0-9.]+)")


def check_audio(
    report: VerifyReport,
    path: str,
    ffmpeg: str,
    duration_s: float,
    gap_s: float = 2.0,
    timeout: float | None = 900,
) -> None:
    """Look for silent stretches long enough to read as a broken reel.

    Audio decodes far faster than video, so this runs over the entire file. A
    long silent run is usually a clip boundary that lost its audio.
    """
    if not any(
        check.name == "audio_stream" and check.severity == "ok"
        for check in report.checks
    ):
        return

    process = _run(
        [
            ffmpeg, "-hide_banner", "-nostats", "-v", "info", "-i", path,
            "-vn", "-af", f"silencedetect=n=-50dB:d={gap_s}", "-f", "null", "-",
        ],
        timeout=timeout,
    )
    if process.returncode != 0:
        report.add(
            "audio_silence", "info", "audio silence scan could not be completed"
        )
        return

    starts = [float(value) for value in _SILENCE_RE.findall(process.stderr)]
    # Only count silence that is not merely the natural end of the reel.
    meaningful = [s for s in starts if s < duration_s - gap_s]
    if meaningful:
        worst = meaningful[0]
        report.add(
            "audio_silence",
            "warning",
            f"{len(meaningful)} silent stretch(es) of >= {gap_s:.0f}s "
            f"(first at {format_timestamp(worst * 1000)})",
            count=len(meaningful),
            first_s=worst,
        )
    else:
        report.add("audio_silence", "ok", f"no silent gaps of {gap_s:.0f}s or more")


# --------------------------------------------------------------------------
# Sampled video checks (the only genuinely expensive step)
# --------------------------------------------------------------------------


def sample_windows(
    duration_s: float, count: int = 8, window_s: float = 6.0
) -> list[tuple[float, float]]:
    """Return ``(start, duration)`` windows spread evenly across a reel.

    Sampling the whole reel is what makes verification slow, so a handful of
    well-spread windows is used instead. Windows are pushed inward from the
    very ends, where a reel legitimately starts or stops on black.
    """
    if duration_s <= 0 or count <= 0:
        return []
    if duration_s <= window_s * count:
        return [(0.0, duration_s)]
    margin = min(window_s, duration_s * 0.02)
    usable_start = margin
    usable_end = duration_s - margin
    span = usable_end - usable_start
    step = span / count
    windows: list[tuple[float, float]] = []
    for index in range(count):
        start = usable_start + index * step
        start = min(start, duration_s - window_s)
        windows.append((round(start, 3), window_s))
    return windows


def check_video_samples(
    report: VerifyReport,
    path: str,
    ffmpeg: str,
    duration_s: float,
    count: int = 8,
    window_s: float = 6.0,
    timeout: float | None = 180,
) -> None:
    """Decode a few windows looking for frames that should not be there.

    Black or frozen frames anywhere in the middle of a reel point at a broken
    clip: a wrong seek, a missing segment, or a stream copy that dropped
    frames. Only a sample is decoded, which keeps this affordable on a long
    reel.
    """
    windows = sample_windows(duration_s, count=count, window_s=window_s)
    if not windows:
        return

    black: list[float] = []
    frozen: list[float] = []
    decoded_s = 0.0
    for start, length in windows:
        process = _run(
            [
                ffmpeg, "-hide_banner", "-nostats", "-v", "info",
                "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", path,
                "-map", "0:v:0",
                "-vf", "blackdetect=d=0.4:pix_th=0.10,freezedetect=n=-60dB:d=1.5",
                "-an", "-f", "null", "-",
            ],
            timeout=timeout,
        )
        decoded_s += length
        if process.returncode != 0:
            continue
        for match in re.finditer(r"black_start:([0-9.]+)", process.stderr):
            black.append(start + float(match.group(1)))
        for match in re.finditer(r"freeze_start:\s*([0-9.]+)", process.stderr):
            frozen.append(start + float(match.group(1)))

    if not black and not frozen:
        report.add(
            "video_samples",
            "ok",
            f"{len(windows)} sampled window(s), {decoded_s:.0f}s decoded: "
            "no black or frozen frames",
            windows=len(windows),
        )
        return

    detail_parts = []
    if black:
        detail_parts.append(f"{len(black)} black stretch(es)")
    if frozen:
        detail_parts.append(f"{len(frozen)} frozen stretch(es)")
    report.add(
        "video_samples",
        "warning",
        f"sampled video contains {' and '.join(detail_parts)} "
        f"(first at {format_timestamp((black or frozen)[0] * 1000)})",
        black=len(black),
        frozen=len(frozen),
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def verify_reel(
    path: str | Path,
    sample_count: int = 8,
    sample_window_s: float = 6.0,
    check_video: bool = True,
    check_audio_flags: bool = True,
) -> VerifyReport:
    """Run every applicable check against a rendered reel.

    The no-decode container checks always run; audio and video checks are
    skipped when the corresponding stream is absent.
    """
    target = Path(path)
    report = VerifyReport(path=target)
    if not target.is_file():
        report.add("exists", "error", f"file does not exist: {target}")
        return report
    if not target.stat().st_size:
        report.add("exists", "error", f"file is empty: {target}")
        return report
    report.add("exists", "ok", f"{target.stat().st_size / 1e6:.1f} MB")

    try:
        ffmpeg, ffprobe = find_tools()
    except Exception as exc:  # pragma: no cover - environment dependent
        report.add("tools", "error", f"FFmpeg is unavailable: {exc}")
        return report

    try:
        probe = probe_streams(ffprobe, str(target))
    except RuntimeError as exc:
        report.add("readable", "error", f"cannot read the output: {exc}")
        return report

    check_container(report, probe)
    check_durations(report, probe)
    check_start_time(report, probe)

    duration_s = _float((probe.get("format") or {}).get("duration"))
    check_subtitles(report, str(target), ffmpeg, duration_s)
    if check_audio_flags:
        check_audio(report, str(target), ffmpeg, duration_s)
    if check_video:
        check_video_samples(
            report, str(target), ffmpeg, duration_s,
            count=sample_count, window_s=sample_window_s,
        )
    return report


def summarise(report: VerifyReport) -> str:
    """Render a short human-readable summary of a report."""
    lines: list[str] = []
    for check in report.checks:
        marker = {
            "error": "FAIL",
            "warning": "WARN",
            "info": "info",
            "ok": "ok",
        }.get(check.severity, check.severity)
        lines.append(f"[{marker:4}] {check.name}: {check.detail}")
    return "\n".join(lines)

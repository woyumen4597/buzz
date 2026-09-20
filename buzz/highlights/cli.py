"""Standalone CLI for generating reviewable highlight candidates."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
from pathlib import Path
from typing import Callable

from .exporter import load_checkpoint
from .auto_edit import (
    ALGORITHM_VERSION,
    resolve_target_duration_seconds,
    selection_summary,
    select_auto_candidates,
)
from .output import automatic_output_path, work_dir_for
from .media import (
    extract_subtitles_command,
    find_tools,
    probe_media,
    render_selected_video,
    source_has_text_subtitles,
)
from .models import HighlightConfig
from .progress import phase
from .subtitles import (
    Subtitle,
    associate_subtitles,
    extract_source_subtitles,
    parse_srt_file,
)
from .verify import summarise, verify_reel
from .windows import (
    generate_candidates,
    scan_audio_activity,
    scan_motion,
    scan_motion_and_scenes,
)

LOG = logging.getLogger(__name__)

#: Review assets (thumbnails, previews, GIFs) are small ffmpeg jobs whose cost
#: is dominated by process startup and source seek, so the wall clock is almost
#: entirely the serial spawn chain. Four at a time matches the performance-core
#: count without starving the encode pool that runs later.
_REVIEW_ASSET_WORKERS = 4


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("target ratio must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="buzz-highlights",
        description="Generate a highlight reel that skips the source's dull stretches.",
    )
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--srt", type=Path)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument("--padding-seconds", type=float, default=1.5)
    parser.add_argument("--scene-threshold", type=float, default=0.35)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--no-scene-detection", action="store_true")
    parser.add_argument("--keep-static-scenes", action="store_true", help="do not auto-ignore low-motion candidates")
    parser.add_argument("--target-ratio", type=_ratio, default=0.3, help="target reel duration as a fraction of the source, from 0 to 1")
    parser.add_argument("--target-duration", type=float, dest="target_duration", help=argparse.SUPPRESS)
    parser.add_argument("--max-auto-clips", type=int, default=0, help="maximum clips in an automatic reel; 0 means unlimited")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="minimum automatic selection score")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument(
        "--verify",
        choices=("off", "fast", "full"),
        default="fast",
        help="validate the rendered reel: fast samples audio/video, full decodes more",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> HighlightConfig:
    return HighlightConfig(
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        padding_seconds=args.padding_seconds,
        scene_threshold=args.scene_threshold,
        max_candidates=args.max_candidates,
        no_scene_detection=args.no_scene_detection,
        keep_existing=args.keep_existing,
        ignore_static_scenes=not getattr(args, "keep_static_scenes", False),
        target_duration_ratio=getattr(args, "target_ratio", 0.3),
        target_duration_seconds=getattr(args, "target_duration", None),
        max_auto_clips=getattr(args, "max_auto_clips", 0),
        score_threshold=getattr(args, "score_threshold", 0.0),
    )


def run(
    args: argparse.Namespace,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    video_path = args.video.expanduser().absolute()
    if not video_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {video_path}")
    if not video_path.stat().st_size:
        raise ValueError(f"input video is empty: {video_path}")
    cleanup_work_dir = args.output_dir is None
    final_output_path = automatic_output_path(video_path)
    output_dir = (
        Path(args.output_dir).expanduser().absolute()
        if args.output_dir
        else work_dir_for(video_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise NotADirectoryError(str(output_dir))

    ffmpeg, _ = find_tools()
    with phase("probing source media", LOG):
        video = probe_media(str(video_path))
    LOG.info(
        "source: %s (%dx%d, %.0fs, video=%s, audio=%s)",
        video_path.name,
        video.width,
        video.height,
        video.duration_ms / 1000,
        video.video_codec or "none",
        video.audio_codec or "none",
    )
    config = _config(args)
    config.target_duration_seconds = resolve_target_duration_seconds(
        video.duration_ms,
        config.target_duration_seconds,
        configured_ratio=config.target_duration_ratio,
    )
    config.max_candidates = 0
    # Failures here are reported through the log and degrade the reel to
    # whatever signals remain; they never abort the run.
    scene_points: list[int] | None = None
    motion_samples: list[tuple[int, float]] = []
    audio_samples: list[tuple[int, float]] = []
    motion_scanned = False
    if not config.no_scene_detection:
        # Motion and scene analysis share one decode pass; on a multi-hour
        # source that is the difference between minutes of fan noise and one
        # continuous scan.
        try:
            with phase("scanning motion and scenes", LOG):
                motion_samples, scene_points = scan_motion_and_scenes(
                    ffmpeg, str(video_path), config.scene_threshold
                )
            motion_scanned = True
            LOG.info(
                "combined scan: %d motion samples, %d scene changes",
                len(motion_samples),
                len(scene_points),
            )
        except Exception as exc:  # scene detection is explicitly best-effort
            warning = f"scene detection failed; using fixed windows: {exc}"
            LOG.warning(warning)
    if not motion_scanned:
        try:
            with phase("scanning motion", LOG):
                motion_samples = scan_motion(ffmpeg, str(video_path))
            LOG.info("motion scan: %d samples", len(motion_samples))
        except Exception as exc:
            warning = f"motion analysis failed; using base scores: {exc}"
            LOG.warning(warning)
    if video.audio_codec:
        try:
            with phase("scanning audio activity", LOG):
                audio_samples = scan_audio_activity(ffmpeg, str(video_path))
            LOG.info("audio scan: %d samples", len(audio_samples))
        except Exception as exc:
            warning = f"audio activity analysis failed; using visual scores: {exc}"
            LOG.warning(warning)

    # Subtitles feed the ranking score, so they must be resolved before
    # candidates are generated. An explicit --srt always wins; otherwise the
    # source's own embedded text track is used when it has one. Extracting once
    # here lets the render stage reuse the same cues.
    source_subtitles: list[Subtitle] | None = None
    if args.srt:
        try:
            source_subtitles, warnings = parse_srt_file(args.srt)
            for warning in warnings:
                LOG.warning("SRT: %s", warning)
            LOG.info("ranking with %d cues from %s", len(source_subtitles), args.srt)
        except OSError as exc:
            LOG.warning("could not read SRT %s: %s", args.srt, exc)
    else:
        try:
            with phase("extracting embedded subtitles", LOG):
                source_subtitles = extract_source_subtitles(
                    ffmpeg,
                    str(video_path),
                    has_text_subtitles=source_has_text_subtitles,
                    build_command=extract_subtitles_command,
                    work_dir=output_dir,
                )
        except Exception as exc:  # ranking without subtitles is still valid
            warning = f"embedded subtitle extraction failed; ranking without transcripts: {exc}"
            LOG.warning(warning)
        if source_subtitles is None:
            LOG.info("no embedded text subtitle track; ranking on audio and video only")
        else:
            LOG.info("ranking with %d cues from the embedded subtitle track", len(source_subtitles))

    with phase("generating candidates", LOG):
        candidates = generate_candidates(video, config, scene_points, motion_samples, audio_samples)
    LOG.info("generated %d candidates", len(candidates))
    checkpoint = load_checkpoint(output_dir) if config.keep_existing else None
    if checkpoint:
        checkpoint_input = checkpoint.get("input", {})
        stat = video_path.stat()
        checkpoint_config = checkpoint.get("config", {})
        compatible = (
            checkpoint_input.get("path") == str(video_path)
            and checkpoint_input.get("size") == stat.st_size
            and checkpoint_input.get("mtime_ns") == stat.st_mtime_ns
            and checkpoint_config.get("window_seconds") == config.window_seconds
            and checkpoint_config.get("stride_seconds") == config.stride_seconds
            and checkpoint_config.get("max_candidates") == config.max_candidates
            and checkpoint_config.get("ignore_static_scenes", False) == config.ignore_static_scenes
            and checkpoint_config.get("static_motion_threshold", 1.5) == config.static_motion_threshold
        )
        if compatible:
            saved_candidates = {item.get("id"): item for item in checkpoint.get("candidates", [])}
            for candidate in candidates:
                saved = saved_candidates.get(candidate.id)
                if saved:
                    for field in ("thumbnail", "preview", "gif", "errors", "selected", "status", "transcript", "score", "reasons"):
                        if field in saved:
                            setattr(candidate, field, saved[field])
        else:
            LOG.warning(
                "existing checkpoint does not match the current input or settings; starting fresh"
            )

    if source_subtitles:
        with phase("scoring transcripts", LOG):
            for candidate in candidates:
                associate_subtitles(candidate, source_subtitles)
        boosted = sum("transcript_density" in candidate.reasons for candidate in candidates)
        rescued = sum("dialogue_rescue" in candidate.reasons for candidate in candidates)
        LOG.info(
            "transcript signal: %d/%d candidates scored, %d rescued from static filtering",
            boosted,
            len(candidates),
            rescued,
        )
        if not boosted:
            LOG.warning("subtitles were found but no candidate overlaps them; check their timing")
        # Text density changes ranking, so keep stable IDs but order by score for assets.
        candidates.sort(key=lambda item: (-item.score, item.start_ms))

    with phase("selecting highlights", LOG):
        auto_selection = select_auto_candidates(candidates, config, source_duration_ms=video.duration_ms)
    if not auto_selection.selected:
        raise ValueError("no usable highlight candidates were found")
    LOG.info(
        "auto selection: %d clips, %.0fs of %.0fs budget (%s)",
        len(auto_selection.selected),
        auto_selection.total_duration_ms / 1000,
        auto_selection.budget_ms / 1000,
        ALGORITHM_VERSION,
    )
    (output_dir / "auto-selection.json").write_text(
        json.dumps({
            **selection_summary(candidates, config, source_duration_ms=video.duration_ms),
            "config": config.to_dict(),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Selection is automatic and the work directory is removed on success, so
    # there is no manual review step to serve assets to.
    total_steps = 1
    if progress_callback:
        progress_callback(0, total_steps, "准备素材")

    render_total = len(auto_selection.selected) + 1

    def render_progress(completed: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(total_steps + completed, total_steps + render_total, message)

    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("highlight generation canceled")
    LOG.info("rendering reel: %d clips", len(auto_selection.selected))
    with phase("rendering highlight reel", LOG):
        render_selected_video(
            ffmpeg,
            str(video_path),
            auto_selection.selected,
            output_dir,
            has_audio=bool(video.audio_codec),
            progress_callback=render_progress,
            output_path=final_output_path,
            source_subtitles=source_subtitles,
            source_duration_ms=video.duration_ms,
        )
    LOG.info("wrote %s", final_output_path)
    verification_note = ""
    if getattr(args, "verify", "fast") != "off":
        verification_note = _verify_output(args, final_output_path, output_dir)
    # A successful one-click run leaves only the final MP4 beside the source.
    if cleanup_work_dir:
        shutil.rmtree(output_dir, ignore_errors=True)
    # The GUI needs the summary after the work dir is gone, so it is returned
    # rather than re-read from a report file that cleanup may have removed.
    return final_output_path, verification_note


def _verify_output(
    args: argparse.Namespace, output_path: Path, work_dir: Path
) -> str:
    """Validate a freshly rendered reel and report through the log.

    Verification is best-effort: a defect in the reel is worth reporting, but
    it must never turn a successful render into a failed run.

    The report is written inside ``work_dir`` rather than beside the reel. The
    reel lives next to the source video, so writing there would leave a stray
    file in the user's library that no later cleanup step owns; ``work_dir`` is
    removed on a one-click run and kept when ``--output-dir`` asks for it.

    Returns a one-line summary for callers that render after cleanup, or an
    empty string when verification could not run.
    """
    mode = getattr(args, "verify", "fast")
    sample_count, window_s = (6, 5.0) if mode == "fast" else (24, 8.0)
    try:
        with phase("verifying highlight reel", LOG):
            report = verify_reel(
                output_path, sample_count=sample_count, sample_window_s=window_s
            )
    except Exception as exc:  # verification must never break a good render
        LOG.warning("reel verification could not run: %s", exc)
        return ""
    for check in report.checks:
        level = {"error": logging.ERROR, "warning": logging.WARNING}.get(
            check.severity, logging.INFO
        )
        LOG.log(level, "verify %s: %s", check.name, check.detail)
    if report.ok:
        LOG.info(
            "reel verified (%d warning(s) to review)",
            len(report.warnings),
        )
    else:
        LOG.error(
            "reel verification found %d problem(s); the output may be unusable",
            len(report.errors),
        )
    try:
        (work_dir / "verification.txt").write_text(
            summarise(report) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        LOG.debug("could not write verification report: %s", exc)
    if report.errors:
        return "Check failed: " + "; ".join(
            check.detail for check in report.errors[:2]
        )
    if report.warnings:
        return f"Checked, {len(report.warnings)} warning(s) to review"
    return "Checked, no problems found"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output_path, _note = run(args)
    except (OSError, ValueError, NotADirectoryError, InterruptedError) as exc:
        parser.error(str(exc))
        return 2
    print(f"Generated reel: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

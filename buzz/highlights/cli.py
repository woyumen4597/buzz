"""Standalone CLI for generating reviewable highlight candidates."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Callable

from .exporter import export_outputs, load_checkpoint, serve_result_page, write_checkpoint
from .media import (
    find_tools,
    generate_gif,
    generate_preview,
    generate_thumbnail,
    probe_media,
)
from .models import HighlightConfig
from .subtitles import associate_subtitles, parse_srt_file
from .windows import generate_candidates, scan_scene_changes

LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="buzz-highlights",
        description="Generate human-reviewable video highlight candidates.",
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
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-limit", type=int, default=20)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--open", action="store_true", dest="open_html")
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
        no_previews=args.no_previews,
        gif=args.gif,
        gif_limit=args.gif_limit,
        keep_existing=args.keep_existing,
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
    output_dir = (args.output_dir or video_path.with_name(f"{video_path.stem}_highlights")).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise NotADirectoryError(str(output_dir))

    ffmpeg, _ = find_tools()
    video = probe_media(str(video_path))
    config = _config(args)
    warnings: list[str] = []
    errors: list[str] = []
    scene_points: list[int] | None = None
    scene_enabled = not config.no_scene_detection
    if not config.no_scene_detection:
        try:
            scene_points = scan_scene_changes(ffmpeg, str(video_path), config.scene_threshold)
        except Exception as exc:  # scene detection is explicitly best-effort
            warning = f"scene detection failed; using fixed windows: {exc}"
            warnings.append(warning)
            LOG.warning(warning)
            scene_enabled = False

    candidates = generate_candidates(video, config, scene_points)
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
            warnings.append("existing checkpoint does not match the current input or settings; starting fresh")

    if args.srt:
        try:
            subtitles, subtitle_warnings = parse_srt_file(args.srt)
            warnings.extend(subtitle_warnings)
            for candidate in candidates:
                associate_subtitles(candidate, subtitles)
            # Text density changes ranking, so keep stable IDs but order by score for assets.
            candidates.sort(key=lambda item: (-item.score, item.start_ms))
        except OSError as exc:
            warnings.append(f"could not read SRT: {exc}")

    thumbnail_dir = output_dir / "thumbnails"
    preview_dir = output_dir / "previews"
    total_steps = max(1, len(candidates) * (1 if config.no_previews else 2))
    completed_steps = 0
    for candidate in candidates:
        thumb_path = thumbnail_dir / f"{candidate.id}.jpg"
        if candidate.thumbnail and thumb_path.exists():
            completed_steps += 1
        if not config.no_previews:
            preview_path = preview_dir / f"{candidate.id}.mp4"
            if candidate.preview and preview_path.exists():
                completed_steps += 1
    if progress_callback:
        progress_callback(completed_steps, total_steps, "继续生成" if completed_steps else "准备素材")
    for candidate in candidates:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("highlight generation canceled")
        thumb_path = thumbnail_dir / f"{candidate.id}.jpg"
        generate_thumbnail(candidate, str(video_path), thumb_path, ffmpeg, config.keep_existing)
        if candidate.thumbnail:
            candidate.thumbnail = thumb_path.relative_to(output_dir).as_posix()
        completed_steps += 1
        if progress_callback:
            progress_callback(completed_steps, total_steps, f"缩略图 {candidate.id}")
        write_checkpoint(output_dir, str(video_path), video, config, candidates)
        if not config.no_previews:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("highlight generation canceled")
            preview_path = preview_dir / f"{candidate.id}.mp4"
            generate_preview(
                candidate,
                str(video_path),
                preview_path,
                ffmpeg,
                config.keep_existing,
                has_audio=bool(video.audio_codec),
            )
            if candidate.preview:
                candidate.preview = preview_path.relative_to(output_dir).as_posix()
            completed_steps += 1
            if progress_callback:
                progress_callback(completed_steps, total_steps, f"预览 {candidate.id}")
            write_checkpoint(output_dir, str(video_path), video, config, candidates)
        if config.gif and candidates.index(candidate) < config.gif_limit:
            gif_path = preview_dir / f"{candidate.id}.gif"
            generate_gif(candidate, str(video_path), gif_path, ffmpeg, config.keep_existing)
            if candidate.gif:
                candidate.gif = gif_path.relative_to(output_dir).as_posix()

    export_config = config
    if not scene_enabled:
        export_config = HighlightConfig(**{**config.to_dict(), "no_scene_detection": True})
    render_server = None
    if args.open_html or getattr(args, "serve_html", False):
        render_server, _thread = serve_result_page(
            output_dir, candidates, str(video_path), ffmpeg, has_audio=bool(video.audio_codec)
        )
    render_endpoint = f"http://127.0.0.1:{render_server.server_port}/render" if render_server else None
    export_outputs(
        output_dir, candidates, video, export_config, str(video_path), errors, warnings,
        render_endpoint=render_endpoint,
    )
    if render_server:
        result_url = f"http://127.0.0.1:{render_server.server_port}"
        (output_dir / ".highlight-server-url").write_text(result_url + "\n", encoding="utf-8")
        if args.open_html:
            webbrowser.open(result_url + "/index.html")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output_dir = run(args)
    except (OSError, ValueError, NotADirectoryError, InterruptedError) as exc:
        parser.error(str(exc))
        return 2
    print(f"Generated {output_dir / 'index.html'} ({len(list(output_dir.glob('thumbnails/*.jpg')))} thumbnails)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

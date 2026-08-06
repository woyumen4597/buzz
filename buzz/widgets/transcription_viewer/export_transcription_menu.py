import json
import logging
import os
import shutil
import sys
import tempfile
from collections import deque

from PyQt6.QtCore import QProcess, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QMenu, QMessageBox, QProgressDialog, QWidget

from buzz.db.entity.transcription import Transcription
from buzz.db.service.transcription_service import TranscriptionService
from buzz.ffmpeg_video_player import _find_ffmpeg, _find_ffprobe
from buzz.locale import _
from buzz.plugins.post_processing import FnRunnable
from buzz.transcriber.file_transcriber import (
    download_video_to_cache,
    find_cached_video,
    is_video_file,
    write_output,
)
from buzz.transcriber.transcriber import (
    OutputFormat,
    Segment,
)


# Video export modes (not part of OutputFormat enum). Format: <label>, <ext>.
MP4_BURNED = "MP4_BURNED"
MP4_SOFT = "MP4_SOFT"
VIDEO_MODES = {
    MP4_BURNED: (_("MP4 - Burned Subtitles"), "mp4"),
    MP4_SOFT: (_("MP4 - Soft Subtitles"), "mp4"),
}

# Codecs the MP4 muxer accepts with stream copy; anything else must be
# transcoded (libx264 video / AAC audio) to stay playable.
_COPY_VIDEO_CODECS = {"h264", "avc1", "hevc", "h265"}
_COPY_AUDIO_CODECS = {"aac", "mp3", "ac3", "alac", "opus"}


class TranslateExportMenu(QMenu):
    """Dropdown for the "Translate & Export" button. Each option picks a
    subtitle format (SRT/TXT/VTT, original or translation) or an MP4 mode and
    hands it back to the viewer, which runs translation first when needed."""

    option_selected = pyqtSignal(str, str)  # (format value or MP4 mode, segment_key)

    def __init__(self, has_translation: bool, parent: QWidget | None = None):
        super().__init__(parent)

        for output_format in OutputFormat:
            self._add_action(
                f"{output_format.value.upper()} - {_('Translation')}",
                output_format.value,
                "translation",
            )
        for output_format in OutputFormat:
            self._add_action(
                f"{output_format.value.upper()} - {_('Text')}",
                output_format.value,
                "text",
            )
        for mode, (label, _ext) in VIDEO_MODES.items():
            self._add_action(
                f"{label} - {_('Translation')}",
                mode,
                "translation",
            )
        self.triggered.connect(self._on_triggered)

    def _add_action(self, text: str, format_or_mode: str, segment_key: str):
        action = QAction(text, self)
        action.setData({"format_or_mode": format_or_mode, "segment_key": segment_key})
        self.addAction(action)

    def _on_triggered(self, action: QAction):
        data = action.data()
        if isinstance(data, dict):
            self.option_selected.emit(data["format_or_mode"], data["segment_key"])


class ExportTranscriptionMenu(QMenu):
    translation_export_requested = pyqtSignal(str, str)

    def __init__(
        self,
        transcription: Transcription,
        transcription_service: TranscriptionService,
        has_translation: bool,
        translation: pyqtSignal,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.transcription = transcription
        self.transcription_service = transcription_service

        translation.connect(self.on_translation_available)

        text_label = _("Text")
        translation_label = _("Translation")
        self.text_actions = [
            QAction(text=f"{output_format.value.upper()} - {text_label}", parent=self)
            for output_format in OutputFormat
        ]
        self.translation_actions = [
            QAction(text=f"{output_format.value.upper()} - {translation_label}", parent=self)
            for output_format in OutputFormat
        ]
        for action in self.translation_actions:
            action.setVisible(has_translation)

        # ponytail: burned subtitles always available (use text if no translation);
        # soft subtitles can embed any srt, so always available too.
        # segment_key stored in action data (not parsed from text — the format name
        # contains " - " which breaks split); flipped to translation once it loads.
        self.video_burned_action = QAction(
            text=f"{VIDEO_MODES[MP4_BURNED][0]} - {translation_label}",
            parent=self,
        )
        self.video_burned_action.setData(
            {"mode": MP4_BURNED, "segment_key": "translation"}
        )
        # URL imports keep only the extracted audio; the original video is
        # fetched on demand, so video export is still offered for them.
        is_video = bool(
            is_video_file(self.transcription.file) or self.transcription.url
        )
        self.video_burned_action.setVisible(is_video)
        self.video_soft_action = QAction(
            text=f"{VIDEO_MODES[MP4_SOFT][0]} - {translation_label}",
            parent=self,
        )
        self.video_soft_action.setData(
            {"mode": MP4_SOFT, "segment_key": "translation"}
        )
        self.video_soft_action.setVisible(is_video)

        actions = (
            self.text_actions
            + self.translation_actions
            + [self.video_burned_action, self.video_soft_action]
        )
        self.addActions(actions)
        self.triggered.connect(self.on_menu_triggered)

    @staticmethod
    def extract_format_and_segment_key(action_text: str):
        parts = action_text.split('-', 1)
        head = parts[0].strip()
        label = parts[1].strip() if len(parts) > 1 else None
        segment_key = 'translation' if label == _('Translation') else 'text'
        return head, segment_key

    def on_translation_available(self):
        for action in self.translation_actions:
            action.setVisible(True)
        self.video_burned_action.setText(
            f"{VIDEO_MODES[MP4_BURNED][0]} - {_('Translation')}"
        )
        self.video_burned_action.setData({"mode": MP4_BURNED, "segment_key": "translation"})
        self.video_soft_action.setText(
            f"{VIDEO_MODES[MP4_SOFT][0]} - {_('Translation')}"
        )
        self.video_soft_action.setData({"mode": MP4_SOFT, "segment_key": "translation"})

    def _get_segments(self) -> list[Segment]:
        return [
            Segment(
                start=segment.start_time,
                end=segment.end_time,
                text=segment.text,
                translation=segment.translation)
            for segment in self.transcription_service.get_transcription_segments(
                transcription_id=self.transcription.id_as_uuid
            )
        ]

    def on_menu_triggered(self, action: QAction):
        action_text = action.text()

        # Video export branch — keyed off the action's data, not the split text
        # (the text contains " - " which would otherwise break head parsing).
        action_data = action.data()
        if isinstance(action_data, dict) and action_data.get("mode") in VIDEO_MODES:
            # segment_key stored in data (原文用 'text', 译文用 'translation'); never
            # parse it from action_text — the format name itself contains " - ".
            segment_key = action_data.get("segment_key", "text")
            if segment_key == "translation":
                self.translation_export_requested.emit(action_data["mode"], segment_key)
            else:
                self._export_video(action_data["mode"], segment_key)
            return

        head, segment_key = self.extract_format_and_segment_key(action_text)

        # Subtitle file branch (original behavior)
        output_format = OutputFormat(head.lower())
        default_path = self.transcription.get_output_file_path(
            output_format=output_format
        )
        (output_file_path, nil) = QFileDialog.getSaveFileName(
            self,
            _("Save File"),
            default_path,
            _("Text files") + f" (*.{output_format.value})",
        )
        if output_file_path == "":
            return
        write_output(
            path=output_file_path,
            segments=self._get_segments(),
            output_format=output_format,
            segment_key=segment_key,
        )

    def _export_video(self, mode: str, segment_key: str):
        """Burn or soft-embed subtitles into an MP4 via ffmpeg (async, QProcess)."""
        media_file = self.transcription.file
        if not media_file:
            QMessageBox.critical(
                self, _("Export Video"),
                _("Source media not found: {}").format(media_file),
            )
            return

        # URL imports only keep the extracted audio; fetch the original video
        # (cached next to the audio) before exporting.
        if not is_video_file(media_file) and self.transcription.url:
            cached = find_cached_video(media_file)
            if cached:
                media_file = cached
            else:
                self._download_video_for_export(mode, segment_key, media_file)
                return

        if not os.path.isfile(media_file):
            QMessageBox.critical(
                self, _("Export Video"),
                _("Source media not found: {}").format(media_file),
            )
            return

        self._start_video_export(mode, segment_key, media_file)

    def _download_video_for_export(self, mode: str, segment_key: str, audio_path: str):
        prog = QProgressDialog(
            _("Downloading original video..."), _("Cancel"), 0, 0, self
        )
        prog.setWindowTitle(_("Export Video"))
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)

        runnable = FnRunnable(
            lambda: download_video_to_cache(self.transcription.url, audio_path)
        )
        runnable.signals.finished.connect(
            lambda: self._on_video_downloaded(prog, mode, segment_key)
        )
        runnable.signals.error.connect(
            lambda error: self._on_video_download_failed(prog, error)
        )
        runnable.signals.finished.connect(prog.close)
        runnable.signals.error.connect(prog.close)
        prog.canceled.connect(prog.close)
        QThreadPool.globalInstance().start(runnable)

    def _on_video_downloaded(self, prog: QProgressDialog, mode: str, segment_key: str):
        if prog.wasCanceled():
            return
        media_file = self.transcription.file
        if media_file:
            cached = find_cached_video(media_file)
            if cached:
                self._start_video_export(mode, segment_key, cached)
                return
        QMessageBox.critical(
            self, _("Export Video"),
            _("Failed to locate the downloaded video."),
        )

    def _on_video_download_failed(self, prog: QProgressDialog, error: str):
        if prog.wasCanceled():
            return
        QMessageBox.critical(
            self, _("Export Video"),
            _("Failed to download the original video: {}").format(error),
        )

    def _start_video_export(self, mode: str, segment_key: str, media_file: str):
        """Validate the target path, write the temp SRT and probe the source."""
        label, ext = VIDEO_MODES[mode]
        base = os.path.splitext(os.path.basename(media_file))[0]
        name = "translated" if segment_key == "translation" else "subtitled"
        default_path = os.path.join(
            os.path.dirname(media_file), f"{base}.{name}.{ext}"
        )
        filter_str = f"{_('Video files')} (*.{ext})"
        output_file_path, _ignored = QFileDialog.getSaveFileName(
            self, _("Save File"), default_path, filter_str
        )
        if output_file_path == "":
            return

        try:
            same_output = os.path.samefile(media_file, output_file_path)
        except OSError:
            same_output = (
                os.path.normcase(os.path.realpath(media_file))
                == os.path.normcase(os.path.realpath(output_file_path))
            )
        if same_output:
            QMessageBox.critical(
                self, _("Export Video"),
                _("The output path must differ from the source file."),
            )
            return

        ffmpeg = _find_ffmpeg()
        ffprobe = _find_ffprobe()
        if not (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)) or not (
            shutil.which(ffprobe) or os.path.isfile(ffprobe)
        ):
            QMessageBox.critical(
                self, _("Export Video"),
                _("ffmpeg and ffprobe are required to export video subtitles. "
                  "Install ffmpeg and make sure it is available in PATH."),
            )
            return

        segments = self._get_segments()
        srt_fd, srt_path = tempfile.mkstemp(suffix=".srt", text=True)
        os.close(srt_fd)
        try:
            write_output(
                path=srt_path,
                segments=segments,
                output_format=OutputFormat.SRT,
                segment_key=segment_key,
            )
        except Exception as e:
            logging.error("Writing temp SRT failed: %s", e)
            self._cleanup_srt(srt_path)
            QMessageBox.critical(
                self, _("Export Video"),
                _("Failed to prepare subtitles: {}").format(e),
            )
            return

        # Language metadata is only written when it is known to match the text
        # (the source language); translated subtitles have no known target.
        subtitle_language = None
        if segment_key == "text" and self.transcription.language:
            subtitle_language = self.transcription.language

        prog = QProgressDialog(
            _("Preparing video export..."), _("Cancel"), 0, 0, self
        )
        prog.setWindowTitle(_("Export Video"))
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.srt_path = srt_path
        proc.output_path = output_file_path
        proc.part_path = f"{output_file_path}.part"
        proc.duration_ms = 0
        proc.dialog = prog
        proc.mode = mode
        proc.media_file = media_file
        proc.subtitle_language = subtitle_language
        proc.output_tail = deque(maxlen=20)
        proc.canceled = False
        prog.canceled.connect(lambda: self._on_export_canceled(proc))
        proc.readyRead.connect(lambda: self._on_ffmpeg_output(proc))
        proc.finished.connect(lambda code, st: self._on_ffmpeg_finished(proc, code, st))
        proc.errorOccurred.connect(
            lambda err: self._on_ffmpeg_error(proc, err)
        )

        self._start_probe(proc, media_file)

    def _start_probe(self, proc: QProcess, media_file: str):
        """Probe the source with ffprobe asynchronously so the GUI stays
        responsive; ffmpeg starts once the codec info is known."""
        ffprobe = _find_ffprobe()
        probe = QProcess(proc)
        probe.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.probe = probe
        probe.finished.connect(
            lambda code, st: self._on_probe_finished(proc, code, st)
        )
        probe.errorOccurred.connect(lambda err: self._on_probe_failed(proc, err))
        probe.start(
            ffprobe,
            ["-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", media_file],
        )

    def _on_probe_finished(self, proc: QProcess, exit_code: int, _exit_status):
        if getattr(proc, "canceled", False):
            self._cleanup_srt(proc.srt_path)
            return
        probe = getattr(proc, "probe", None)
        raw = bytes(probe.readAll()).decode("utf-8", errors="replace") if probe else ""
        info = {}
        if exit_code == 0 and raw.strip():
            try:
                info = self._parse_probe_json(raw)
            except Exception as exc:
                logging.warning("Failed to parse ffprobe output: %s", exc)
        if not info:
            logging.warning(
                "ffprobe failed for %s (exit %s): %s",
                proc.media_file,
                exit_code,
                "\n".join(raw.splitlines()[-10:]) if raw else "(no output)",
            )
        self._continue_after_probe(proc, info)

    def _on_probe_failed(self, proc: QProcess, err):
        if getattr(proc, "canceled", False):
            self._cleanup_srt(proc.srt_path)
            return
        probe = getattr(proc, "probe", None)
        raw = bytes(probe.readAll()).decode("utf-8", errors="replace") if probe else ""
        logging.warning("ffprobe could not start (%s): %s", err, raw[-1000:])
        self._continue_after_probe(proc, {})

    @staticmethod
    def _parse_probe_json(raw: str) -> dict:
        info = json.loads(raw)
        duration_s = float(info.get("format", {}).get("duration", 0) or 0)
        video_codec = audio_codec = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video" and video_codec is None:
                video_codec = stream.get("codec_name")
                if not duration_s:
                    duration_s = float(stream.get("duration", 0) or 0)
            elif stream.get("codec_type") == "audio" and audio_codec is None:
                audio_codec = stream.get("codec_name")
        return {
            "duration_ms": int(duration_s * 1000),
            "video_codec": video_codec,
            "audio_codec": audio_codec,
        }

    def _continue_after_probe(self, proc: QProcess, info: dict):
        proc.duration_ms = int(info.get("duration_ms") or 0)
        video_codec = (info.get("video_codec") or "").lower()
        audio_codec = (info.get("audio_codec") or "").lower()
        # Known-incompatible codecs (e.g. AV1) transcode directly; unknown
        # codecs (probe failed) try stream copy and fall back on failure.
        copy_video = video_codec in _COPY_VIDEO_CODECS if video_codec else True
        copy_audio = audio_codec in _COPY_AUDIO_CODECS if audio_codec else True
        proc.dialog.setRange(0, 100)
        proc.dialog.setValue(0)
        self._start_ffmpeg(
            proc, proc.mode, proc.media_file, proc.srt_path, proc.output_path,
            copy_video=copy_video, copy_audio=copy_audio,
            subtitle_language=proc.subtitle_language,
        )

    def _on_export_canceled(self, proc: QProcess):
        proc.canceled = True
        probe = getattr(proc, "probe", None)
        if probe is not None and probe.state() != QProcess.ProcessState.NotRunning:
            probe.kill()
        if proc.state() != QProcess.ProcessState.NotRunning:
            proc.kill()

    @staticmethod
    def _start_ffmpeg(
        proc: QProcess,
        mode: str,
        media_file: str,
        srt_path: str,
        output_path: str,
        copy_video: bool = False,
        copy_audio: bool = False,
        subtitle_language: str | None = None,
    ):
        ffmpeg = _find_ffmpeg()
        # Escape backslashes/colons so the subtitles filter parses the path.
        srt_filter_arg = srt_path.replace(chr(92), chr(92) * 2)
        srt_filter_arg = srt_filter_arg.replace(":", chr(92) + ":")
        if os.name == "nt":
            srt_escaped = srt_path.replace(chr(92), "/" + chr(92))
            srt_filter_arg = srt_escaped

        part_path = f"{output_path}.part"
        proc.part_path = part_path

        if mode == MP4_BURNED:
            cmd = [
                ffmpeg, "-y", "-i", media_file,
                "-vf", f"subtitles={srt_filter_arg}",
                "-c:a", "copy",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-progress", "pipe:1", "-nostats",
                "-f", "mp4",
                part_path,
            ]
        else:  # MP4_SOFT
            # Prefer stream copy (no re-encode) when the source codecs are MP4
            # compatible; fall back to H.264/AAC transcoding when they are not.
            copy_cmd = ExportTranscriptionMenu._soft_subtitle_cmd(
                ffmpeg, media_file, srt_path, part_path,
                copy_video=True, copy_audio=True,
                subtitle_language=subtitle_language,
            )
            transcode_cmd = ExportTranscriptionMenu._soft_subtitle_cmd(
                ffmpeg, media_file, srt_path, part_path,
                copy_video=False, copy_audio=False,
                subtitle_language=subtitle_language,
            )
            proc.copy_cmd = copy_cmd
            proc.transcode_cmd = transcode_cmd
            proc.try_copy = copy_video
            cmd = copy_cmd if copy_video else transcode_cmd
        proc.start(cmd[0], cmd[1:])

    @staticmethod
    def _soft_subtitle_cmd(
        ffmpeg: str,
        media_file: str,
        srt_path: str,
        part_path: str,
        copy_video: bool,
        copy_audio: bool,
        subtitle_language: str | None = None,
    ) -> list:
        cmd = [
            ffmpeg, "-y", "-i", media_file, "-i", srt_path,
            # Explicitly map only the first video/audio stream so pre-existing
            # subtitle streams are dropped; our SRT is the only subtitle stream.
            "-map", "0:v:0", "-map", "0:a:0?", "-map", "1:0",
            "-c:s:0", "mov_text",
            "-progress", "pipe:1", "-nostats",
        ]
        if subtitle_language:
            cmd += ["-metadata:s:s:0", f"language={subtitle_language}"]
        if copy_video:
            cmd += ["-c:v", "copy"]
        else:
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast"]
        if copy_audio:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += ["-f", "mp4", part_path]
        return cmd

    def _on_ffmpeg_output(self, proc: QProcess):
        data = bytes(proc.readAll()).decode("utf-8", errors="replace")
        tail = getattr(proc, "output_tail", None)
        for line in data.splitlines():
            line = line.strip()
            if tail is not None:
                tail.append(line)
            if line.startswith("out_time_ms=") or line.startswith("out_time_us="):
                try:
                    us = int(line.split("=", 1)[1])
                    ms = max(0, us // 1000)
                except ValueError:
                    continue
                if proc.duration_ms:
                    pct = min(100, int(ms * 100 / proc.duration_ms))
                    proc.dialog.setValue(pct)
            elif line == "progress=end":
                proc.dialog.setValue(100)

    def _on_ffmpeg_finished(self, proc: QProcess, exit_code: int, _exit_status):
        part_path = getattr(proc, "part_path", f"{proc.output_path}.part")
        if exit_code == 0 and os.path.isfile(part_path):
            self._cleanup_srt(proc.srt_path)
            try:
                os.replace(part_path, proc.output_path)
            except OSError:
                pass
            else:
                self._clear_quarantine(proc.output_path)
                proc.dialog.close()
                QMessageBox.information(
                    self, _("Export Video"),
                    _("Saved: {}").format(proc.output_path),
                )
                return

        if getattr(proc, "canceled", False):
            self._cleanup_srt(proc.srt_path)
            self._cleanup_part(part_path)
            proc.dialog.close()
            return

        # A failed stream-copy attempt usually fails fast (incompatible muxer
        # combination); retry once with full transcoding instead of giving up.
        if (
            exit_code != 0
            and getattr(proc, "try_copy", False)
            and getattr(proc, "transcode_cmd", None)
        ):
            logging.warning(
                "Soft subtitle stream copy failed (exit %s), falling back to "
                "transcoding", exit_code,
            )
            self._cleanup_part(part_path)
            proc.try_copy = False
            proc.start(proc.transcode_cmd[0], proc.transcode_cmd[1:])
            return

        self._cleanup_srt(proc.srt_path)
        self._cleanup_part(part_path)
        proc.dialog.close()
        tail = "\n".join(getattr(proc, "output_tail", ()))
        if tail:
            logging.warning(
                "ffmpeg export failed (exit %s):\n%s", exit_code, tail
            )
        QMessageBox.critical(
            self, _("Export Video"),
            _("ffmpeg failed (exit code {}).\n\n{}").format(
                exit_code, tail or _("No error details available.")
            ),
        )

    def _on_ffmpeg_error(self, proc: QProcess, err):
        self._cleanup_srt(proc.srt_path)
        self._cleanup_part(getattr(proc, "part_path", f"{proc.output_path}.part"))
        if getattr(proc, "canceled", False):
            proc.dialog.close()
            return
        proc.dialog.close()
        QMessageBox.critical(
            self, _("Export Video"),
            _("Could not start ffmpeg: {}").format(err),
        )

    @staticmethod
    def _clear_quarantine(path: str):
        # Generated locally; don't carry a source download's macOS quarantine
        # marker onto the exported video.
        if sys.platform != "darwin":
            return
        try:
            os.removexattr(path, "com.apple.quarantine")
        except (AttributeError, OSError):
            pass

    @staticmethod
    def _cleanup_srt(srt_path):
        try:
            os.remove(srt_path)
        except OSError:
            pass

    @staticmethod
    def _cleanup_part(part_path):
        try:
            os.remove(part_path)
        except OSError:
            pass

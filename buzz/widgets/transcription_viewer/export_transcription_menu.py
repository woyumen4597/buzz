import logging
import os
import tempfile

from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QProcess, pyqtSignal
from PyQt6.QtWidgets import QWidget, QMenu, QFileDialog, QMessageBox, QProgressDialog

from buzz.db.entity.transcription import Transcription
from buzz.db.service.transcription_service import TranscriptionService
from buzz.ffmpeg_video_player import _find_ffmpeg, probe_video
from buzz.locale import _
from buzz.transcriber.file_transcriber import is_video_file, write_output
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


class ExportTranscriptionMenu(QMenu):
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
            text=f"{VIDEO_MODES[MP4_BURNED][0]} - {translation_label}"
            if has_translation
            else f"{VIDEO_MODES[MP4_BURNED][0]} - {text_label}",
            parent=self,
        )
        self.video_burned_action.setData({"mode": MP4_BURNED, "segment_key": "translation" if has_translation else "text"})
        is_video = is_video_file(self.transcription.file) if self.transcription.file else False
        self.video_burned_action.setVisible(is_video)
        self.video_soft_action = QAction(
            text=f"{VIDEO_MODES[MP4_SOFT][0]} - {translation_label}"
            if has_translation
            else f"{VIDEO_MODES[MP4_SOFT][0]} - {text_label}",
            parent=self,
        )
        self.video_soft_action.setData({"mode": MP4_SOFT, "segment_key": "translation" if has_translation else "text"})
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
        if not media_file or not os.path.isfile(media_file):
            QMessageBox.critical(
                self, _("Export Video"),
                _("Source media not found: {}").format(media_file or ""),
            )
            return

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

        info = self._probe_info(media_file)
        duration_ms = info.get("duration_ms") or 0
        video_codec = (info.get("video_codec") or "").lower()
        audio_codec = (info.get("audio_codec") or "").lower()
        # Known-incompatible codecs (e.g. AV1) transcode directly; unknown
        # codecs (probe failed) try stream copy and fall back on failure.
        copy_video = video_codec in _COPY_VIDEO_CODECS if video_codec else True
        copy_audio = audio_codec in _COPY_AUDIO_CODECS if audio_codec else True

        prog = QProgressDialog(
            _("Exporting video with subtitles..."), _("Cancel"), 0, 100, self
        )
        prog.setWindowTitle(_("Export Video"))
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.srt_path = srt_path
        proc.output_path = output_file_path
        proc.part_path = f"{output_file_path}.part"
        proc.duration_ms = duration_ms
        proc.dialog = prog
        prog.canceled.connect(proc.kill)

        proc.readyRead.connect(lambda: self._on_ffmpeg_output(proc))
        proc.finished.connect(lambda code, st: self._on_ffmpeg_finished(proc, code, st))
        proc.errorOccurred.connect(
            lambda err: self._on_ffmpeg_error(proc, err)
        )

        self._start_ffmpeg(
            proc, mode, media_file, srt_path, output_file_path,
            copy_video=copy_video, copy_audio=copy_audio,
        )

    @staticmethod
    def _start_ffmpeg(
        proc: QProcess,
        mode: str,
        media_file: str,
        srt_path: str,
        output_path: str,
        copy_video: bool = False,
        copy_audio: bool = False,
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
                part_path,
            ]
        else:  # MP4_SOFT
            # Prefer stream copy (no re-encode) when the source codecs are MP4
            # compatible; fall back to H.264/AAC transcoding when they are not.
            copy_cmd = ExportTranscriptionMenu._soft_subtitle_cmd(
                ffmpeg, media_file, srt_path, part_path,
                copy_video=True, copy_audio=True,
            )
            transcode_cmd = ExportTranscriptionMenu._soft_subtitle_cmd(
                ffmpeg, media_file, srt_path, part_path,
                copy_video=False, copy_audio=False,
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
    ) -> list:
        cmd = [
            ffmpeg, "-y", "-i", media_file, "-i", srt_path,
            "-map", "0", "-map", "1",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=chi",
            "-progress", "pipe:1", "-nostats",
            part_path,
        ]
        if copy_video:
            cmd += ["-c:v", "copy"]
        else:
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast"]
        if copy_audio:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        return cmd

    def _on_ffmpeg_output(self, proc: QProcess):
        data = bytes(proc.readAll()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            line = line.strip()
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
                proc.dialog.close()
                QMessageBox.information(
                    self, _("Export Video"),
                    _("Saved: {}").format(proc.output_path),
                )
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
        QMessageBox.critical(
            self, _("Export Video"),
            _("ffmpeg failed (exit code {}). Check that ffmpeg is installed and the output path is writable.").format(exit_code),
        )

    def _on_ffmpeg_error(self, proc: QProcess, _err):
        self._cleanup_srt(proc.srt_path)
        self._cleanup_part(getattr(proc, "part_path", f"{proc.output_path}.part"))
        proc.dialog.close()
        QMessageBox.critical(
            self, _("Export Video"),
            _("Could not start ffmpeg. Make sure ffmpeg is installed and available in PATH."),
        )

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

    @staticmethod
    def _probe_info(media_file: str) -> dict:
        try:
            return probe_video(media_file)
        except Exception:
            return {}

import json
import pathlib
import uuid

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from pytestqt.qtbot import QtBot

from buzz.db.entity.transcription import Transcription
from buzz.db.entity.transcription_segment import TranscriptionSegment
from buzz.model_loader import ModelType, WhisperModelSize
from buzz.transcriber.transcriber import Task
from buzz.widgets.transcription_viewer.export_transcription_menu import (
    ExportTranscriptionMenu,
    MP4_BURNED,
    MP4_SOFT,
)
from tests.audio import test_audio_path


class TranslationSignal(QObject):
    translation = pyqtSignal(str, int)


def probe_payload(duration="12.0", video_codec="h264", audio_codec="aac") -> bytes:
    streams = [{"codec_type": "video", "codec_name": video_codec}]
    if audio_codec:
        streams.append({"codec_type": "audio", "codec_name": audio_codec})
    return json.dumps(
        {"format": {"duration": duration}, "streams": streams}
    ).encode()


class TestExportTranscriptionMenu:
    @pytest.fixture()
    def transcription(
        self, transcription_dao, transcription_segment_dao
    ) -> Transcription:
        id = uuid.uuid4()
        transcription_dao.insert(
            Transcription(
                id=str(id),
                status="completed",
                file=test_audio_path,
                task=Task.TRANSCRIBE.value,
                model_type=ModelType.WHISPER.value,
                whisper_model_size=WhisperModelSize.TINY.value,
            )
        )
        transcription_segment_dao.insert(TranscriptionSegment(40, 299, "Bien", "", str(id)))
        transcription_segment_dao.insert(
            TranscriptionSegment(299, 329, "venue dans", "", str(id))
        )

        return transcription_dao.find_by_id(str(id))

    def _make_widget(self, transcription, transcription_service):
        translation_signal = TranslationSignal()
        return ExportTranscriptionMenu(
            transcription,
            transcription_service,
            False,
            translation_signal.translation,
        )

    def _export_and_probe(self, widget, mode, segment_key, mocker, payload):
        """Run _export_video with a mocked async probe and finish the probe;
        returns the ffmpeg proc and the mocked _start_ffmpeg."""
        mocker.patch(
            "buzz.widgets.transcription_viewer.export_transcription_menu.shutil.which",
            return_value="/usr/bin/ffmpeg",
        )
        start_ffmpeg = mocker.patch.object(widget, "_start_ffmpeg")
        captured = {}
        mocker.patch.object(
            widget,
            "_start_probe",
            side_effect=lambda proc, media: captured.__setitem__("proc", proc),
        )
        widget._export_video(mode, segment_key)
        proc = captured["proc"]
        proc.probe = mocker.Mock()
        proc.probe.readAll.return_value = payload
        widget._on_probe_finished(proc, 0, None)
        return proc, start_ffmpeg

    def test_should_export_segments(
        self,
        tmp_path: pathlib.Path,
        qtbot: QtBot,
        transcription,
        transcription_service,
        shortcuts,
        mocker,
    ):
        output_file_path = tmp_path / "whisper.txt"
        mocker.patch(
            "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(output_file_path), ""),
        )

        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)

        widget.actions()[0].trigger()

        with open(output_file_path, encoding="utf-8") as output_file:
            assert "Bien venue dans" in output_file.read()

    def test_hides_video_actions_for_audio(
        self, qtbot: QtBot, transcription, transcription_service
    ):
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)

        assert not widget.video_burned_action.isVisible()
        assert not widget.video_soft_action.isVisible()

    def test_shows_video_actions_for_url_import(
        self, qtbot: QtBot, transcription, transcription_service
    ):
        transcription.url = "https://example.com/video"
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)

        assert widget.video_burned_action.isVisible()
        assert widget.video_soft_action.isVisible()

    @pytest.mark.parametrize(
        ("segment_key", "suffix"),
        [("text", "subtitled"), ("translation", "translated")],
    )
    def test_video_default_name_and_rejects_source_path(
        self,
        tmp_path: pathlib.Path,
        qtbot: QtBot,
        transcription,
        transcription_service,
        mocker,
        segment_key,
        suffix,
    ):
        source_path = tmp_path / "movie.mkv"
        source_path.touch()
        transcription.file = str(source_path)
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        mocker.patch.object(widget, "_get_segments", return_value=[])
        get_save_file_name = mocker.patch(
            "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.mp4"), ""),
        )

        proc, start_ffmpeg = self._export_and_probe(
            widget, MP4_BURNED, segment_key, mocker, probe_payload()
        )

        assert get_save_file_name.call_args.args[2] == str(
            tmp_path / f"movie.{suffix}.mp4"
        )
        start_ffmpeg.assert_called_once()
        widget._cleanup_srt(start_ffmpeg.call_args.args[3])

        get_save_file_name.return_value = (str(source_path), "")
        critical = mocker.patch(
            "PyQt6.QtWidgets.QMessageBox.critical"
        )
        start_ffmpeg.reset_mock()

        widget._export_video(MP4_BURNED, segment_key)

        critical.assert_called_once()
        start_ffmpeg.assert_not_called()

    def test_renames_part_on_success_and_cleans_on_failure(
        self, tmp_path: pathlib.Path, qtbot: QtBot, transcription, transcription_service, mocker
    ):
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        information = mocker.patch(
            "PyQt6.QtWidgets.QMessageBox.information"
        )
        critical = mocker.patch("PyQt6.QtWidgets.QMessageBox.critical")

        part_path = tmp_path / "movie.mp4.part"
        output_path = tmp_path / "movie.mp4"
        srt_path = tmp_path / "movie.srt"
        part_path.write_bytes(b"video")
        srt_path.write_text("1", encoding="utf-8")
        proc = mocker.Mock(
            part_path=str(part_path),
            output_path=str(output_path),
            srt_path=str(srt_path),
            dialog=mocker.Mock(),
            try_copy=False,
            transcode_cmd=None,
            canceled=False,
            output_tail=[],
        )

        widget._on_ffmpeg_finished(proc, 0, None)

        assert output_path.read_bytes() == b"video"
        assert not part_path.exists()
        assert not srt_path.exists()
        information.assert_called_once()

        part_path.write_bytes(b"partial")
        srt_path.write_text("1", encoding="utf-8")
        widget._on_ffmpeg_finished(proc, 1, None)

        assert not part_path.exists()
        assert not srt_path.exists()
        critical.assert_called_once()

    def test_soft_subtitle_copy_command_uses_copy_for_compatible_codecs(
        self,
        tmp_path: pathlib.Path,
        qtbot: QtBot,
        transcription,
        transcription_service,
        mocker,
    ):
        source_path = tmp_path / "movie.mp4"
        source_path.touch()
        transcription.file = str(source_path)
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        mocker.patch.object(widget, "_get_segments", return_value=[])
        mocker.patch(
            "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.mp4"), ""),
        )

        proc, start_ffmpeg = self._export_and_probe(
            widget, MP4_SOFT, "text", mocker,
            probe_payload(video_codec="h264", audio_codec="aac"),
        )

        assert start_ffmpeg.call_args.kwargs == {
            "copy_video": True, "copy_audio": True, "subtitle_language": None,
        }
        widget._cleanup_srt(start_ffmpeg.call_args.args[3])

    def test_soft_subtitle_command_transcodes_incompatible_codecs(
        self,
        tmp_path: pathlib.Path,
        qtbot: QtBot,
        transcription,
        transcription_service,
        mocker,
    ):
        source_path = tmp_path / "movie.mkv"
        source_path.touch()
        transcription.file = str(source_path)
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        mocker.patch.object(widget, "_get_segments", return_value=[])
        mocker.patch(
            "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.mp4"), ""),
        )

        proc, start_ffmpeg = self._export_and_probe(
            widget, MP4_SOFT, "text", mocker,
            probe_payload(video_codec="av1", audio_codec="flac"),
        )

        assert start_ffmpeg.call_args.kwargs == {
            "copy_video": False, "copy_audio": False, "subtitle_language": None,
        }
        widget._cleanup_srt(start_ffmpeg.call_args.args[3])

    @pytest.mark.parametrize("payload", [b"", b"not json"])
    def test_probe_failure_falls_back_to_copy_then_transcode(
        self,
        tmp_path: pathlib.Path,
        qtbot: QtBot,
        transcription,
        transcription_service,
        mocker,
        payload,
    ):
        source_path = tmp_path / "movie.mp4"
        source_path.touch()
        transcription.file = str(source_path)
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        mocker.patch.object(widget, "_get_segments", return_value=[])
        mocker.patch(
            "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.mp4"), ""),
        )

        proc, start_ffmpeg = self._export_and_probe(
            widget, MP4_SOFT, "text", mocker, payload
        )

        # Unknown codecs: try stream copy first, the fallback handles failure.
        assert start_ffmpeg.call_args.kwargs["copy_video"] is True
        assert start_ffmpeg.call_args.kwargs["copy_audio"] is True
        widget._cleanup_srt(start_ffmpeg.call_args.args[3])

    def test_soft_subtitle_language_metadata_only_for_source_text(
        self,
        tmp_path: pathlib.Path,
        qtbot: QtBot,
        transcription,
        transcription_service,
        mocker,
    ):
        source_path = tmp_path / "movie.mp4"
        source_path.touch()
        transcription.file = str(source_path)
        transcription.language = "ja"
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        mocker.patch.object(widget, "_get_segments", return_value=[])
        mocker.patch(
            "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.mp4"), ""),
        )

        proc, start_ffmpeg = self._export_and_probe(
            widget, MP4_SOFT, "text", mocker, probe_payload()
        )
        assert start_ffmpeg.call_args.kwargs["subtitle_language"] == "ja"
        widget._cleanup_srt(start_ffmpeg.call_args.args[3])

        start_ffmpeg.reset_mock()
        proc, start_ffmpeg = self._export_and_probe(
            widget, MP4_SOFT, "translation", mocker, probe_payload()
        )
        assert start_ffmpeg.call_args.kwargs["subtitle_language"] is None
        widget._cleanup_srt(start_ffmpeg.call_args.args[3])

    def test_soft_subtitle_cmd_builds_copy_and_transcode_variants(
        self, tmp_path: pathlib.Path, qtbot: QtBot, transcription, transcription_service, mocker
    ):
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        proc = mocker.Mock()
        media = str(tmp_path / "in.mp4")
        srt = str(tmp_path / "subs.srt")
        out = str(tmp_path / "out.mp4")

        # Compatible source: stream copy, no re-encode.
        widget._start_ffmpeg(
            proc, MP4_SOFT, media, srt, out,
            copy_video=True, copy_audio=True,
        )
        assert proc.try_copy is True
        copy_cmd = proc.start.call_args.args[1]
        assert copy_cmd[copy_cmd.index("-c:v") + 1] == "copy"
        assert "libx264" not in copy_cmd
        assert copy_cmd[copy_cmd.index("-c:a") + 1] == "copy"
        # Explicit stream maps: source subtitles are never copied over.
        maps = [copy_cmd[i + 1] for i, arg in enumerate(copy_cmd) if arg == "-map"]
        assert maps == ["0:v:0", "0:a:0?", "1:0"]
        # No language metadata when the target language is unknown.
        assert "-metadata:s:s:0" not in copy_cmd

        # Incompatible source: H.264 + AAC transcode.
        proc.start.reset_mock()
        widget._start_ffmpeg(
            proc, MP4_SOFT, media, srt, out,
            copy_video=False, copy_audio=False,
        )
        assert proc.try_copy is False
        transcode_cmd = proc.start.call_args.args[1]
        assert "libx264" in transcode_cmd
        assert "aac" in transcode_cmd
        assert proc.transcode_cmd[1:] == transcode_cmd

        # Known language: metadata written only for source text.
        proc.start.reset_mock()
        widget._start_ffmpeg(
            proc, MP4_SOFT, media, srt, out,
            copy_video=True, copy_audio=True, subtitle_language="ja",
        )
        lang_cmd = proc.start.call_args.args[1]
        assert "language=ja" in lang_cmd

    def test_soft_subtitle_copy_failure_falls_back_to_transcode(
        self, tmp_path: pathlib.Path, qtbot: QtBot, transcription, transcription_service, mocker
    ):
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        srt_path = tmp_path / "movie.srt"
        srt_path.write_text("1", encoding="utf-8")
        part_path = tmp_path / "out.mp4.part"
        part_path.write_bytes(b"partial")
        transcode_cmd = ["ffmpeg", "-y", "-i", "in.mp4", "-c:v", "libx264"]
        proc = mocker.Mock(
            part_path=str(part_path),
            output_path=str(tmp_path / "out.mp4"),
            srt_path=str(srt_path),
            dialog=mocker.Mock(),
            try_copy=True,
            transcode_cmd=transcode_cmd,
            canceled=False,
            output_tail=[],
        )

        widget._on_ffmpeg_finished(proc, 1, None)

        proc.start.assert_called_once_with("ffmpeg", transcode_cmd[1:])
        assert proc.try_copy is False
        # The temp SRT stays alive for the transcode retry
        assert srt_path.exists()
        assert not part_path.exists()
        proc.dialog.close.assert_not_called()

    def test_canceled_export_cleans_up_without_error_dialog(
        self, tmp_path: pathlib.Path, qtbot: QtBot, transcription, transcription_service, mocker
    ):
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        critical = mocker.patch("PyQt6.QtWidgets.QMessageBox.critical")
        part_path = tmp_path / "out.mp4.part"
        part_path.write_bytes(b"partial")
        srt_path = tmp_path / "movie.srt"
        srt_path.write_text("1", encoding="utf-8")
        proc = mocker.Mock(
            part_path=str(part_path),
            output_path=str(tmp_path / "out.mp4"),
            srt_path=str(srt_path),
            dialog=mocker.Mock(),
            try_copy=False,
            transcode_cmd=None,
            canceled=True,
        )

        widget._on_ffmpeg_finished(proc, 1, None)

        assert not part_path.exists()
        assert not srt_path.exists()
        critical.assert_not_called()

    def test_ffmpeg_failure_includes_stderr_tail(
        self, tmp_path: pathlib.Path, qtbot: QtBot, transcription, transcription_service, mocker
    ):
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        critical = mocker.patch("PyQt6.QtWidgets.QMessageBox.critical")
        srt_path = tmp_path / "movie.srt"
        srt_path.write_text("1", encoding="utf-8")
        part_path = tmp_path / "out.mp4.part"
        part_path.write_bytes(b"partial")
        proc = mocker.Mock(
            part_path=str(part_path),
            output_path=str(tmp_path / "out.mp4"),
            srt_path=str(srt_path),
            dialog=mocker.Mock(),
            try_copy=False,
            transcode_cmd=None,
            canceled=False,
            output_tail=["[error] No such file or directory"],
        )

        widget._on_ffmpeg_finished(proc, 1, None)

        assert "[error] No such file or directory" in critical.call_args.args[2]

    def test_url_export_downloads_video_on_demand(
        self, tmp_path: pathlib.Path, qtbot: QtBot, transcription, transcription_service, mocker
    ):
        wav_path = tmp_path / "title.wav"
        wav_path.touch()
        transcription.file = str(wav_path)
        transcription.url = "https://example.com/video"
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        download = mocker.patch.object(widget, "_download_video_for_export")

        widget._export_video(MP4_SOFT, "text")

        download.assert_called_once_with(MP4_SOFT, "text", str(wav_path))

    def test_url_export_uses_cached_video(
        self, tmp_path: pathlib.Path, qtbot: QtBot, transcription, transcription_service, mocker
    ):
        wav_path = tmp_path / "title.wav"
        wav_path.touch()
        video_path = tmp_path / "title.mp4"
        video_path.touch()
        transcription.file = str(wav_path)
        transcription.url = "https://example.com/video"
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        start_video_export = mocker.patch.object(widget, "_start_video_export")

        widget._export_video(MP4_SOFT, "text")

        start_video_export.assert_called_once_with(MP4_SOFT, "text", str(video_path))

    def test_video_download_failure_shows_real_error(
        self, qtbot: QtBot, transcription, transcription_service, mocker
    ):
        widget = self._make_widget(transcription, transcription_service)
        qtbot.add_widget(widget)
        critical = mocker.patch("PyQt6.QtWidgets.QMessageBox.critical")
        prog = mocker.Mock()
        prog.wasCanceled.return_value = False

        widget._on_video_download_failed(prog, "HTTP Error 403: Forbidden")

        critical.assert_called_once()
        assert "HTTP Error 403: Forbidden" in critical.call_args.args[2]

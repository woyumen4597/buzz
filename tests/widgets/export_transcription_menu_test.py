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
)
from tests.audio import test_audio_path


class TranslationSignal(QObject):
    translation = pyqtSignal(str, int)


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

        translation_signal = TranslationSignal()

        widget = ExportTranscriptionMenu(
            transcription,
            transcription_service,
            False,
            translation_signal.translation
        )
        qtbot.add_widget(widget)

        widget.actions()[0].trigger()

        with open(output_file_path, encoding="utf-8") as output_file:
            assert "Bien venue dans" in output_file.read()

    def test_hides_video_actions_for_audio(
        self, qtbot: QtBot, transcription, transcription_service
    ):
        translation_signal = TranslationSignal()
        widget = ExportTranscriptionMenu(
            transcription,
            transcription_service,
            False,
            translation_signal.translation,
        )
        qtbot.add_widget(widget)

        assert not widget.video_burned_action.isVisible()
        assert not widget.video_soft_action.isVisible()

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
        translation_signal = TranslationSignal()
        widget = ExportTranscriptionMenu(
            transcription,
            transcription_service,
            False,
            translation_signal.translation,
        )
        qtbot.add_widget(widget)
        mocker.patch.object(widget, "_get_segments", return_value=[])
        start_ffmpeg = mocker.patch.object(widget, "_start_ffmpeg")
        get_save_file_name = mocker.patch(
            "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "out.mp4"), ""),
        )

        widget._export_video(MP4_BURNED, segment_key)

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
        translation_signal = TranslationSignal()
        widget = ExportTranscriptionMenu(
            transcription,
            transcription_service,
            False,
            translation_signal.translation,
        )
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

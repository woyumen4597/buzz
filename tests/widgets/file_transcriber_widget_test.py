from unittest.mock import Mock, patch

from PyQt6.QtCore import QSettings, Qt
from pytestqt.qtbot import QtBot

from buzz.model_loader import ModelType, TranscriptionModel
from buzz.transcriber.transcriber import FileTranscriptionOptions, TranscriptionOptions
from buzz.widgets.preferences_dialog.models.file_transcription_preferences import (
    FileTranscriptionPreferences,
)
from buzz.widgets.transcriber.file_transcriber_widget import FileTranscriberWidget
from buzz.widgets.transcriber.file_transcription_form_widget import (
    FileTranscriptionFormWidget,
)
from tests.audio import test_audio_path


class TestFileTranscriberWidget:
    def test_file_translation_defaults_to_ai_enabled(self, tmp_path):
        settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)

        preferences = FileTranscriptionPreferences.load(settings)
        assert preferences.enable_llm_translation is True

        settings.setValue("enable_llm_translation", False)
        assert FileTranscriptionPreferences.load(settings).enable_llm_translation is False

    def test_vad_defaults_off_and_an_explicit_choice_is_persisted(self, tmp_path):
        settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)

        preferences = FileTranscriptionPreferences.load(settings)
        assert preferences.use_vad is False
        assert preferences.use_vad_explicit is False

        preferences.use_vad = True
        preferences.use_vad_explicit = True
        preferences.save(settings)

        assert FileTranscriptionPreferences.load(settings).use_vad is True

    def test_a_stored_vad_default_does_not_pin_the_old_value(self, tmp_path):
        """The form saves its options on close, so a settings file can hold a
        value the user never chose. Whatever it holds, a value nobody chose must
        keep following the current default instead of pinning the old one."""
        settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
        settings.setValue("use_vad", True)
        settings.setValue("settings_version", 2)

        assert FileTranscriptionPreferences.load(settings).use_vad is False

        # Saving the loaded state must not turn that default into a choice.
        FileTranscriptionPreferences.load(settings).save(settings)

        reloaded = FileTranscriptionPreferences.load(settings)
        assert reloaded.use_vad is False
        assert reloaded.use_vad_explicit is False

    def test_version_2_vad_is_migrated_back_to_the_current_default(self, tmp_path):
        """Version 2 turned VAD on for installations that had never touched the
        checkbox, so a stored on without use_vad_explicit belongs to one of
        those and must not survive as if the user had asked for it."""
        settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
        settings.setValue("use_vad", True)
        settings.setValue("settings_version", 2)

        assert FileTranscriptionPreferences.load(settings).use_vad is False

    def test_vad_checkbox_reports_a_user_choice(self, qtbot: QtBot):
        form = FileTranscriptionFormWidget(
            transcription_options=TranscriptionOptions(),
            file_transcription_options=FileTranscriptionOptions(),
        )
        qtbot.add_widget(form)

        chosen = []
        form.use_vad_chosen.connect(lambda: chosen.append(True))

        # Populating the form is not a choice; moving the checkbox is.
        form.use_vad_checkbox.setChecked(False)
        assert chosen == []

        form.use_vad_checkbox.setChecked(True)
        assert chosen == [True]

    def test_vad_checkbox_updates_transcription_options(self, qtbot: QtBot):
        form = FileTranscriptionFormWidget(
            transcription_options=TranscriptionOptions(),
            file_transcription_options=FileTranscriptionOptions(),
        )
        qtbot.add_widget(form)

        assert form.use_vad_checkbox.isChecked() is False
        form.use_vad_checkbox.setChecked(True)
        assert form.transcription_options.use_vad is True

    def test_should_set_window_title(self, qtbot: QtBot):
        widget = FileTranscriberWidget(
            file_paths=[test_audio_path],
        )
        qtbot.add_widget(widget)
        assert widget.windowTitle() == "whisper-french.mp3"

    def test_should_emit_triggered_event(self, qtbot: QtBot):
        widget = FileTranscriberWidget(
            file_paths=[test_audio_path],
        )
        qtbot.add_widget(widget)

        mock_triggered = Mock()
        widget.triggered.connect(mock_triggered)

        with qtbot.wait_signal(widget.triggered, timeout=30 * 1000):
            qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)

        (
            transcription_options,
            file_transcription_options,
            model_path,
        ) = mock_triggered.call_args[0][0]
        assert transcription_options.language is None
        assert file_transcription_options.file_paths == [test_audio_path]
        assert len(model_path) > 0

    def test_on_model_loaded_empty_path_shows_error_for_local_model(self, qtbot: QtBot):
        widget = FileTranscriberWidget(file_paths=[test_audio_path])
        qtbot.add_widget(widget)
        widget.transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.FASTER_WHISPER)
        )

        mock_triggered = Mock()
        widget.triggered.connect(mock_triggered)

        with patch("buzz.widgets.transcriber.file_transcriber_widget.show_model_download_error_dialog") as mock_err, \
             patch.object(widget, "save_preferences"):
            widget.on_model_loaded("")
            mock_err.assert_called_once()
            mock_triggered.assert_not_called()

    def test_on_model_loaded_empty_path_allowed_for_openai_api(self, qtbot: QtBot):
        widget = FileTranscriberWidget(file_paths=[test_audio_path])
        qtbot.add_widget(widget)
        widget.transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.OPEN_AI_WHISPER_API)
        )

        mock_triggered = Mock()
        widget.triggered.connect(mock_triggered)

        with patch("buzz.widgets.transcriber.file_transcriber_widget.show_model_download_error_dialog") as mock_err, \
             patch.object(widget, "save_preferences"):
            widget.on_model_loaded("")
            mock_err.assert_not_called()
            mock_triggered.assert_called_once()

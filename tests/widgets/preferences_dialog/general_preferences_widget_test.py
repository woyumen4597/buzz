from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QMessageBox, QLineEdit, QCheckBox

from buzz.locale import _
from buzz.settings.settings import DEFAULT_TRANSCRIPTION_CONCURRENCY, Settings
from buzz.widgets.preferences_dialog.general_preferences_widget import (
    GeneralPreferencesWidget, ValidateOpenAIApiKeyJob
)
from buzz.translator import CHAT_COMPLETIONS_PROTOCOL, RESPONSES_PROTOCOL


class TestGeneralPreferencesWidget:
    def test_transcription_concurrency_preferences(self, qtbot):
        settings = Settings()
        key = Settings.Key.TRANSCRIPTION_CONCURRENCY
        previous_value = settings.value(key, DEFAULT_TRANSCRIPTION_CONCURRENCY)
        settings.set_value(key, DEFAULT_TRANSCRIPTION_CONCURRENCY)

        try:
            widget = GeneralPreferencesWidget()
            qtbot.add_widget(widget)

            assert (
                widget.transcription_concurrency_spin_box.value()
                == DEFAULT_TRANSCRIPTION_CONCURRENCY
            )

            widget.transcription_concurrency_spin_box.setValue(4)

            assert settings.value(key, DEFAULT_TRANSCRIPTION_CONCURRENCY) == 4
        finally:
            settings.set_value(key, previous_value)

    def test_should_disable_test_button_if_no_api_key(self, qtbot, mocker):
        mocker.patch(
            "buzz.widgets.preferences_dialog.general_preferences_widget.get_password",
            return_value="",
        )

        widget = GeneralPreferencesWidget()
        qtbot.add_widget(widget)

        test_button = widget.findChild(QPushButton, "TestOpenAIAPIKeyButton")
        assert isinstance(test_button, QPushButton)

        assert test_button.text() == _("Test")
        assert not test_button.isEnabled()

        line_edit = widget.findChild(QLineEdit, "OpenAIAPIKeyLineEdit")
        assert isinstance(line_edit, QLineEdit)
        line_edit.setText("123")

        assert test_button.isEnabled()

    def test_should_test_openai_api_key(self, qtbot, mocker, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", CHAT_COMPLETIONS_PROTOCOL)
        mocker.patch(
            "buzz.widgets.preferences_dialog.general_preferences_widget.get_password",
            return_value="wrong-api-key",
        )
        mock_response = mocker.Mock(status_code=401)
        mock_response.json.return_value = {
            "error": {"message": "Incorrect API key provided"}
        }
        mocker.patch(
            "buzz.widgets.preferences_dialog.general_preferences_widget.httpx.post",
            return_value=mock_response,
        )

        widget = GeneralPreferencesWidget()
        qtbot.add_widget(widget)

        test_button = widget.findChild(QPushButton, "TestOpenAIAPIKeyButton")
        assert isinstance(test_button, QPushButton)

        test_button.click()

        message_box_warning_mock = mocker.Mock()
        QMessageBox.warning = message_box_warning_mock

        def mock_called():
            message_box_warning_mock.assert_called()
            assert message_box_warning_mock.call_args[0][1] == _("OpenAI API Key Test")
            assert (
                    message_box_warning_mock.call_args[0][2]
                    == "OpenAI API key test failed (Incorrect API key provided). "
                       "Check the API key, base URL and model name."
            )

        qtbot.waitUntil(mock_called)

    def test_recording_export_preferences(self, qtbot, mocker):
        mocker.patch(
            "PyQt6.QtWidgets.QFileDialog.getExistingDirectory",
            return_value="/path/to/export/folder",
        )

        widget = GeneralPreferencesWidget()
        qtbot.add_widget(widget)

        browse_button = widget.findChild(QPushButton, "RecordingExportFolderBrowseButton")
        checkbox = widget.findChild(QCheckBox, "EnableRecordingExportCheckbox")

        browse_button_enabled = browse_button.isEnabled()

        qtbot.mouseClick(widget.export_enabled_checkbox, Qt.MouseButton.LeftButton)
        checkbox.setChecked(not browse_button_enabled)

        assert browse_button.isEnabled() != browse_button_enabled

        qtbot.mouseClick(widget.recording_export_folder_browse_button, Qt.MouseButton.LeftButton)

        assert widget.recording_export_folder_line_edit.text() == "/path/to/export/folder"

        assert widget.settings.value(
            key=widget.settings.Key.RECORDING_TRANSCRIBER_EXPORT_ENABLED,
            default_value=False) != browse_button_enabled
        assert widget.settings.value(
            key=widget.settings.Key.RECORDING_TRANSCRIBER_EXPORT_FOLDER,
            default_value='/home/user/documents') == '/path/to/export/folder'

    def test_openai_base_url_preferences(self, qtbot, mocker):
        widget = GeneralPreferencesWidget()
        qtbot.add_widget(widget)

        settings = Settings()

        openai_base_url = settings.value(
            key=Settings.Key.CUSTOM_OPENAI_BASE_URL, default_value=""
        )

        assert openai_base_url == ""
        assert widget.custom_openai_base_url_line_edit.text() == ""

        widget.custom_openai_base_url_line_edit.setText("http://localhost:11434/v1")

        updated_openai_base_url = settings.value(
            key=Settings.Key.CUSTOM_OPENAI_BASE_URL, default_value=""
        )

        assert updated_openai_base_url == "http://localhost:11434/v1"

    def test_translation_api_protocol_preferences(self, qtbot):
        settings = Settings()
        key = Settings.Key.TRANSLATION_API_PROTOCOL
        previous_value = settings.value(key, CHAT_COMPLETIONS_PROTOCOL)
        settings.set_value(key, CHAT_COMPLETIONS_PROTOCOL)

        try:
            widget = GeneralPreferencesWidget()
            qtbot.add_widget(widget)
            widget.translation_api_protocol_combo_box.setCurrentIndex(
                widget.translation_api_protocol_combo_box.findData(RESPONSES_PROTOCOL)
            )

            assert settings.value(key, "") == RESPONSES_PROTOCOL
        finally:
            settings.set_value(key, previous_value)


class TestTestOpenAIApiKeyJob:
    # No error = success
    def test_run_success(self, mocker):
        mock_response = mocker.Mock(status_code=200)
        mock_post = mocker.patch(
            'buzz.widgets.preferences_dialog.general_preferences_widget.httpx.post',
            return_value=mock_response,
        )
        mocker.patch('buzz.settings.settings.Settings.value', return_value="") # No custom base URL

        job = ValidateOpenAIApiKeyJob(api_key="test_key")
        mock_success = mocker.Mock()
        mock_failed = mocker.Mock()
        job.signals.success.connect(mock_success)
        job.signals.failed.connect(mock_failed)

        job.run()

        mock_success.assert_called_once()
        mock_failed.assert_not_called()
        mock_post.assert_called_once()

    def test_run_responses(self, mocker, monkeypatch):
        monkeypatch.setenv("BUZZ_TRANSLATION_API_PROTOCOL", RESPONSES_PROTOCOL)
        mock_response = mocker.Mock(status_code=200)
        mock_post = mocker.patch(
            "buzz.widgets.preferences_dialog.general_preferences_widget.httpx.post",
            return_value=mock_response,
        )
        mocker.patch("buzz.settings.settings.Settings.value", return_value="")

        job = ValidateOpenAIApiKeyJob(api_key="test_key")
        job.run()

        mock_post.assert_called_once_with(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": "Bearer test_key",
                "Content-Type": "application/json",
            },
            json={"model": "", "max_output_tokens": 8, "input": "hi"},
            timeout=20,
        )

    # Has error = failure
    def test_run_authentication_error(self, mocker):
        mock_response = mocker.Mock(status_code=401)
        mock_response.json.return_value = {
            "error": {"message": "Incorrect API key provided"}
        }
        mock_post = mocker.patch(
            'buzz.widgets.preferences_dialog.general_preferences_widget.httpx.post',
            return_value=mock_response,
        )
        mocker.patch('buzz.settings.settings.Settings.value', return_value="") # No custom base URL

        job = ValidateOpenAIApiKeyJob(api_key="wrong_key")
        mock_success = mocker.Mock()
        mock_failed = mocker.Mock()
        job.signals.success.connect(mock_success)
        job.signals.failed.connect(mock_failed)

        job.run()

        mock_success.assert_not_called()
        mock_failed.assert_called_once_with(
            "OpenAI API key test failed (Incorrect API key provided). "
            "Check the API key, base URL and model name."
        )
        mock_post.assert_called_once()

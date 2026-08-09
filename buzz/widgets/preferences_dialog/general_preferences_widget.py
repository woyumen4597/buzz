import os
import re
import httpx
from typing import Optional
from platformdirs import user_documents_dir

from PyQt6.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool, QLocale, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QFormLayout,
    QPushButton,
    QMessageBox,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QSpinBox,
    QComboBox,
    QLabel,
    QSizePolicy,
    QGroupBox,
    QScrollArea,
    QFrame,
)
from PyQt6.QtGui import QIcon

from buzz.settings.settings import (
    DEFAULT_TRANSCRIPTION_CONCURRENCY,
    DEFAULT_TRANSLATION_BATCH_SIZE,
    MAX_TRANSCRIPTION_CONCURRENCY,
    MAX_TRANSLATION_BATCH_SIZE,
    MIN_TRANSCRIPTION_CONCURRENCY,
    MIN_TRANSLATION_BATCH_SIZE,
    Settings,
)
from buzz.store.keyring_store import get_password, Key
from buzz.transcriber.download_cookies import supported_browsers
from buzz.widgets.line_edit import LineEdit
from buzz.widgets.openai_api_key_line_edit import OpenAIAPIKeyLineEdit
from buzz.locale import _
from buzz.widgets.icon import INFO_ICON_PATH
from buzz.settings.recording_transcriber_mode import RecordingTranscriberMode
from buzz.translator import (
    CHAT_COMPLETIONS_PROTOCOL,
    DEFAULT_OPENAI_BASE_URL,
    RESPONSES_PROTOCOL,
    _chat_completions_url,
    _responses_url,
    _translation_api_protocol,
)

BASE64_PATTERN = re.compile(r'^[A-Za-z0-9+/=_-]*$')

ui_locales = {
    "en_US": _("English"),
    "ca_ES": _("Catalan"),
    "da_DK": _("Danish"),
    "nl": _("Dutch"),
    "de_DE": _("German"),
    "es_ES": _("Spanish"),
    "it_IT": _("Italian"),
    "ja_JP": _("Japanese"),
    "lv_LV": _("Latvian"),
    "pl_PL": _("Polish"),
    "pt_BR": _("Portuguese (Brazil)"),
    "ru": _("Russian"),
    "uk_UA": _("Ukrainian"),
    "zh_CN": _("Chinese (Simplified)"),
    "zh_TW": _("Chinese (Traditional)")
}


class GeneralPreferencesWidget(QWidget):
    openai_api_key_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.settings = Settings()

        self.openai_api_key = get_password(Key.OPENAI_API_KEY)

        # One box per concern, and the whole thing scrolls. The old flat form was
        # taller than the dialog, so Qt compressed the rows on top of each other.
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)

        def add_group(title: str) -> QFormLayout:
            group = QGroupBox(title, content)
            form = QFormLayout(group)
            # Fields share a single width instead of each one picking its own.
            form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            form.setLabelAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)
            content_layout.addWidget(group)
            return form

        interface_layout = add_group(_("Interface"))
        transcription_layout = add_group(_("Transcription"))
        openai_layout = add_group(_("OpenAI API"))
        url_layout = add_group(_("URL download"))
        export_layout = add_group(_("Export and live recording"))

        self.ui_language_combo_box = QComboBox(self)
        self.ui_language_combo_box.addItems(ui_locales.values())
        system_locale = self.settings.value(Settings.Key.UI_LOCALE, QLocale().name())
        locale_index = 0
        for i, (code, language) in enumerate(ui_locales.items()):
            if code == system_locale:
                locale_index = i
                break
        self.ui_language_combo_box.setCurrentIndex(locale_index)
        self.ui_language_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ui_language_combo_box.currentIndexChanged.connect(self.on_language_changed)

        self.ui_locale_layout = QHBoxLayout()
        self.ui_locale_layout.setContentsMargins(0, 0, 0, 0)
        self.ui_locale_layout.setSpacing(0)
        self.ui_locale_layout.addWidget(self.ui_language_combo_box)

        self.load_note_tooltip_icon = QLabel()
        self.load_note_tooltip_icon.setPixmap(QIcon(INFO_ICON_PATH).pixmap(23, 23))
        self.load_note_tooltip_icon.setToolTip(_("Restart required!"))
        self.load_note_tooltip_icon.setVisible(False)
        self.ui_locale_layout.addWidget(self.load_note_tooltip_icon)

        interface_layout.addRow(_("Ui Language"), self.ui_locale_layout)

        self.font_size_spin_box = QSpinBox(self)
        self.font_size_spin_box.setMinimum(8)
        self.font_size_spin_box.setMaximum(32)
        self.font_size_spin_box.setValue(self.font().pointSize())
        self.font_size_spin_box.valueChanged.connect(self.on_font_size_changed)
        # Numeric fields keep their natural size instead of stretching the row.
        self.font_size_spin_box.setMaximumWidth(90)
        self.font_size_spin_box.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )

        interface_layout.addRow(_("Font Size"), self.font_size_spin_box)

        self.transcription_concurrency_spin_box = QSpinBox(self)
        self.transcription_concurrency_spin_box.setRange(
            MIN_TRANSCRIPTION_CONCURRENCY, MAX_TRANSCRIPTION_CONCURRENCY
        )
        self.transcription_concurrency_spin_box.setValue(
            self.settings.value(
                Settings.Key.TRANSCRIPTION_CONCURRENCY,
                DEFAULT_TRANSCRIPTION_CONCURRENCY,
            )
        )
        self.transcription_concurrency_spin_box.setToolTip(_("Restart required!"))
        self.transcription_concurrency_spin_box.valueChanged.connect(
            self.on_transcription_concurrency_changed
        )
        self.transcription_concurrency_spin_box.setMaximumWidth(90)
        self.transcription_concurrency_spin_box.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        transcription_layout.addRow(
            _("Transcription concurrency"),
            self.transcription_concurrency_spin_box,
        )

        self.openai_api_key_line_edit = OpenAIAPIKeyLineEdit(self.openai_api_key, self)
        self.openai_api_key_line_edit.key_changed.connect(
            self.on_openai_api_key_changed
        )
        self.openai_api_key_line_edit.focus_out.connect(self.on_openai_api_key_focus_out)
        self.openai_api_key_line_edit.setMinimumWidth(200)
        self.openai_api_key_line_edit.setObjectName("OpenAIAPIKeyLineEdit")

        self.test_openai_api_key_button = QPushButton(_("Test"))
        self.test_openai_api_key_button.setObjectName("TestOpenAIAPIKeyButton")
        self.test_openai_api_key_button.clicked.connect(
            self.on_click_test_openai_api_key_button
        )
        self.update_test_openai_api_key_button()

        # Inline, so the button keeps its natural size instead of being stretched
        # across an otherwise empty row.
        openai_api_key_row = QHBoxLayout()
        openai_api_key_row.addWidget(self.openai_api_key_line_edit)
        openai_api_key_row.addWidget(self.test_openai_api_key_button)

        openai_layout.addRow(_("OpenAI API key"), openai_api_key_row)

        self.custom_openai_base_url = self.settings.value(
            key=Settings.Key.CUSTOM_OPENAI_BASE_URL, default_value=""
        )

        self.custom_openai_base_url_line_edit = LineEdit(self.custom_openai_base_url, self)
        self.custom_openai_base_url_line_edit.textChanged.connect(
            self.on_custom_openai_base_url_changed
        )
        self.custom_openai_base_url_line_edit.setMinimumWidth(200)
        self.custom_openai_base_url_line_edit.setPlaceholderText("https://api.openai.com/v1")
        openai_layout.addRow(
            _("OpenAI base url"), self.custom_openai_base_url_line_edit
        )

        self.translation_api_protocol_combo_box = QComboBox(self)
        self.translation_api_protocol_combo_box.setObjectName(
            "TranslationAPIProtocolComboBox"
        )
        self.translation_api_protocol_combo_box.addItem(
            _("Chat Completions (/chat/completions)"), CHAT_COMPLETIONS_PROTOCOL
        )
        self.translation_api_protocol_combo_box.addItem(
            _("Responses (/responses)"), RESPONSES_PROTOCOL
        )
        protocol = self.settings.value(
            Settings.Key.TRANSLATION_API_PROTOCOL,
            CHAT_COMPLETIONS_PROTOCOL,
        )
        protocol_index = self.translation_api_protocol_combo_box.findData(protocol)
        self.translation_api_protocol_combo_box.setCurrentIndex(
            protocol_index if protocol_index >= 0 else 0
        )
        self.translation_api_protocol_combo_box.currentIndexChanged.connect(
            self.on_translation_api_protocol_changed
        )
        openai_layout.addRow(
            _("OpenAI API protocol"), self.translation_api_protocol_combo_box
        )

        self.translation_batch_size_spin_box = QSpinBox(self)
        self.translation_batch_size_spin_box.setObjectName(
            "TranslationBatchSizeSpinBox"
        )
        self.translation_batch_size_spin_box.setRange(
            MIN_TRANSLATION_BATCH_SIZE, MAX_TRANSLATION_BATCH_SIZE
        )
        self.translation_batch_size_spin_box.setValue(
            self.settings.value(
                Settings.Key.TRANSLATION_BATCH_SIZE,
                DEFAULT_TRANSLATION_BATCH_SIZE,
            )
        )
        self.translation_batch_size_spin_box.setToolTip(
            _(
                "Number of segments sent in one translation request. Larger "
                "values mean fewer requests but a longer wait for the first "
                "result. Character and token limits still apply."
            )
        )
        self.translation_batch_size_spin_box.valueChanged.connect(
            self.on_translation_batch_size_changed
        )
        self.translation_batch_size_spin_box.setMaximumWidth(90)
        self.translation_batch_size_spin_box.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        openai_layout.addRow(
            _("Translation batch size"), self.translation_batch_size_spin_box
        )

        self.openai_api_model = self.settings.value(
            key=Settings.Key.OPENAI_API_MODEL, default_value="whisper-1"
        )

        self.openai_api_model_line_edit = LineEdit(self.openai_api_model, self)
        self.openai_api_model_line_edit.textChanged.connect(
            self.on_openai_api_model_changed
        )
        self.openai_api_model_line_edit.setMinimumWidth(200)
        self.openai_api_model_line_edit.setPlaceholderText("whisper-1")
        openai_layout.addRow(_("OpenAI API model"), self.openai_api_model_line_edit)

        default_export_file_name = self.settings.get_default_export_file_template()

        default_export_file_name_line_edit = LineEdit(default_export_file_name, self)
        default_export_file_name_line_edit.textChanged.connect(
            self.on_default_export_file_name_changed
        )
        default_export_file_name_line_edit.setMinimumWidth(200)
        export_layout.addRow(
            _("Default export file name"), default_export_file_name_line_edit
        )

        self.recording_export_enabled = self.settings.value(
            key=Settings.Key.RECORDING_TRANSCRIBER_EXPORT_ENABLED, default_value=False
        )

        self.export_enabled_checkbox = QCheckBox(_("Enable live recording transcription export"))
        self.export_enabled_checkbox.setChecked(self.recording_export_enabled)
        self.export_enabled_checkbox.setObjectName("EnableRecordingExportCheckbox")
        self.export_enabled_checkbox.stateChanged.connect(self.on_recording_export_enable_changed)
        # Spans both columns: the checkbox carries its own text, so an empty
        # label column just shifted it out of alignment with everything else.
        export_layout.addRow(self.export_enabled_checkbox)

        self.recording_export_folder_browse_button = QPushButton(_("Browse"))
        self.recording_export_folder_browse_button.clicked.connect(self.on_click_browse_export_folder)
        self.recording_export_folder_browse_button.setObjectName("RecordingExportFolderBrowseButton")

        recording_export_folder = self.settings.value(
            key=Settings.Key.RECORDING_TRANSCRIBER_EXPORT_FOLDER, default_value=user_documents_dir()
        )

        recording_export_folder_row = QHBoxLayout()
        self.recording_export_folder_line_edit = LineEdit(recording_export_folder, self)
        self.recording_export_folder_line_edit.textChanged.connect(self.on_recording_export_folder_changed)
        self.recording_export_folder_line_edit.setObjectName("RecordingExportFolderLineEdit")

        self.recording_export_folder_line_edit.setEnabled(self.recording_export_enabled)
        self.recording_export_folder_browse_button.setEnabled(self.recording_export_enabled)

        recording_export_folder_row.addWidget(self.recording_export_folder_line_edit)
        recording_export_folder_row.addWidget(self.recording_export_folder_browse_button)

        export_layout.addRow(_("Export folder"), recording_export_folder_row)

        self.recording_transcriber_mode = QComboBox(self)
        for mode in RecordingTranscriberMode:
            self.recording_transcriber_mode.addItem(mode.value)

        self.recording_transcriber_mode.setCurrentIndex(
            self.settings.value(Settings.Key.RECORDING_TRANSCRIBER_MODE, 0)
        )
        self.recording_transcriber_mode.currentIndexChanged.connect(self.on_recording_transcriber_mode_changed)

        export_layout.addRow(_("Live recording mode"), self.recording_transcriber_mode)

        export_note_label = QLabel(
            _("Note: Live recording export settings will be moved to the Advanced Settings in the Live Recording screen in a future version."),
            self,
        )
        export_note_label.setWordWrap(True)
        export_note_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # Spans both columns so the wrapped text is not squeezed into the narrow
        # field column, where it overflowed the dialog.
        export_layout.addRow(export_note_label)

        # Sites like bilibili only serve their full set of formats to a logged-in
        # client, so URL transcription needs a way to borrow the user's login.
        self.cookies_from_browser_combo_box = QComboBox(self)
        self.cookies_from_browser_combo_box.setObjectName("CookiesFromBrowserComboBox")
        self.cookies_from_browser_combo_box.addItem(_("None"), "")
        for browser in supported_browsers():
            self.cookies_from_browser_combo_box.addItem(browser, browser)
        saved_browser = self.settings.value(
            key=Settings.Key.DOWNLOAD_COOKIES_FROM_BROWSER, default_value=""
        )
        browser_index = self.cookies_from_browser_combo_box.findData(
            saved_browser or ""
        )
        self.cookies_from_browser_combo_box.setCurrentIndex(
            browser_index if browser_index >= 0 else 0
        )
        self.cookies_from_browser_combo_box.setToolTip(
            _("Use the login cookies of this browser when downloading from a URL. "
              "Needed for member-only or age-restricted videos.")
        )
        self.cookies_from_browser_combo_box.currentIndexChanged.connect(
            self.on_cookies_from_browser_changed
        )
        url_layout.addRow(_("Browser cookies"), self.cookies_from_browser_combo_box)

        self.cookiefile_browse_button = QPushButton(_("Browse"))
        self.cookiefile_browse_button.setObjectName("CookiefileBrowseButton")
        self.cookiefile_browse_button.clicked.connect(self.on_click_browse_cookiefile)

        cookiefile_row = QHBoxLayout()
        self.cookiefile_line_edit = LineEdit(
            self.settings.value(
                key=Settings.Key.DOWNLOAD_COOKIEFILE, default_value=""
            ),
            self,
        )
        self.cookiefile_line_edit.setObjectName("CookiefileLineEdit")
        self.cookiefile_line_edit.setMinimumWidth(200)
        self.cookiefile_line_edit.setPlaceholderText(_("Optional cookies.txt file"))
        self.cookiefile_line_edit.textChanged.connect(self.on_cookiefile_changed)
        cookiefile_row.addWidget(self.cookiefile_line_edit)
        cookiefile_row.addWidget(self.cookiefile_browse_button)

        url_layout.addRow(_("Cookie file"), cookiefile_row)

        self.reduce_gpu_memory_enabled = self.settings.value(
            key=Settings.Key.REDUCE_GPU_MEMORY, default_value=False
        )

        self.reduce_gpu_memory_checkbox = QCheckBox(_("Use 8-bit quantization to reduce memory usage"))
        self.reduce_gpu_memory_checkbox.setChecked(self.reduce_gpu_memory_enabled)
        self.reduce_gpu_memory_checkbox.setObjectName("ReduceGPUMemoryCheckbox")
        self.reduce_gpu_memory_checkbox.setToolTip(
            _("Applies to Huggingface and Faster Whisper models. "
              "Reduces GPU memory usage but may slightly decrease transcription quality.")
        )
        self.reduce_gpu_memory_checkbox.stateChanged.connect(self.on_reduce_gpu_memory_changed)
        transcription_layout.addRow(
            _("Reduce GPU RAM"), self.reduce_gpu_memory_checkbox
        )

        self.force_cpu_enabled = self.settings.value(
            key=Settings.Key.FORCE_CPU, default_value=False
        )

        self.force_cpu_checkbox = QCheckBox(_("Use only CPU and disable GPU acceleration"))
        self.force_cpu_checkbox.setChecked(self.force_cpu_enabled)
        self.force_cpu_checkbox.setObjectName("ForceCPUCheckbox")
        self.force_cpu_checkbox.setToolTip(_("Set this if larger models do not fit your GPU memory and Buzz crashes"))
        self.force_cpu_checkbox.stateChanged.connect(self.on_force_cpu_changed)
        transcription_layout.addRow(_("Disable GPU"), self.force_cpu_checkbox)

        content_layout.addStretch()

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)

    def on_cookies_from_browser_changed(self, _index: int):
        browser = self.cookies_from_browser_combo_box.currentData() or ""
        self.settings.set_value(Settings.Key.DOWNLOAD_COOKIES_FROM_BROWSER, browser)

    def on_cookiefile_changed(self, text: str):
        self.settings.set_value(Settings.Key.DOWNLOAD_COOKIEFILE, text.strip())

    def on_click_browse_cookiefile(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, _("Select Cookie File"), "", _("All Files (*)")
        )
        if path:
            # textChanged fires from this, which persists the value.
            self.cookiefile_line_edit.setText(path)

    def on_default_export_file_name_changed(self, text: str):
        self.settings.set_value(Settings.Key.DEFAULT_EXPORT_FILE_NAME, text)

    def update_test_openai_api_key_button(self):
        self.test_openai_api_key_button.setEnabled(len(self.openai_api_key) > 0)

    def on_click_test_openai_api_key_button(self):
        self.test_openai_api_key_button.setEnabled(False)

        job = ValidateOpenAIApiKeyJob(api_key=self.openai_api_key)
        job.signals.success.connect(self.on_test_openai_api_key_success)
        job.signals.failed.connect(self.on_test_openai_api_key_failure)
        job.setAutoDelete(True)

        thread_pool = QThreadPool.globalInstance()
        thread_pool.start(job)

    def on_test_openai_api_key_success(self):
        self.test_openai_api_key_button.setEnabled(True)
        QMessageBox.information(
            self,
            _("OpenAI API Key Test"),
            _("Your API key is valid. Buzz will use this key to perform Whisper API transcriptions and AI translations."),
        )

    def on_test_openai_api_key_failure(self, error: str):
        self.test_openai_api_key_button.setEnabled(True)
        QMessageBox.warning(self, _("OpenAI API Key Test"), error)

    def on_openai_api_key_changed(self, key: str):
        self.openai_api_key = key
        self.update_test_openai_api_key_button()
        self.openai_api_key_changed.emit(key)

    def on_openai_api_key_focus_out(self):
        if not BASE64_PATTERN.match(self.openai_api_key):
            QMessageBox.warning(
                self,
                _("Invalid API key"),
                _("API supports only base64 characters (A-Za-z0-9+/=_-). Other characters in API key may cause errors."),
            )

    def on_custom_openai_base_url_changed(self, text: str):
        self.settings.set_value(Settings.Key.CUSTOM_OPENAI_BASE_URL, text)

    def on_openai_api_model_changed(self, text: str):
        self.settings.set_value(Settings.Key.OPENAI_API_MODEL, text)

    def on_translation_api_protocol_changed(self, index: int):
        protocol = self.translation_api_protocol_combo_box.itemData(index)
        if protocol in {CHAT_COMPLETIONS_PROTOCOL, RESPONSES_PROTOCOL}:
            self.settings.set_value(Settings.Key.TRANSLATION_API_PROTOCOL, protocol)

    def on_translation_batch_size_changed(self, value: int):
        self.settings.set_value(Settings.Key.TRANSLATION_BATCH_SIZE, value)

    def on_recording_export_enable_changed(self, state: int):
        self.recording_export_enabled = state == 2

        self.recording_export_folder_line_edit.setEnabled(self.recording_export_enabled)
        self.recording_export_folder_browse_button.setEnabled(self.recording_export_enabled)

        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_EXPORT_ENABLED,
            self.recording_export_enabled,
        )

    def on_click_browse_export_folder(self):
        folder = QFileDialog.getExistingDirectory(self, _("Select Export Folder"))
        self.recording_export_folder_line_edit.setText(folder)
        self.on_recording_export_folder_changed(folder)

    def on_recording_export_folder_changed(self, folder):
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_EXPORT_FOLDER,
            folder,
        )

    def on_language_changed(self, index):
        selected_language = self.ui_language_combo_box.itemText(index)
        locale_code = next((code for code, lang in ui_locales.items() if lang == selected_language), "en_US")

        self.load_note_tooltip_icon.setVisible(True)

        self.settings.set_value(Settings.Key.UI_LOCALE, locale_code)

    def on_font_size_changed(self, value):
        from buzz.widgets.application import Application
        font = self.font()
        font.setPointSize(value)
        self.setFont(font)
        Application.instance().setFont(font)

        self.settings.set_value(Settings.Key.FONT_SIZE, value)

    def on_transcription_concurrency_changed(self, value):
        self.settings.set_value(Settings.Key.TRANSCRIPTION_CONCURRENCY, value)

    def on_recording_transcriber_mode_changed(self, value):
        self.settings.set_value(Settings.Key.RECORDING_TRANSCRIBER_MODE, value)

    def on_force_cpu_changed(self, state: int):
        import os
        self.force_cpu_enabled = state == 2
        self.settings.set_value(Settings.Key.FORCE_CPU, self.force_cpu_enabled)

        if self.force_cpu_enabled:
            os.environ["BUZZ_FORCE_CPU"] = "true"
        else:
            os.environ.pop("BUZZ_FORCE_CPU", None)

    def on_reduce_gpu_memory_changed(self, state: int):
        import os
        self.reduce_gpu_memory_enabled = state == 2
        self.settings.set_value(Settings.Key.REDUCE_GPU_MEMORY, self.reduce_gpu_memory_enabled)

        if self.reduce_gpu_memory_enabled:
            os.environ["BUZZ_REDUCE_GPU_MEMORY"] = "true"
        else:
            os.environ.pop("BUZZ_REDUCE_GPU_MEMORY", None)


class ValidateOpenAIApiKeyJob(QRunnable):
    class Signals(QObject):
        success = pyqtSignal()
        failed = pyqtSignal(str)

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.signals = self.Signals()

    def run(self):
        settings = Settings()
        configured_base_url = os.getenv(
            "BUZZ_TRANSLATION_API_BASE_URL",
            os.getenv(
                "BUZZ_TRANSLATION_API_BASE_URl",
                settings.value(
                    key=Settings.Key.CUSTOM_OPENAI_BASE_URL, default_value=""
                ),
            ),
        )
        base_url = configured_base_url or DEFAULT_OPENAI_BASE_URL
        model = os.getenv(
            "BUZZ_TRANSLATION_API_MODEL",
            settings.value(key=Settings.Key.OPENAI_API_MODEL, default_value=""),
        )
        protocol = _translation_api_protocol(
            settings.value(
                Settings.Key.TRANSLATION_API_PROTOCOL,
                CHAT_COMPLETIONS_PROTOCOL,
            )
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if protocol == RESPONSES_PROTOCOL:
            url = _responses_url(base_url)
            body = {
                "model": model,
                "max_output_tokens": 8,
                "input": "hi",
            }
        else:
            url = _chat_completions_url(base_url)
            body = {
                "model": model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            }

        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=20)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return

        if resp.status_code == 200:
            self.signals.success.emit()
            return

        # Surface the API's own error message where possible.
        message = str(resp.status_code)
        try:
            data = resp.json()
            err = data.get("error", {})
            if isinstance(err, dict) and err.get("message"):
                message = err["message"]
        except Exception:
            pass
        self.signals.failed.emit(
            _("{} API key test failed ({}). Check the API key, base URL and model name.").format(
                "OpenAI", message
            )
        )

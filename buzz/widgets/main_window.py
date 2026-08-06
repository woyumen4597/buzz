import os
import logging
import time
from typing import Dict, Tuple, List, Optional
from uuid import UUID

from PyQt6 import QtGui
from PyQt6.QtCore import (
    Qt,
    QThread,
    QThreadPool,
    QModelIndex,
    pyqtSignal
)

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QFileDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from buzz.db.entity.transcription import Transcription
from buzz.db.service.transcription_service import TranscriptionService
from buzz.file_transcriber_queue_worker import FileTranscriberQueueWorker
from buzz.locale import _
from buzz.plugins.manager import PluginManager
from buzz.plugins.post_processing import FnRunnable
from buzz.settings.settings import (
    APP_NAME,
    DEFAULT_TRANSCRIPTION_CONCURRENCY,
    MAX_TRANSCRIPTION_CONCURRENCY,
    MIN_TRANSCRIPTION_CONCURRENCY,
    Settings,
)
from buzz.update_checker import UpdateChecker, UpdateInfo
from buzz.widgets.update_dialog import UpdateDialog
from buzz.settings.shortcuts import Shortcuts
from buzz.store.keyring_store import get_password, set_password, Key
from buzz.transcriber.transcriber import (
    FileTranscriptionTask,
    TranscriptionOptions,
    FileTranscriptionOptions,
    SUPPORTED_AUDIO_FORMATS,
    Segment,
    deserialize_segment_checkpoint,
    deserialize_task_metadata,
    deserialize_task_options,
    source_file_matches_fingerprint,
)
from buzz.widgets.icon import BUZZ_ICON_PATH
from buzz.widgets.import_url_dialog import ImportURLDialog
from buzz.widgets.main_window_toolbar import MainWindowToolbar
from buzz.widgets.menu_bar import MenuBar
from buzz.widgets.preferences_dialog.models.preferences import Preferences
from buzz.widgets.transcriber.file_transcriber_widget import FileTranscriberWidget
from buzz.widgets.transcription_task_folder_watcher import (
    TranscriptionTaskFolderWatcher,
    SUPPORTED_EXTENSIONS,
)
from buzz.widgets.transcription_tasks_table_widget import (
    TranscriptionTasksTableWidget,
)
from buzz.widgets.transcription_viewer.transcription_viewer_widget import (
    TranscriptionViewerWidget,
)

# Progress events are coalesced before hitting the database and the task table:
# write at most every 250ms or when the value moved by 1%, whichever comes
# first, so a long transcription does not hammer SQLite with per-segment I/O.
PROGRESS_WRITE_MIN_INTERVAL_S = 0.25
PROGRESS_WRITE_MIN_DELTA = 0.01


class MainWindow(QMainWindow):
    table_widget: TranscriptionTasksTableWidget
    transcriptions_updated = pyqtSignal(UUID)

    def __init__(self, transcription_service: TranscriptionService):
        super().__init__(flags=Qt.WindowType.Window)

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(BUZZ_ICON_PATH))

        self.setAcceptDrops(True)

        # Per-task (last written progress, monotonic time of that write) for
        # coalescing progress events; see PROGRESS_WRITE_MIN_*.
        self._progress_state: Dict[UUID, Tuple[float, float]] = {}
        self._download_progress_state: Dict[UUID, Tuple[float, float]] = {}
        # Tasks that reached a terminal state; late progress events are ignored.
        self._finished_tasks = set()

        self.settings = Settings()

        self.shortcuts = Shortcuts(settings=self.settings)

        self.quit_on_complete = False
        self.transcription_service = transcription_service

        self.plugin_manager = PluginManager(self.transcription_service, self.settings)
        try:
            self.plugin_manager.initialize()
        except Exception as exc:
            logging.error(f"Failed to initialize plugins: {exc}", exc_info=True)

        #update checker
        self._update_info: Optional[UpdateInfo] = None

        self.toolbar = MainWindowToolbar(shortcuts=self.shortcuts, parent=self)
        self.toolbar.setObjectName("MainToolbar")
        self.toolbar.new_transcription_action_triggered.connect(
            self.on_new_transcription_action_triggered
        )
        self.toolbar.new_url_transcription_action_triggered.connect(
            self.on_new_url_transcription_action_triggered
        )
        self.toolbar.open_transcript_action_triggered.connect(
            self.open_transcript_viewer
        )
        self.toolbar.clear_history_action_triggered.connect(
            self.on_clear_history_action_triggered
        )
        self.toolbar.stop_transcription_action_triggered.connect(
            self.on_stop_transcription_action_triggered
        )
        self.addToolBar(self.toolbar)
        self.toolbar.update_action_triggered.connect(self.on_update_action_triggered)
        self.setUnifiedTitleAndToolBarOnMac(True)

        self.preferences = self.load_preferences(settings=self.settings)
        self.menu_bar = MenuBar(
            shortcuts=self.shortcuts,
            preferences=self.preferences,
            plugin_manager=self.plugin_manager,
            parent=self,
        )
        self.menu_bar.import_action_triggered.connect(
            self.on_new_transcription_action_triggered
        )
        self.menu_bar.import_url_action_triggered.connect(
            self.on_new_url_transcription_action_triggered
        )
        self.menu_bar.import_folder_action_triggered.connect(
            self.on_import_folder_action_triggered
        )
        self.menu_bar.shortcuts_changed.connect(self.on_shortcuts_changed)
        self.menu_bar.openai_api_key_changed.connect(
            self.on_openai_access_token_changed
        )
        self.menu_bar.preferences_changed.connect(self.on_preferences_changed)
        self.setMenuBar(self.menu_bar)

        self.table_widget = TranscriptionTasksTableWidget(self)
        self.table_widget.transcription_service = self.transcription_service
        self.table_widget.doubleClicked.connect(self.on_table_double_clicked)
        self.table_widget.return_clicked.connect(self.open_transcript_viewer)
        self.table_widget.delete_requested.connect(self.on_clear_history_action_triggered)
        self.table_widget.selectionModel().selectionChanged.connect(
            self.on_table_selection_changed
        )
        self.transcriptions_updated.connect(
            self.on_transcriptions_updated
        )

        self._setup_task_library()

        transcriber_count = int(
            self.settings.value(
                Settings.Key.TRANSCRIPTION_CONCURRENCY,
                DEFAULT_TRANSCRIPTION_CONCURRENCY,
            )
        )
        # ponytail: cap at 8; each worker increases model memory usage.
        transcriber_count = max(
            MIN_TRANSCRIPTION_CONCURRENCY,
            min(MAX_TRANSCRIPTION_CONCURRENCY, transcriber_count),
        )
        self.transcriber_threads = [QThread() for _ in range(transcriber_count)]
        self.transcriber_workers = [
            FileTranscriberQueueWorker() for _ in self.transcriber_threads
        ]
        self._next_transcriber_worker = 0

        # Keep the first worker/thread names for compatibility.
        self.transcriber_thread = self.transcriber_threads[0]
        self.transcriber_worker = self.transcriber_workers[0]

        for worker, thread in zip(self.transcriber_workers, self.transcriber_threads):
            worker.plugin_manager = self.plugin_manager
            worker.moveToThread(thread)

            worker.task_started.connect(self.on_task_started)
            worker.task_progress.connect(self.on_task_progress)
            worker.task_download_progress.connect(self.on_task_download_progress)
            worker.task_checkpoint.connect(self.on_task_checkpoint)
            worker.task_error.connect(self.on_task_error)
            worker.task_completed.connect(self.on_task_completed)

            worker.completed.connect(thread.quit)
            thread.started.connect(worker.run)
            thread.start()

        self.restore_unfinished_tasks()

        self.load_geometry()

        self.folder_watcher = TranscriptionTaskFolderWatcher(
            tasks={},
            preferences=self.preferences.folder_watch,
        )
        self.folder_watcher.task_found.connect(self.add_task)
        self.folder_watcher.find_tasks()

        self.transcription_viewer_widget = None

        #Initialize and run update checker
        self._init_update_checker()

    def on_preferences_changed(self, preferences: Preferences):
        self.preferences = preferences
        self.save_preferences(preferences)
        self.folder_watcher.set_preferences(preferences.folder_watch)
        self.folder_watcher.find_tasks()

    def save_preferences(self, preferences: Preferences):
        self.settings.settings.beginGroup("preferences")
        preferences.save(self.settings.settings)
        self.settings.settings.endGroup()

    def load_preferences(self, settings: Settings):
        settings.settings.beginGroup("preferences")
        preferences = Preferences.load(settings.settings)
        settings.settings.endGroup()
        return preferences

    def _setup_task_library(self):
        self.search_input = QLineEdit()
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setPlaceholderText(
            _("Search file name, URL, model, or notes")
        )
        self.search_input.textChanged.connect(self._apply_task_filter)

        self.status_filter = QComboBox()
        self.status_filter.setMinimumWidth(126)
        self.status_filter.addItem(_("All statuses"), "")
        status_labels = {
            FileTranscriptionTask.Status.QUEUED: _("Queued"),
            FileTranscriptionTask.Status.IN_PROGRESS: _("In Progress"),
            FileTranscriptionTask.Status.COMPLETED: _("Completed"),
            FileTranscriptionTask.Status.FAILED: _("Failed"),
            FileTranscriptionTask.Status.CANCELED: _("Canceled"),
            FileTranscriptionTask.Status.SKIPPED: _("Skipped"),
        }
        for status in (
            FileTranscriptionTask.Status.QUEUED,
            FileTranscriptionTask.Status.IN_PROGRESS,
            FileTranscriptionTask.Status.COMPLETED,
            FileTranscriptionTask.Status.FAILED,
            FileTranscriptionTask.Status.CANCELED,
            FileTranscriptionTask.Status.SKIPPED,
        ):
            self.status_filter.addItem(status_labels[status], status.value)
        self.status_filter.currentIndexChanged.connect(self._apply_task_filter)

        self.clear_filters_button = QPushButton(_("Clear filters"))
        self.clear_filters_button.setObjectName("ClearFilters")
        self.clear_filters_button.setMinimumWidth(104)
        self.clear_filters_button.setEnabled(False)
        self.clear_filters_button.clicked.connect(self._clear_task_filters)

        self.task_count_label = QLabel()
        self.task_count_label.setObjectName("TaskCount")

        central_widget = QWidget(self)
        central_widget.setObjectName("TaskLibrary")
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(28, 24, 28, 28)
        central_layout.setSpacing(16)

        heading_layout = QHBoxLayout()
        heading_layout.setSpacing(16)
        heading_text = QVBoxLayout()
        heading_text.setSpacing(3)

        eyebrow = QLabel("BUZZ / TRANSCRIPTION WORKSPACE")
        eyebrow.setObjectName("Eyebrow")
        heading_text.addWidget(eyebrow)

        title = QLabel(_("Transcription workspace"))
        title.setObjectName("PageTitle")
        heading_text.addWidget(title)

        subtitle = QLabel(_("Search, sort, and manage your transcriptions."))
        subtitle.setObjectName("PageSubtitle")
        heading_text.addWidget(subtitle)

        heading_layout.addLayout(heading_text)
        heading_layout.addStretch()
        heading_layout.addWidget(self.task_count_label, 0, Qt.AlignmentFlag.AlignTop)
        central_layout.addLayout(heading_layout)

        filter_bar = QFrame()
        filter_bar.setObjectName("FilterBar")
        filter_layout = QHBoxLayout(filter_bar)
        filter_layout.setContentsMargins(14, 10, 14, 10)
        filter_layout.setSpacing(10)

        filter_label = QLabel(_("Filter"))
        filter_label.setObjectName("FilterLabel")
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.search_input, 1)

        status_label = QLabel(_("Status"))
        status_label.setObjectName("FilterLabel")
        filter_layout.addWidget(status_label)
        filter_layout.addWidget(self.status_filter)
        filter_layout.addWidget(self.clear_filters_button)
        central_layout.addWidget(filter_bar)
        central_layout.addWidget(self.table_widget, 1)

        self.setCentralWidget(central_widget)
        self.table_widget.model().modelReset.connect(self._update_task_count)
        self.table_widget.model().rowsInserted.connect(self._update_task_count)
        self.table_widget.model().rowsRemoved.connect(self._update_task_count)
        self._update_task_count()
        self._apply_visual_style()

    def _apply_task_filter(self, *_args):
        self.table_widget.set_filter(
            self.search_input.text(), self.status_filter.currentData() or ""
        )
        self.clear_filters_button.setEnabled(
            bool(self.search_input.text().strip())
            or bool(self.status_filter.currentData())
        )
        self._update_task_count()

    def _clear_task_filters(self):
        self.search_input.clear()
        self.status_filter.setCurrentIndex(0)

    def _update_task_count(self, *_args):
        self.task_count_label.setText(
            _("{} tasks").format(self.table_widget.model().rowCount())
        )

    def _apply_visual_style(self):
        dark = self.palette().window().color().lightness() < 128
        colors = (
            {
                "background": "#10181D",
                "surface": "#172329",
                "surface_alt": "#1D2C33",
                "border": "#2D414A",
                "text": "#E7F3F0",
                "muted": "#A0B4B8",
                "accent": "#35C8AE",
                "accent_soft": "#24483F",
                "toolbar": "#0C1216",
                "toolbar_text": "#E7F3F0",
            }
            if dark
            else {
                "background": "#F3F6F7",
                "surface": "#FFFFFF",
                "surface_alt": "#F8FBFA",
                "border": "#DCE7E7",
                "text": "#153039",
                "muted": "#6D7E84",
                "accent": "#0E9F8A",
                "accent_soft": "#DFF5EF",
                "toolbar": "#182B33",
                "toolbar_text": "#E8F3F0",
            }
        )
        self.setStyleSheet(
            f"""
            QMainWindow, #TaskLibrary {{ background: {colors['background']}; }}
            QToolBar#MainToolbar {{
                background: {colors['toolbar']};
                border: 0;
                border-bottom: 1px solid {colors['border']};
                padding: 8px 14px;
                spacing: 4px;
            }}
            QToolBar#MainToolbar QToolButton {{
                color: {colors['toolbar_text']};
                background: transparent;
                border: 0;
                border-radius: 8px;
                padding: 8px 10px;
                margin: 0 2px;
            }}
            QToolBar#MainToolbar QToolButton:hover {{
                background: {colors['accent_soft']};
            }}
            QToolBar#MainToolbar QToolButton:disabled {{
                color: {colors['muted']};
            }}
            QLabel#Eyebrow {{
                color: {colors['accent']};
                font-size: 11px;
                font-weight: 700;
            }}
            QLabel#PageTitle {{
                color: {colors['text']};
                font-size: 24px;
                font-weight: 700;
            }}
            QLabel#PageSubtitle {{ color: {colors['muted']}; font-size: 13px; }}
            QLabel#TaskCount {{
                color: {colors['accent']};
                background: {colors['accent_soft']};
                border-radius: 12px;
                padding: 7px 11px;
                font-weight: 700;
            }}
            QFrame#FilterBar {{
                background: {colors['surface']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
            }}
            QLabel#FilterLabel {{
                color: {colors['muted']};
                font-size: 12px;
                font-weight: 700;
            }}
            QLineEdit, QComboBox {{
                color: {colors['text']};
                background: {colors['surface_alt']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 8px 10px;
                min-height: 18px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 1px solid {colors['accent']};
            }}
            QPushButton#ClearFilters {{
                color: {colors['accent']};
                background: transparent;
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 8px 12px;
            }}
            QPushButton#ClearFilters:hover {{
                background: {colors['accent_soft']};
                border-color: {colors['accent']};
            }}
            QTableView#TaskTable {{
                color: {colors['text']};
                background: {colors['surface']};
                alternate-background-color: {colors['surface_alt']};
                border: 1px solid {colors['border']};
                border-radius: 14px;
                gridline-color: transparent;
                outline: 0;
                selection-background-color: {colors['accent_soft']};
                selection-color: {colors['text']};
            }}
            QTableView#TaskTable::item {{
                padding: 8px 10px;
                border-bottom: 1px solid {colors['border']};
            }}
            QHeaderView::section {{
                color: {colors['muted']};
                background: {colors['surface_alt']};
                border: 0;
                border-bottom: 1px solid {colors['border']};
                padding: 11px 10px;
                font-size: 12px;
                font-weight: 700;
            }}
            """
        )

    def dragEnterEvent(self, event):
        # Accept file drag events
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.open_file_transcriber_widget(file_paths=file_paths)

    def on_file_transcriber_triggered(
        self, options: Tuple[TranscriptionOptions, FileTranscriptionOptions, str]
    ):
        transcription_options, file_transcription_options, model_path = options

        if file_transcription_options.file_paths is not None:
            for file_path in file_transcription_options.file_paths:
                task = FileTranscriptionTask(
                    transcription_options=transcription_options,
                    file_transcription_options=file_transcription_options,
                    model_path=model_path,
                    file_path=file_path,
                    source=FileTranscriptionTask.Source.FILE_IMPORT,
                )
                self.add_task(task)
        else:
            task = FileTranscriptionTask(
                transcription_options=transcription_options,
                file_transcription_options=file_transcription_options,
                model_path=model_path,
                url=file_transcription_options.url,
                source=FileTranscriptionTask.Source.URL_IMPORT,
            )
            self.add_task(task)

    def on_clear_history_action_triggered(self):
        selected_rows = self.table_widget.selectionModel().selectedRows()
        if len(selected_rows) == 0:
            return

        question_box = QMessageBox()
        question_box.setWindowTitle(_("Clear History"))
        question_box.setIcon(QMessageBox.Icon.Question)
        question_box.setText(
            _(
                "Are you sure you want to delete the selected transcription(s)? "
                "This action cannot be undone."
            ),
        )
        question_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        question_box.button(QMessageBox.StandardButton.Yes).setText(_("Ok"))
        question_box.button(QMessageBox.StandardButton.No).setText(_("Cancel"))

        reply = question_box.exec()

        if reply == QMessageBox.StandardButton.Yes:
            self.table_widget.delete_transcriptions(selected_rows)

    def on_stop_transcription_action_triggered(self):
        selected_transcriptions = self.table_widget.selected_transcriptions()
        for transcription in selected_transcriptions:
            transcription_id = transcription.id_as_uuid
            for worker in self.transcriber_workers:
                worker.cancel_task(transcription_id)
            self.transcription_service.update_transcription_as_canceled(
                transcription_id
            )
            self.table_widget.refresh_row(transcription_id)
            self.on_table_selection_changed()

    def on_new_transcription_action_triggered(self):
        last_folder = self.settings.value(Settings.Key.LAST_IMPORT_FOLDER, "")

        (file_paths, __) = QFileDialog.getOpenFileNames(
            self, _("Select audio file"), last_folder, SUPPORTED_AUDIO_FORMATS
        )
        if len(file_paths) == 0:
            return

        self.settings.set_value(
            Settings.Key.LAST_IMPORT_FOLDER, os.path.dirname(file_paths[0])
        )

        self.open_file_transcriber_widget(file_paths)

    def on_new_url_transcription_action_triggered(self):
        url = ImportURLDialog.prompt(parent=self)
        if url is not None:
            self.open_file_transcriber_widget(url=url)

    def on_import_folder_action_triggered(self):
        last_folder = self.settings.value(Settings.Key.LAST_IMPORT_FOLDER, "")
        folder = QFileDialog.getExistingDirectory(
            self, _("Select folder"), last_folder
        )
        if not folder:
            return
        self.settings.set_value(Settings.Key.LAST_IMPORT_FOLDER, folder)
        file_paths = []
        for dirpath, _dirs, filenames in os.walk(folder):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    file_paths.append(os.path.join(dirpath, filename))
        if not file_paths:
            return
        self.open_file_transcriber_widget(file_paths)

    def open_file_transcriber_widget(
        self, file_paths: Optional[List[str]] = None, url: Optional[str] = None
    ):
        file_transcriber_window = FileTranscriberWidget(
            file_paths=file_paths,
            url=url,
            parent=self,
            flags=Qt.WindowType.Window,
        )
        file_transcriber_window.triggered.connect(self.on_file_transcriber_triggered)
        file_transcriber_window.openai_access_token_changed.connect(
            self.on_openai_access_token_changed
        )
        file_transcriber_window.show()
        file_transcriber_window.raise_()
        file_transcriber_window.activateWindow()

    @staticmethod
    def on_openai_access_token_changed(access_token: str):
        try:
            set_password(Key.OPENAI_API_KEY, access_token)
        except Exception as exc:
            logging.error("Unable to write to keyring: %s", exc)
            QMessageBox.critical(
                None, _("Error"), _("Unable to save OpenAI API key to keyring")
            )

    def open_transcript_viewer(self):
        selected_rows = self.table_widget.selectionModel().selectedRows()
        for selected_row in selected_rows:
            transcription = self.table_widget.transcription(selected_row)
            self.open_transcription_viewer(transcription)

    def on_table_selection_changed(self):
        self.toolbar.set_open_transcript_action_enabled(
            self.should_enable_open_transcript_action()
        )
        self.toolbar.set_stop_transcription_action_enabled(
            self.should_enable_stop_transcription_action()
        )
        self.toolbar.set_clear_history_action_enabled(
            self.should_enable_clear_history_action()
        )

    def should_enable_open_transcript_action(self):
        selected_transcriptions = self.table_widget.selected_transcriptions()
        if len(selected_transcriptions) == 0:
            return False
        return all(
            MainWindow.can_open_transcript(transcription)
            for transcription in selected_transcriptions
        )

    @staticmethod
    def can_open_transcript(transcription: Transcription) -> bool:
        return FileTranscriptionTask.Status(transcription.status) in (
            FileTranscriptionTask.Status.COMPLETED,
            FileTranscriptionTask.Status.SKIPPED,
        )

    def should_enable_stop_transcription_action(self):
        return self.selected_tasks_have_status(
            [
                FileTranscriptionTask.Status.IN_PROGRESS,
                FileTranscriptionTask.Status.QUEUED,
            ]
        )

    def should_enable_clear_history_action(self):
        return self.selected_tasks_have_status(
            [
                FileTranscriptionTask.Status.COMPLETED,
                FileTranscriptionTask.Status.FAILED,
                FileTranscriptionTask.Status.CANCELED,
                FileTranscriptionTask.Status.SKIPPED,
            ]
        )

    def selected_tasks_have_status(self, statuses: List[FileTranscriptionTask.Status]):
        transcriptions = self.table_widget.selected_transcriptions()
        if len(transcriptions) == 0:
            return False

        return all(
            [
                transcription.status_as_status in statuses
                for transcription in transcriptions
            ]
        )

    def on_table_double_clicked(self, index: QModelIndex):
        transcription = self.table_widget.transcription(index)
        if not MainWindow.can_open_transcript(transcription):
            return
        self.open_transcription_viewer(transcription)

    def open_transcription_viewer(self, transcription: Transcription):
        self.transcription_viewer_widget = TranscriptionViewerWidget(
            transcription=transcription,
            transcription_service=self.transcription_service,
            shortcuts=self.shortcuts,
            parent=self,
            flags=Qt.WindowType.Window,
            transcriptions_updated_signal=self.transcriptions_updated,
        )
        self.transcription_viewer_widget.show()

    def add_task(self, task: FileTranscriptionTask):
        self.transcription_service.create_transcription(task)
        self.table_widget.refresh_all()
        worker = self._pick_transcriber_worker()
        worker.add_task(task)

    def restore_unfinished_tasks(self):
        """Put interrupted work back on the queue after validating its input."""
        for transcription in self.transcription_service.get_unfinished_transcriptions():
            try:
                task = self._restore_task(transcription)
                self.transcription_service.queue_transcription_for_recovery(
                    transcription.id_as_uuid
                )
                self._pick_transcriber_worker().add_task(task)
            except Exception as exc:
                message = _("Could not recover transcription: {}").format(exc)
                logging.warning("%s (%s)", message, transcription.id)
                self.transcription_service.update_transcription_as_failed(
                    transcription.id_as_uuid, message
                )
        self.table_widget.refresh_all()

    def _restore_task(self, transcription: Transcription) -> FileTranscriptionTask:
        task_metadata = deserialize_task_metadata(transcription.task_options_json)
        source_value = transcription.source or task_metadata.get("source")
        source_file = transcription.file or task_metadata.get("file_path")
        if source_value != FileTranscriptionTask.Source.URL_IMPORT.value:
            if not source_file or not os.path.isfile(source_file):
                raise FileNotFoundError(_("Source file is missing"))
            if (
                transcription.source_file_fingerprint
                and not source_file_matches_fingerprint(
                    source_file, transcription.source_file_fingerprint
                )
            ):
                raise ValueError(_("Source file has changed"))

        transcription_options, file_transcription_options = deserialize_task_options(
            transcription.task_options_json,
            fallback=transcription,
            openai_access_token=get_password(Key.OPENAI_API_KEY),
        )
        model_path = transcription_options.model.get_local_model_path()
        if model_path is None:
            from buzz.model_loader import ModelDownloader

            ModelDownloader(model=transcription_options.model).run()
            model_path = transcription_options.model.get_local_model_path()
        if model_path is None:
            raise RuntimeError(_("Model is not available"))

        checkpoint_segments, checkpoint_next_chunk = deserialize_segment_checkpoint(
            transcription.segment_checkpoint_json
        )
        try:
            source = FileTranscriptionTask.Source(source_value)
        except (TypeError, ValueError):
            source = FileTranscriptionTask.Source.FILE_IMPORT
        return FileTranscriptionTask(
            transcription_options=transcription_options,
            file_transcription_options=file_transcription_options,
            model_path=model_path,
            uid=transcription.id_as_uuid,
            source=source,
            file_path=source_file,
            original_file_path=task_metadata.get("original_file_path"),
            delete_source_file=bool(task_metadata.get("delete_source_file")),
            url=transcription.url or task_metadata.get("url") or file_transcription_options.url,
            output_directory=(
                transcription.output_folder or task_metadata.get("output_directory")
            ),
            segments=checkpoint_segments,
            fraction_completed=transcription.progress or 0.0,
            fraction_downloaded=transcription.download_progress or 0.0,
            checkpoint_next_chunk=checkpoint_next_chunk,
        )

    def _pick_transcriber_worker(self):
        """Prefer an idle worker; otherwise the least-loaded queue.

        Round-robin scanning among idle workers keeps batch dispatch fair, and
        falls back to queue size so a long video occupying one worker doesn't
        stall short tasks that could run on an idle sibling.
        """
        workers = self.transcriber_workers
        n = len(workers)
        for offset in range(n):
            idx = (self._next_transcriber_worker + offset) % n
            if not workers[idx].is_running:
                self._next_transcriber_worker = (idx + 1) % n
                return workers[idx]
        # All workers busy: dispatch to the least-loaded queue.
        return min(workers, key=lambda worker: worker.tasks_queue.qsize())

    def on_transcriptions_updated(self):
        self.table_widget.refresh_all()

    def on_task_started(self, task: FileTranscriptionTask):
        self._finished_tasks.discard(task.uid)
        self._progress_state.pop(task.uid, None)
        self._download_progress_state.pop(task.uid, None)
        self.transcription_service.update_transcription_as_started(task.uid)
        self.table_widget.refresh_row(task.uid)

    def on_task_progress(self, task: FileTranscriptionTask, progress: float):
        if task.uid in self._finished_tasks:
            return
        pct = max(0.0, min(1.0, progress))
        now = time.monotonic()
        last_pct, last_write = self._progress_state.get(task.uid, (-1.0, 0.0))
        if pct - last_pct < PROGRESS_WRITE_MIN_DELTA and now - last_write < PROGRESS_WRITE_MIN_INTERVAL_S:
            return
        self._progress_state[task.uid] = (pct, now)
        self.transcription_service.update_transcription_progress(task.uid, pct)
        self.table_widget.refresh_row(task.uid)

    def on_task_download_progress(
        self, task: FileTranscriptionTask, fraction_downloaded: float
    ):
        if task.uid in self._finished_tasks:
            return
        pct = max(0.0, min(1.0, fraction_downloaded))
        now = time.monotonic()
        last_pct, last_write = self._download_progress_state.get(
            task.uid, (-1.0, 0.0)
        )
        if (
            pct - last_pct < PROGRESS_WRITE_MIN_DELTA
            and now - last_write < PROGRESS_WRITE_MIN_INTERVAL_S
        ):
            return
        task.fraction_downloaded = pct
        self._download_progress_state[task.uid] = (pct, now)
        self.transcription_service.update_transcription_download_progress(
            task.uid, pct
        )
        self.table_widget.refresh_row(task.uid)

    def on_task_checkpoint(
        self, task: FileTranscriptionTask, segments: List[Segment]
    ):
        if task.uid in self._finished_tasks:
            return
        task.segments = segments
        self.transcription_service.update_transcription_segment_checkpoint(
            task.uid, task
        )

    def on_task_completed(self, task: FileTranscriptionTask, segments: List[Segment]):
        # Force the final 100% into the database: the completed marker does
        # not carry a progress value, and the last coalesced write may be lower.
        self._finished_tasks.add(task.uid)
        self._progress_state.pop(task.uid, None)
        self._download_progress_state.pop(task.uid, None)
        self.transcription_service.update_transcription_progress(task.uid, 1.0)
        if task.source == FileTranscriptionTask.Source.URL_IMPORT:
            self.transcription_service.update_transcription_download_progress(
                task.uid, 1.0
            )

        # Handle skipped tasks (e.g. plugin detected file already transcribed)
        if task.status == FileTranscriptionTask.Status.SKIPPED:
            self.transcription_service.update_transcription_as_skipped(task.uid, segments)
            self.table_widget.refresh_row(task.uid)
            if self.quit_on_complete:
                self.close()
                QApplication.quit()
            return

        # Update file path in database only for URL imports where file is downloaded
        if task.source == FileTranscriptionTask.Source.URL_IMPORT and task.file_path:
            logging.debug(f"Updating transcription file path: {task.file_path}")
            # Use the file basename (video title) as the display name
            basename = os.path.basename(task.file_path)
            name = os.path.splitext(basename)[0]  # Remove .wav extension
            self.transcription_service.update_transcription_file_and_name(task.uid, task.file_path, name)

        # When plugins are enabled, run the after_transcription / save / on_complete
        # pipeline on a background thread so slow plugin work (e.g. network calls)
        # doesn't freeze the UI. DB writes are marshaled back to the main thread by
        # the plugin manager. When quitting on complete we run synchronously so the
        # work isn't cut short.
        run_async = (
            self.plugin_manager.has_enabled_post_hooks() and not self.quit_on_complete
        )
        if run_async:
            runnable = FnRunnable(
                lambda: self.plugin_manager.process_completed(task, segments)
            )
            runnable.signals.finished.connect(
                lambda: self.table_widget.refresh_row(task.uid)
            )
            runnable.signals.error.connect(
                lambda e: logging.error(f"Plugin post-processing failed: {e}")
            )
            QThreadPool.globalInstance().start(runnable)
        elif self.plugin_manager.has_enabled_post_hooks():
            self.plugin_manager.process_completed(task, segments)
            self.table_widget.refresh_row(task.uid)
        else:
            self.transcription_service.update_transcription_as_completed(task.uid, segments)
            self.table_widget.refresh_row(task.uid)

        if self.quit_on_complete:
            self.close()
            QApplication.quit()


    def on_task_error(self, task: FileTranscriptionTask, error: str):
        self._finished_tasks.add(task.uid)
        self._progress_state.pop(task.uid, None)
        self._download_progress_state.pop(task.uid, None)
        self.transcription_service.update_transcription_as_failed(task.uid, error)
        self.table_widget.refresh_row(task.uid)

        if self.quit_on_complete:
            self.close()
            QApplication.quit()

    def on_shortcuts_changed(self):
        self.menu_bar.reset_shortcuts()
        self.toolbar.reset_shortcuts()

    def resizeEvent(self, event):
        self.save_geometry()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.save_geometry()
        self.settings.settings.sync()

        if self.folder_watcher:
            try:
                self.folder_watcher.task_found.disconnect()
                if len(self.folder_watcher.directories()) > 0:
                    self.folder_watcher.removePaths(self.folder_watcher.directories())
            except Exception as e:
                logging.warning(f"Error cleaning up folder watcher: {e}")

        for worker in self.transcriber_workers:
            try:
                worker.task_started.disconnect()
                worker.task_progress.disconnect()
                worker.task_download_progress.disconnect()
                worker.task_checkpoint.disconnect()
                worker.task_error.disconnect()
                worker.task_completed.disconnect()
            except Exception as e:
                logging.warning(f"Error disconnecting signals: {e}")

        for worker in self.transcriber_workers:
            worker.stop()

        for thread in self.transcriber_threads:
            thread.quit()

        for thread in self.transcriber_threads:
            if thread.isRunning():
                if not thread.wait(10000):
                    logging.warning("Transcriber thread did not finish within 10s timeout, terminating")
                    thread.terminate()
                    if not thread.wait(2000):
                        logging.error("Transcriber thread could not be terminated")

        if self.transcription_viewer_widget is not None:
            self.transcription_viewer_widget.close()

        try:
            from buzz.widgets.application import Application
            app = Application.instance()
            if app and hasattr(app, 'close_database'):
                app.close_database()
        except Exception as e:
            logging.warning(f"Error closing database: {e}")

        logging.debug("MainWindow closeEvent completed")

        super().closeEvent(event)

    def save_geometry(self):
        self.settings.begin_group(Settings.Key.MAIN_WINDOW)
        self.settings.settings.setValue("geometry", self.saveGeometry())
        self.settings.end_group()

    def load_geometry(self):
        self.settings.begin_group(Settings.Key.MAIN_WINDOW)
        geometry = self.settings.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.setBaseSize(1240, 600)
            self.resize(1240, 600)
        self.settings.end_group()

    def _init_update_checker(self):
        """Initializes and runs the update checker."""
        self.update_checker = UpdateChecker(settings=self.settings, parent=self)
        self.update_checker.update_available.connect(self._on_update_available)

        # Allow disabling the automatic startup update check (e.g. in tests).
        # An in-flight QNetworkAccessManager request interferes with
        # ``multiprocessing`` spawn on Windows and crashes child transcription
        # processes; tests should also never depend on network availability.
        if os.getenv("BUZZ_DISABLE_UPDATE_CHECK"):
            logging.debug("Startup update check disabled via BUZZ_DISABLE_UPDATE_CHECK")
            return

        # Check for updates on startup
        self.update_checker.check_for_updates()

    def _on_update_available(self, update_info: UpdateInfo):
        """Called when an update is available."""
        self._update_info = update_info
        self.toolbar.set_update_available(True)

    def on_update_action_triggered(self):
        """Called when user clicks the update action in toolbar."""
        if self._update_info is None:
            return

        dialog = UpdateDialog(
            update_info=self._update_info,
            parent=self
        )
        dialog.exec()

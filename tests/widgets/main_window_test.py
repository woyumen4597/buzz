import logging
import os
import tempfile
from typing import List
from unittest.mock import patch, Mock

import pytest
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QKeyEvent, QAction
from PyQt6.QtWidgets import (
    QMessageBox,
    QPushButton,
    QToolBar,
    QMenuBar,
    QTableView,
)
from pytestqt.qtbot import QtBot

from buzz.locale import _
from buzz.db.entity.transcription import Transcription
from buzz.db.service.transcription_service import TranscriptionService
from buzz.model_loader import TranscriptionModel, ModelType, WhisperModelSize
from buzz.settings.settings import Settings
from buzz.transcriber.transcriber import (
    Task,
    OutputFormat,
    FileTranscriptionTask,
    TranscriptionOptions,
    FileTranscriptionOptions,
    Segment,
)
from buzz.widgets.main_window import MainWindow
from buzz.widgets.preferences_dialog.models.file_transcription_preferences import FileTranscriptionPreferences
from buzz.widgets.transcriber.file_transcriber_widget import FileTranscriberWidget

mock_transcriptions: List[Transcription] = [
    Transcription(status="completed"),
    Transcription(status="canceled"),
    Transcription(status="failed", error_message=_("Error")),
]


def get_test_asset(filename: str):
    return os.path.join(os.path.dirname(__file__), "../../testdata/", filename)


class TestMainWindow:
    def test_should_restore_unfinished_task_configuration(
        self, tmp_path, transcription_dao, transcription_service, monkeypatch
    ):
        source = tmp_path / "source.wav"
        source.write_bytes(b"audio")
        task = FileTranscriptionTask(
            transcription_options=TranscriptionOptions(
                language="zh",
                task=Task.TRANSLATE,
                model=TranscriptionModel(
                    model_type=ModelType.OPEN_AI_WHISPER_API,
                    whisper_model_size=None,
                ),
                temperature=(0.1, 0.7),
                initial_prompt="保留专有名词",
                enable_llm_translation=True,
                llm_prompt="准确翻译",
                llm_model="gpt-test",
            ),
            file_transcription_options=FileTranscriptionOptions(
                file_paths=[str(source)],
                output_formats={OutputFormat.SRT},
                translate=True,
            ),
            model_path="",
            file_path=str(source),
            fraction_downloaded=0.4,
        )
        task.segments = [Segment(start=0, end=100, text="partial")]
        task.checkpoint_next_chunk = 1
        transcription_service.create_transcription(task)
        transcription_service.update_transcription_progress(task.uid, 0.3)
        transcription_service.update_transcription_segment_checkpoint(task.uid, task)
        transcription = transcription_dao.find_by_id(str(task.uid))
        monkeypatch.setattr(
            "buzz.widgets.main_window.get_password", lambda _key: "current-token"
        )

        restored = MainWindow._restore_task(MainWindow.__new__(MainWindow), transcription)

        assert restored.transcription_options.initial_prompt == "保留专有名词"
        assert restored.transcription_options.temperature == (0.1, 0.7)
        assert restored.transcription_options.llm_prompt == "准确翻译"
        assert restored.transcription_options.llm_model == "gpt-test"
        assert restored.transcription_options.openai_access_token == "current-token"
        assert restored.file_transcription_options.translate is True
        assert restored.fraction_downloaded == 0.4
        assert restored.segments == task.segments
        assert restored.checkpoint_next_chunk == 1

    def test_should_set_window_title_and_icon(self, qtbot, transcription_service):
        window = MainWindow(transcription_service)
        qtbot.add_widget(window)
        assert window.windowTitle() == "Buzz"
        assert window.windowIcon().pixmap(QSize(64, 64)).isNull() is False
        window.close()

    def test_should_run_file_transcription_task(
        self, qtbot: QtBot, transcription_service
    ):
        window = MainWindow(transcription_service)

        self._import_file_and_start_transcription(window)

        open_transcript_action = self._get_toolbar_action(window, _("Open Transcript"))
        assert open_transcript_action.isEnabled() is False

        table_widget = self._get_tasks_table(window)
        qtbot.wait_until(
            self._get_assert_task_status_callback(table_widget, 0, "completed"),
            timeout=2 * 60 * 1000,
        )

        table_widget.setCurrentIndex(table_widget.model().index(0, 0))
        assert open_transcript_action.isEnabled()
        window.close()

    @staticmethod
    def _get_tasks_table(window: MainWindow) -> QTableView:
        return window.findChild(QTableView)

    def test_should_run_url_import_file_transcription_task(
        self, qtbot: QtBot, db, transcription_service
    ):
        window = MainWindow(transcription_service)
        menu: QMenuBar = window.menuBar()
        file_action = menu.actions()[0]
        import_url_action: QAction = file_action.menu().actions()[1]

        with patch(
            "buzz.widgets.import_url_dialog.ImportURLDialog.prompt"
        ) as prompt_mock:
            prompt_mock.return_value = "https://github.com/chidiwilliams/buzz/raw/main/testdata/whisper-french.mp3"
            import_url_action.trigger()

        file_transcriber_widget: FileTranscriberWidget = window.findChild(
            FileTranscriberWidget
        )
        run_button: QPushButton = file_transcriber_widget.findChild(QPushButton)
        run_button.click()

        table_widget = self._get_tasks_table(window)
        qtbot.wait_until(
            self._get_assert_task_status_callback(table_widget, 0, "completed"),
            timeout=2 * 60 * 1000,
        )

        window.close()

    @pytest.mark.timeout(300)
    def test_should_run_and_cancel_transcription_task(
        self, qtbot, db, transcription_service
    ):
        window = MainWindow(transcription_service)
        qtbot.add_widget(window)

        self._import_file_and_start_transcription(window, long_audio=True)

        table_widget = self._get_tasks_table(window)

        try:
            qtbot.wait_until(
                self._get_assert_task_status_callback(table_widget, 0, "in_progress"),
                timeout=60 * 1000,
            )
        except Exception:
            logging.error("Task never reached 'in_progress' status")
            assert False, "Task did not start as expected"

        logging.debug("Will cancel transcription task")

        table_widget.selectRow(0)
        
        # Force immediate processing of pending events before triggering cancellation
        qtbot.wait(100)
        
        window.toolbar.stop_transcription_action.trigger()
        
        # Give some time for the cancellation to be processed
        qtbot.wait(500)

        logging.debug("Will wait for task to reach 'canceled' status")

        try:
            qtbot.wait_until(
                self._get_assert_task_status_callback(table_widget, 0, "canceled"),
                timeout=30 * 1000,
            )
        except Exception:
            # On Windows, the cancellation might be slower, check final state
            final_status = self._get_status(table_widget, 0)
            logging.error(f"Task status after timeout: {final_status}")
            if "canceled" not in final_status.lower():
                assert False, f"Task did not cancel as expected. Final status: {final_status}"

        logging.debug("Task canceled")

        qtbot.wait(200)

        table_widget.selectRow(0)
        assert window.toolbar.stop_transcription_action.isEnabled() is False
        assert window.toolbar.open_transcript_action.isEnabled() is False

        window.close()

    @pytest.mark.parametrize("transcription_dao", [mock_transcriptions], indirect=True)
    def test_should_load_tasks_from_cache(
        self, qtbot, transcription_dao, transcription_segment_dao, monkeypatch
    ):
        # Mock the queue worker to prevent it from processing tasks
        mock_queue_worker = Mock()
        mock_queue_worker.task_started = Mock()
        mock_queue_worker.task_progress = Mock()
        mock_queue_worker.task_download_progress = Mock()
        mock_queue_worker.task_error = Mock()
        mock_queue_worker.task_completed = Mock()
        mock_queue_worker.completed = Mock()
        mock_queue_worker.cancel_task = Mock()
        mock_queue_worker.add_task = Mock()
        mock_queue_worker.stop = Mock()
        
        monkeypatch.setattr("buzz.widgets.main_window.FileTranscriberQueueWorker", Mock(return_value=mock_queue_worker))
        
        window = MainWindow(
            TranscriptionService(transcription_dao, transcription_segment_dao)
        )
        qtbot.add_widget(window)

        table_widget = self._get_tasks_table(window)
        assert table_widget.model().rowCount() == 3

        # Get all statuses and verify they match expected values
        statuses = [self._get_status(table_widget, i) for i in range(3)]
        expected_statuses = {"completed", "canceled", "failed"}
        assert set(statuses) == expected_statuses, f"Expected {expected_statuses}, got {statuses}"

        # Test that completed transcriptions enable the open action, others don't
        for i in range(3):
            table_widget.selectRow(i)
            status = self._get_status(table_widget, i)
            if status == "completed":
                assert window.toolbar.open_transcript_action.isEnabled()
            else:
                assert window.toolbar.open_transcript_action.isEnabled() is False

        window.close()

    def test_should_dispatch_tasks_across_transcriber_workers(
        self, qtbot, transcription_service, monkeypatch
    ):
        original_settings_value = Settings.value

        def configured_settings_value(settings, key, default_value, value_type=None):
            if key == Settings.Key.TRANSCRIPTION_CONCURRENCY:
                return 3
            return original_settings_value(settings, key, default_value, value_type)

        monkeypatch.setattr(Settings, "value", configured_settings_value)

        workers = []
        for _ in range(3):
            worker = Mock()
            for signal_name in (
                "task_started",
                "task_progress",
                "task_download_progress",
                "task_error",
                "task_completed",
                "completed",
            ):
                setattr(worker, signal_name, Mock())
            worker.is_running = False
            worker.tasks_queue = Mock()
            workers.append(worker)

        monkeypatch.setattr(
            "buzz.widgets.main_window.FileTranscriberQueueWorker",
            Mock(side_effect=workers),
        )

        window = MainWindow(transcription_service)
        qtbot.add_widget(window)
        window.transcription_service.create_transcription = Mock()
        window.table_widget.refresh_all = Mock()

        tasks = [Mock(), Mock(), Mock(), Mock()]
        for task in tasks:
            window.add_task(task)

        assert [args[0] for args, _ in workers[0].add_task.call_args_list] == [
            tasks[0],
            tasks[3],
        ]
        assert [args[0] for args, _ in workers[1].add_task.call_args_list] == [tasks[1]]
        assert [args[0] for args, _ in workers[2].add_task.call_args_list] == [tasks[2]]

        window.close()

    def test_should_dispatch_to_idle_worker_over_busy_ones(
        self, qtbot, transcription_service, monkeypatch
    ):
        original_settings_value = Settings.value

        def configured_settings_value(settings, key, default_value, value_type=None):
            if key == Settings.Key.TRANSCRIPTION_CONCURRENCY:
                return 3
            return original_settings_value(settings, key, default_value, value_type)

        monkeypatch.setattr(Settings, "value", configured_settings_value)

        def make_worker():
            worker = Mock()
            for signal_name in (
                "task_started",
                "task_progress",
                "task_download_progress",
                "task_error",
                "task_completed",
                "completed",
            ):
                setattr(worker, signal_name, Mock())
            worker.tasks_queue = Mock()
            return worker

        workers = [make_worker(), make_worker(), make_worker()]
        # Workers 0 and 1 are busy; worker 2 is idle.
        workers[0].is_running = True
        workers[0].tasks_queue.qsize.return_value = 5
        workers[1].is_running = True
        workers[1].tasks_queue.qsize.return_value = 2
        workers[2].is_running = False

        monkeypatch.setattr(
            "buzz.widgets.main_window.FileTranscriberQueueWorker",
            Mock(side_effect=workers),
        )

        window = MainWindow(transcription_service)
        qtbot.add_widget(window)
        window.transcription_service.create_transcription = Mock()
        window.table_widget.refresh_all = Mock()

        window.add_task(Mock())
        workers[2].add_task.assert_called_once()

        # Now all workers busy: the task must go to the least-loaded queue.
        workers[2].is_running = True
        workers[2].tasks_queue.qsize.return_value = 7
        window.add_task(Mock())
        workers[1].add_task.assert_called_once()

        window.close()

    @pytest.mark.parametrize("transcription_dao", [mock_transcriptions], indirect=True)
    def test_should_clear_history_with_rows_selected(
        self, qtbot, transcription_dao, transcription_segment_dao
    ):
        window = MainWindow(
            TranscriptionService(transcription_dao, transcription_segment_dao)
        )
        qtbot.add_widget(window)

        table_widget = self._get_tasks_table(window)
        table_widget.selectAll()

        with patch("PyQt6.QtWidgets.QMessageBox.exec") as question_message_box_mock:
            question_message_box_mock.return_value = QMessageBox.StandardButton.Yes
            window.toolbar.clear_history_action.trigger()

        assert table_widget.model().rowCount() == 0
        window.close()

    @pytest.mark.parametrize("transcription_dao", [mock_transcriptions], indirect=True)
    def test_should_have_clear_history_action_disabled_with_no_rows_selected(
        self, qtbot, transcription_dao, transcription_segment_dao
    ):
        window = MainWindow(
            TranscriptionService(transcription_dao, transcription_segment_dao)
        )
        qtbot.add_widget(window)

        assert window.toolbar.clear_history_action.isEnabled() is False
        window.close()

    @pytest.mark.parametrize("transcription_dao", [mock_transcriptions], indirect=True)
    def test_should_open_transcription_viewer_when_menu_action_is_clicked(
        self, qtbot, transcription_dao, transcription_segment_dao
    ):
        window = MainWindow(
            TranscriptionService(transcription_dao, transcription_segment_dao)
        )
        qtbot.add_widget(window)

        table_widget = self._get_tasks_table(window)

        # Find and select the completed transcription row
        completed_row = None
        for i in range(table_widget.model().rowCount()):
            if self._get_status(table_widget, i) == "completed":
                completed_row = i
                break

        assert completed_row is not None, "No completed transcription found"
        table_widget.selectRow(completed_row)

        window.toolbar.open_transcript_action.trigger()

        assert window.transcription_viewer_widget is not None

        window.close()

    @pytest.mark.parametrize("transcription_dao", [mock_transcriptions], indirect=True)
    def test_should_open_transcription_viewer_when_return_clicked(
        self, qtbot, transcription_dao, transcription_segment_dao
    ):
        window = MainWindow(
            TranscriptionService(transcription_dao, transcription_segment_dao)
        )
        qtbot.add_widget(window)

        table_widget = self._get_tasks_table(window)

        # Find and select the completed transcription row
        completed_row = None
        for i in range(table_widget.model().rowCount()):
            if self._get_status(table_widget, i) == "completed":
                completed_row = i
                break

        assert completed_row is not None, "No completed transcription found"
        table_widget.selectRow(completed_row)

        table_widget.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Return,
                Qt.KeyboardModifier.NoModifier,
                "\r",
            )
        )

        assert window.transcription_viewer_widget is not None

        window.close()

    @pytest.mark.parametrize("transcription_dao", [mock_transcriptions], indirect=True)
    def test_should_have_open_transcript_action_disabled_with_no_rows_selected(
        self, qtbot, transcription_dao, transcription_segment_dao
    ):
        window = MainWindow(
            TranscriptionService(transcription_dao, transcription_segment_dao)
        )
        qtbot.add_widget(window)

        assert window.toolbar.open_transcript_action.isEnabled() is False
        window.close()

    def test_import_folder_opens_file_transcriber_with_supported_files(
        self, qtbot, transcription_service
    ):
        window = MainWindow(transcription_service)
        qtbot.add_widget(window)

        with tempfile.TemporaryDirectory() as folder:
            # Create supported and unsupported files
            supported = ["audio.mp3", "video.mp4", "clip.wav"]
            unsupported = ["document.txt", "image.png"]
            subdir = os.path.join(folder, "sub")
            os.makedirs(subdir)
            nested = "nested.flac"

            for name in supported + unsupported:
                open(os.path.join(folder, name), "w").close()
            open(os.path.join(subdir, nested), "w").close()

            with patch("PyQt6.QtWidgets.QFileDialog.getExistingDirectory") as mock_dir, \
                 patch.object(window, "open_file_transcriber_widget") as mock_open:
                mock_dir.return_value = folder
                window.on_import_folder_action_triggered()

            collected = mock_open.call_args[0][0]
            collected_names = {os.path.basename(p) for p in collected}
            assert collected_names == {"audio.mp3", "video.mp4", "clip.wav", "nested.flac"}

        window.close()

    def test_import_folder_does_nothing_when_cancelled(
        self, qtbot, transcription_service
    ):
        window = MainWindow(transcription_service)
        qtbot.add_widget(window)

        with patch("PyQt6.QtWidgets.QFileDialog.getExistingDirectory") as mock_dir, \
             patch.object(window, "open_file_transcriber_widget") as mock_open:
            mock_dir.return_value = ""
            window.on_import_folder_action_triggered()

        mock_open.assert_not_called()
        window.close()

    def test_import_folder_does_nothing_when_no_supported_files(
        self, qtbot, transcription_service
    ):
        window = MainWindow(transcription_service)
        qtbot.add_widget(window)

        with tempfile.TemporaryDirectory() as folder:
            open(os.path.join(folder, "readme.txt"), "w").close()
            open(os.path.join(folder, "image.jpg"), "w").close()

            with patch("PyQt6.QtWidgets.QFileDialog.getExistingDirectory") as mock_dir, \
                 patch.object(window, "open_file_transcriber_widget") as mock_open:
                mock_dir.return_value = folder
                window.on_import_folder_action_triggered()

        mock_open.assert_not_called()
        window.close()

    def test_remembers_last_import_folder_for_file_dialog(
        self, qtbot, transcription_service
    ):
        from buzz.settings.settings import Settings

        window = MainWindow(transcription_service)
        qtbot.add_widget(window)

        with tempfile.TemporaryDirectory() as folder:
            file_path = os.path.join(folder, "audio.mp3")
            open(file_path, "w").close()

            with patch(
                "PyQt6.QtWidgets.QFileDialog.getOpenFileNames"
            ) as mock_dialog, patch.object(
                window, "open_file_transcriber_widget"
            ):
                mock_dialog.return_value = ([file_path], "")
                window.on_new_transcription_action_triggered()

            # The selected file's folder should be persisted
            assert window.settings.value(
                Settings.Key.LAST_IMPORT_FOLDER, ""
            ) == os.path.dirname(file_path)

            # On the next open, the dialog should be pre-pointed at that folder
            with patch(
                "PyQt6.QtWidgets.QFileDialog.getOpenFileNames"
            ) as mock_dialog, patch.object(
                window, "open_file_transcriber_widget"
            ):
                mock_dialog.return_value = ([], "")
                window.on_new_transcription_action_triggered()
                assert mock_dialog.call_args[0][2] == os.path.dirname(file_path)

        window.settings.set_value(Settings.Key.LAST_IMPORT_FOLDER, "")
        window.close()

    def test_remembers_last_import_folder_for_folder_dialog(
        self, qtbot, transcription_service
    ):
        from buzz.settings.settings import Settings

        window = MainWindow(transcription_service)
        qtbot.add_widget(window)

        with tempfile.TemporaryDirectory() as folder:
            open(os.path.join(folder, "audio.mp3"), "w").close()

            with patch(
                "PyQt6.QtWidgets.QFileDialog.getExistingDirectory"
            ) as mock_dir, patch.object(window, "open_file_transcriber_widget"):
                mock_dir.return_value = folder
                window.on_import_folder_action_triggered()

            assert window.settings.value(Settings.Key.LAST_IMPORT_FOLDER, "") == folder

            # The next folder dialog should start at the remembered folder
            with patch(
                "PyQt6.QtWidgets.QFileDialog.getExistingDirectory"
            ) as mock_dir, patch.object(window, "open_file_transcriber_widget"):
                mock_dir.return_value = ""
                window.on_import_folder_action_triggered()
                assert mock_dir.call_args[0][2] == folder

        window.settings.set_value(Settings.Key.LAST_IMPORT_FOLDER, "")
        window.close()

    @staticmethod
    def _import_file_and_start_transcription(
        window: MainWindow, long_audio: bool = False
    ):
        default_prefs = FileTranscriptionPreferences(
            language=None,
            task=Task.TRANSCRIBE,
            model=TranscriptionModel(
                model_type=ModelType.WHISPER,
                whisper_model_size=WhisperModelSize.TINY,
            ),
            word_level_timings=False,
            extract_speech=False,
            initial_prompt="",
            enable_llm_translation=False,
            llm_prompt="",
            llm_model="",
            output_formats={OutputFormat.TXT},
        )

        with patch(
            "PyQt6.QtWidgets.QFileDialog.getOpenFileNames"
        ) as open_file_names_mock, patch.object(
            FileTranscriberWidget, "load_preferences", return_value=default_prefs
        ):
            open_file_names_mock.return_value = (
                [
                    get_test_asset(
                        "audio-long.mp3" if long_audio else "whisper-french.mp3"
                    )
                ],
                "",
            )
            new_transcription_action = TestMainWindow._get_toolbar_action(
                window, _("New File Transcription")
            )
            new_transcription_action.trigger()

        file_transcriber_widget: FileTranscriberWidget = window.findChild(
            FileTranscriberWidget
        )
        run_button: QPushButton = file_transcriber_widget.findChild(QPushButton)
        run_button.click()

    @staticmethod
    def _get_assert_task_status_callback(
        table_widget: QTableView,
        row_index: int,
        expected_status: str,
    ):
        def assert_task_status():
            assert table_widget.model().rowCount() > 0
            assert expected_status in TestMainWindow._get_status(
                table_widget, row_index
            )

        return assert_task_status

    @staticmethod
    def _get_status(table_widget: QTableView, row_index: int):
        return table_widget.model().index(row_index, 9).data()

    @staticmethod
    def _get_toolbar_action(window: MainWindow, text: str):
        toolbar: QToolBar = window.findChild(QToolBar)
        return [action for action in toolbar.actions() if action.text() == text][0]


def _make_progress_task() -> FileTranscriptionTask:
    return FileTranscriptionTask(
        transcription_options=TranscriptionOptions(
            model=TranscriptionModel(
                model_type=ModelType.WHISPER_CPP,
                whisper_model_size=WhisperModelSize.TINY,
            )
        ),
        file_transcription_options=FileTranscriptionOptions(),
        file_path=get_test_asset("whisper-french.mp3"),
        model_path="mock_path",
    )


class TestTaskProgressThrottling:
    """Progress events are coalesced before hitting the DB / task table."""

    def test_progress_within_one_percent_is_coalesced(
        self, qtbot, transcription_service
    ):
        window = MainWindow(transcription_service)
        qtbot.add_widget(window)
        task = _make_progress_task()
        window.transcription_service.update_transcription_progress = Mock()
        window.table_widget.refresh_row = Mock()

        window.on_task_progress(task, 0.0)
        window.on_task_progress(task, 0.004)
        window.on_task_progress(task, 0.009)
        assert window.transcription_service.update_transcription_progress.call_count == 1

        window.on_task_progress(task, 0.02)
        assert window.transcription_service.update_transcription_progress.call_count == 2
        window.transcription_service.update_transcription_progress.assert_called_with(
            task.uid, 0.02
        )

    def test_completion_forces_final_progress_and_ignores_late_events(
        self, qtbot, transcription_service
    ):
        window = MainWindow(transcription_service)
        qtbot.add_widget(window)
        task = _make_progress_task()
        window.transcription_service.update_transcription_progress = Mock()
        window.transcription_service.update_transcription_as_completed = Mock()
        window.table_widget.refresh_row = Mock()

        window.on_task_progress(task, 0.5)
        window.on_task_completed(task, [])

        window.transcription_service.update_transcription_progress.assert_called_with(
            task.uid, 1.0
        )
        # A progress event arriving after completion must not regress the value.
        window.on_task_progress(task, 0.55)
        assert window.transcription_service.update_transcription_progress.call_count == 2

    def test_error_ignores_late_progress_events(self, qtbot, transcription_service):
        window = MainWindow(transcription_service)
        qtbot.add_widget(window)
        task = _make_progress_task()
        window.transcription_service.update_transcription_progress = Mock()
        window.transcription_service.update_transcription_as_failed = Mock()
        window.table_widget.refresh_row = Mock()

        window.on_task_progress(task, 0.3)
        window.on_task_error(task, "boom")
        window.on_task_progress(task, 0.4)

        assert window.transcription_service.update_transcription_progress.call_count == 1

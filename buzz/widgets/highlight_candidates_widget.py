"""Lightweight GUI launcher for the standalone highlight candidate exporter."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from PyQt6.QtCore import QObject, QThread, QUrl, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from buzz.locale import _


class HighlightGenerationWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, args: argparse.Namespace, cancel_event: threading.Event):
        super().__init__()
        self.args = args
        self.cancel_event = cancel_event

    @pyqtSlot()
    def run(self):
        try:
            # Keep the media-heavy implementation off the GUI thread.
            from buzz.highlights.cli import run

            output_dir = run(
                self.args,
                progress_callback=lambda completed, total, message: self.progress.emit(
                    completed, total, message
                ),
                cancel_event=self.cancel_event,
            )
            self.finished.emit(str(output_dir))
        except Exception as exc:  # surfaced in the page instead of crashing Qt
            self.failed.emit(str(exc))


class HighlightCandidatesWidget(QWidget):
    """Select inputs and launch the existing static highlight browser."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("HighlightCandidates")
        self._thread: Optional[QThread] = None
        self._worker: Optional[HighlightGenerationWorker] = None
        self._cancel_event: Optional[threading.Event] = None
        self._output_dir: Optional[Path] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        eyebrow = QLabel(_("BUZZ / VIDEO WORKSPACE"))
        eyebrow.setObjectName("Eyebrow")
        layout.addWidget(eyebrow)

        title = QLabel(_("Automatic video highlights"))
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            _("Choose a video and generate a highlight reel automatically. The default reel length is one third of the source video.")
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.video_input = QLineEdit(self)
        self.video_input.setObjectName("HighlightVideoInput")
        self.video_input.setPlaceholderText(_("Choose a video file"))
        video_browse = QPushButton(_("Browse"), self)
        video_browse.clicked.connect(self._choose_video)
        form.addRow(_("Video"), self._with_button(self.video_input, video_browse))

        self.srt_input = QLineEdit(self)
        self.srt_input.setObjectName("HighlightSrtInput")
        self.srt_input.setPlaceholderText(_("Optional SRT transcript"))
        srt_browse = QPushButton(_("Browse"), self)
        srt_browse.clicked.connect(self._choose_srt)
        form.addRow(_("Transcript (SRT)"), self._with_button(self.srt_input, srt_browse))

        # Output location and candidate tuning are intentionally internal in
        # one-click mode; the default is <video>_highlights.
        self._window_seconds = 20.0
        self._stride_seconds = 10.0
        self._max_candidates = 0
        self.target_duration = self._double_spin(0.0, 0.0, 24 * 3600.0)
        self.target_duration.setSpecialValueText(_("Automatic: one third of source video"))
        form.addRow(_("Highlight duration (seconds, 0 = automatic)"), self.target_duration)
        form_container = QWidget(self)
        form_container.setObjectName("HighlightFormContainer")
        form_container.setMaximumWidth(820)
        form_container.setLayout(form)
        layout.addWidget(form_container, 0, Qt.AlignmentFlag.AlignLeft)

        self.no_scene_detection = False
        self.no_previews = False
        self.ignore_static_scenes = True
        self.auto_edit = True

        actions = QHBoxLayout()
        self.run_button = QPushButton(_("Generate highlight reel"), self)
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.start_generation)
        actions.addWidget(self.run_button)
        self.open_button = QPushButton(_("Open result"), self)
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_result)
        actions.addWidget(self.open_button)
        self.cancel_button = QPushButton(_("Cancel"), self)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        actions.addWidget(self.cancel_button)
        actions.addStretch()
        layout.addLayout(actions)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel(_("Choose a video to begin."), self)
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("HighlightStatus")
        layout.addWidget(self.status_label)
        layout.addStretch()

    @staticmethod
    def _with_button(field: QLineEdit, button: QPushButton) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field, 1)
        row.addWidget(button)
        return container

    @staticmethod
    def _double_spin(value: float, minimum: float, maximum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        spin.setDecimals(1)
        return spin

    def _choose_video(self):
        path, _filter = QFileDialog.getOpenFileName(
            self,
            _("Choose video"),
            "",
            _("Video files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;All files (*)"),
        )
        if path:
            self.video_input.setText(path)

    def _choose_srt(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, _("Choose SRT transcript"), "", _("SubRip subtitles (*.srt);;All files (*)")
        )
        if path:
            self.srt_input.setText(path)

    def _build_args(self) -> argparse.Namespace:
        video = Path(self.video_input.text().strip()).expanduser()
        srt_text = self.srt_input.text().strip()
        return SimpleNamespace(
            video=video,
            output_dir=None,
            srt=Path(srt_text).expanduser() if srt_text else None,
            window_seconds=self._window_seconds,
            stride_seconds=self._stride_seconds,
            padding_seconds=1.5,
            scene_threshold=0.35,
            max_candidates=self._max_candidates,
            no_scene_detection=self.no_scene_detection,
            no_previews=self.no_previews,
            static_motion_threshold=1.5,
            auto_edit=self.auto_edit,
            target_duration=self.target_duration.value(),
            max_auto_clips=0,
            score_threshold=0.0,
            gif=False,
            gif_limit=0,
            keep_existing=True,
            open_html=False,
            serve_html=True,
            keep_static_scenes=False,
            verbose=False,
        )

    def start_generation(self):
        video = Path(self.video_input.text().strip()).expanduser()
        if not video.is_file():
            self.status_label.setText(_("Select an existing video file first."))
            return
        if self._thread is not None and self._thread.isRunning():
            return

        self._output_dir = None
        self.open_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(_("Generating highlight reel. This may take a while..."))

        self._cancel_event = threading.Event()
        self._thread = QThread(self)
        self._worker = HighlightGenerationWorker(self._build_args(), self._cancel_event)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._generation_progress)
        self._worker.finished.connect(self._generation_finished)
        self._worker.failed.connect(self._generation_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()

    @pyqtSlot(int, int, str)
    def _generation_progress(self, completed: int, total: int, message: str):
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(min(completed, total))
        self.status_label.setText(f"{message} ({self.progress_bar.value() * 100 // max(1, total)}%)")

    @pyqtSlot(str)
    def _generation_finished(self, output_dir: str):
        self._output_dir = Path(output_dir)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.open_button.setEnabled(True)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText(_("Highlight reel is ready: {}" ).format(output_dir))

    @pyqtSlot(str)
    def _generation_failed(self, message: str):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText(_("Candidate generation failed: {}" ).format(message))

    def cancel_generation(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.status_label.setText(_("Canceling. Completed files will be kept for the next run..."))

    def _thread_finished(self):
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._cancel_event = None

    def open_result(self):
        if self._output_dir is None:
            return
        if self._output_dir.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_dir)))
            return
        server_url = self._output_dir / ".highlight-server-url"
        if server_url.is_file():
            QDesktopServices.openUrl(QUrl(server_url.read_text(encoding="utf-8").strip() + "/index.html"))
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_dir / "index.html")))

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._thread.quit()
            self._thread.wait(1000)
        super().closeEvent(event)

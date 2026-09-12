"""Lightweight GUI launcher for the standalone highlight candidate exporter."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from PyQt6.QtCore import QObject, QThread, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
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

        eyebrow = QLabel(_("BUZZ / VIDEO WORKSPACE"))
        eyebrow.setObjectName("Eyebrow")
        layout.addWidget(eyebrow)

        title = QLabel(_("Video highlight candidates"))
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            _("Generate reviewable candidate clips and open the static browser when ready.")
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)

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

        self.output_input = QLineEdit(self)
        self.output_input.setObjectName("HighlightOutputInput")
        self.output_input.setPlaceholderText(_("Defaults to <video>_highlights"))
        output_browse = QPushButton(_("Browse"), self)
        output_browse.clicked.connect(self._choose_output)
        form.addRow(_("Output folder"), self._with_button(self.output_input, output_browse))

        self.window_seconds = self._double_spin(20.0, 1.0, 300.0)
        form.addRow(_("Window (seconds)"), self.window_seconds)
        self.stride_seconds = self._double_spin(10.0, 0.5, 300.0)
        form.addRow(_("Stride (seconds)"), self.stride_seconds)
        self.max_candidates = QSpinBox(self)
        self.max_candidates.setRange(1, 2000)
        self.max_candidates.setValue(200)
        form.addRow(_("Maximum candidates"), self.max_candidates)
        layout.addLayout(form)

        self.no_scene_detection = QCheckBox(_("Skip scene detection"), self)
        self.no_previews = QCheckBox(_("Skip MP4 previews for a faster first pass"), self)
        self.ignore_static_scenes = QCheckBox(_("Auto-ignore mostly static scenes"), self)
        self.ignore_static_scenes.setChecked(True)
        layout.addWidget(self.no_scene_detection)
        layout.addWidget(self.no_previews)
        layout.addWidget(self.ignore_static_scenes)

        actions = QHBoxLayout()
        self.run_button = QPushButton(_("Generate candidates"), self)
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

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(self, _("Choose output folder"))
        if path:
            self.output_input.setText(path)

    def _build_args(self) -> argparse.Namespace:
        video = Path(self.video_input.text().strip()).expanduser()
        srt_text = self.srt_input.text().strip()
        output_text = self.output_input.text().strip()
        return SimpleNamespace(
            video=video,
            output_dir=Path(output_text).expanduser() if output_text else None,
            srt=Path(srt_text).expanduser() if srt_text else None,
            window_seconds=self.window_seconds.value(),
            stride_seconds=self.stride_seconds.value(),
            padding_seconds=1.5,
            scene_threshold=0.35,
            max_candidates=self.max_candidates.value(),
            no_scene_detection=self.no_scene_detection.isChecked(),
            no_previews=self.no_previews.isChecked(),
            static_motion_threshold=1.5,
            gif=False,
            gif_limit=0,
            keep_existing=True,
            open_html=False,
            serve_html=True,
            keep_static_scenes=not self.ignore_static_scenes.isChecked(),
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
        self.status_label.setText(_("Generating candidates. This may take a while..."))

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
        self.status_label.setText(_("Candidates are ready: {}" ).format(output_dir))

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

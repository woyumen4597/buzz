import enum
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID

from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QSignalBlocker
from PyQt6.QtCore import pyqtSignal, QModelIndex
from PyQt6.QtSql import QSqlTableModel, QSqlRecord
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QMenu,
    QHeaderView,
    QTableView,
    QAbstractItemView,
    QFrame,
    QStyledItemDelegate,
)

from buzz.db.entity.transcription import Transcription
from buzz.locale import _
from buzz.settings.settings import Settings
from buzz.transcriber.transcriber import FileTranscriptionTask, Task, TASK_LABEL_TRANSLATIONS
from buzz.widgets.record_delegate import RecordDelegate
from buzz.widgets.transcription_record import TranscriptionRecord


class Column(enum.Enum):
    ID = 0
    ERROR_MESSAGE = 1
    EXPORT_FORMATS = 2
    FILE = 3
    OUTPUT_FOLDER = 4
    PROGRESS = 5
    LANGUAGE = 6
    MODEL_TYPE = 7
    SOURCE = 8
    STATUS = 9
    TASK = 10
    TIME_ENDED = 11
    TIME_QUEUED = 12
    TIME_STARTED = 13
    URL = 14
    WHISPER_MODEL_SIZE = 15
    HUGGING_FACE_MODEL_ID = 16
    WORD_LEVEL_TIMINGS = 17
    EXTRACT_SPEECH = 18
    NAME = 19
    NOTES = 20


@dataclass
class ColDef:
    id: str
    header: str
    column: Column
    width: Optional[int] = None
    delegate: Optional[QStyledItemDelegate] = None
    hidden_toggleable: bool = True
    # Floor for both viewport compaction and widths restored from settings, so a
    # column cannot come back from a previous session too narrow to read.
    min_width: int = 120


def record_download_progress(record: QSqlRecord) -> Optional[float]:
    """Download fraction while a URL task is still downloading, else None.

    URL tasks download before transcribing, and transcription progress stays at
    0 for the whole download. Both the status and task columns key off this so
    the download phase is not reported as a transcription stuck at 0%.
    """
    if FileTranscriptionTask.Status(record.value("status")) != (
        FileTranscriptionTask.Status.IN_PROGRESS
    ):
        return None

    progress = record.value("progress") or 0.0
    download_progress = record.value("download_progress") or 0.0
    if progress <= 0 and 0 < download_progress < 1:
        return download_progress
    return None


def format_record_task_text(record: QSqlRecord) -> str:
    # Nothing is being transcribed or translated yet while bytes are still
    # coming down, so name the phase that is actually running.
    if record_download_progress(record) is not None:
        return _("Downloading")
    return TASK_LABEL_TRANSLATIONS[Task(record.value("task"))]


def format_record_status_text(record: QSqlRecord) -> str:
    status = FileTranscriptionTask.Status(record.value("status"))
    match status:
        case FileTranscriptionTask.Status.IN_PROGRESS:
            in_progress_label = _("In Progress")
            progress = record.value("progress") or 0.0
            started_at = record.value("time_started")

            download_progress = record_download_progress(record)
            if download_progress is not None:
                return _("{status} ({progress:.0%})").format(
                    status=_("Downloading"), progress=download_progress
                )
            if started_at and progress and progress > 0 and progress < 1:
                try:
                    started_dt = datetime.fromisoformat(started_at)
                    elapsed = datetime.now() - started_dt
                    elapsed_s = elapsed.total_seconds()
                    if elapsed_s > 2:
                        remaining_s = (1 - progress) * elapsed_s / progress
                        eta_str = TranscriptionTasksTableWidget.format_timedelta(
                            timedelta(seconds=int(remaining_s))
                        )
                        return _("{status} ({progress:.0%}) — ETA: {eta}").format(
                            status=in_progress_label, progress=progress, eta=eta_str
                        )
                except (ValueError, OSError):
                    pass
            return _("{status} ({progress:.0%})").format(
                status=in_progress_label, progress=progress
            )
        case FileTranscriptionTask.Status.COMPLETED:
            status = _("Completed")
            started_at = record.value("time_started")
            completed_at = record.value("time_ended")
            if started_at != "" and completed_at != "":
                return _("{status} ({duration})").format(
                    status=status,
                    duration=TranscriptionTasksTableWidget.format_timedelta(
                        datetime.fromisoformat(completed_at)
                        - datetime.fromisoformat(started_at)
                    ),
                )
            return status
        case FileTranscriptionTask.Status.FAILED:
            failed_label = _("Failed")
            return f'{failed_label} ({record.value("error_message")})'
        case FileTranscriptionTask.Status.CANCELED:
            return _("Canceled")
        case FileTranscriptionTask.Status.QUEUED:
            return _("Queued")
        case FileTranscriptionTask.Status.SKIPPED:
            return _("Skipped")
        case _: # Case to handle UNKNOWN status
            return ""

column_definitions = [
    ColDef(
        id="file_name",
        header=_("File Name / URL"),
        column=Column.FILE,
        width=400,
        min_width=240,
        delegate=RecordDelegate(
            text_getter=lambda record: record.value("name") or (
                os.path.basename(record.value("file")) if record.value("file")
                else record.value("url") or ""
            )
        ),
        hidden_toggleable=False,
    ),
    ColDef(
        id="model",
        header=_("Model"),
        column=Column.MODEL_TYPE,
        width=180,
        min_width=130,
        delegate=RecordDelegate(
            text_getter=lambda record: str(TranscriptionRecord.model(record))
        ),
    ),
    ColDef(
        id="task",
        header=_("Task"),
        column=Column.TASK,
        width=120,
        min_width=90,
        delegate=RecordDelegate(text_getter=format_record_task_text),
    ),
    ColDef(
        id="status",
        header=_("Status"),
        column=Column.STATUS,
        # Status carries the longest text in the table: a progress percentage
        # plus an ETA, or a whole error message. 180px elided all of it.
        width=320,
        min_width=220,
        delegate=RecordDelegate(text_getter=format_record_status_text),
        hidden_toggleable=True,
    ),

    ColDef(
        id="date_completed",
        header=_("Date Completed"),
        column=Column.TIME_ENDED,
        width=180,
        min_width=180,
        delegate=RecordDelegate(
            text_getter=lambda record: datetime.fromisoformat(
                record.value("time_ended")
            ).strftime("%Y-%m-%d %H:%M:%S")
            if record.value("time_ended") != ""
            else ""
        ),
    ),    ColDef(
        id="date_added",
        header=_("Date Added"),
        column=Column.TIME_QUEUED,
        width=180,
        min_width=180,
        delegate=RecordDelegate(
            text_getter=lambda record: datetime.fromisoformat(
                record.value("time_queued")
            ).strftime("%Y-%m-%d %H:%M:%S")
        ),
    ),
    ColDef(
        id="notes",
        header=_("Notes"),
        column=Column.NOTES,
        width=300,
        delegate=RecordDelegate(
            text_getter=lambda record: record.value("notes") or ""
        ),
        hidden_toggleable=True,
    ),
]

class TranscriptionTasksTableHeaderView(QHeaderView):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        
        # Add reset column order option
        menu.addAction(_("Reset Column Order")).triggered.connect(self.parent().reset_column_order)
        menu.addSeparator()
        
        # Add column visibility toggles
        for definition in column_definitions:
            if definition.hidden_toggleable:
                action = menu.addAction(definition.header)
                action.setCheckable(True)
                action.setChecked(not self.parent().isColumnHidden(definition.column.value))
                action.toggled.connect(
                    lambda checked, column_index=definition.column.value: self.on_column_checked(
                        column_index, checked
                    )
                )
        menu.exec(event.globalPos())

    def on_column_checked(self, column_index: int, checked: bool):
        # Find the column definition for this index
        column_def = None
        for definition in column_definitions:
            if definition.column.value == column_index:
                column_def = definition
                break

        # If we're hiding the column, save its current width first
        if not checked and not self.parent().isColumnHidden(column_index):
            current_width = self.parent().columnWidth(column_index)
            if current_width > 0:  # Only save if there's a meaningful width
                self.parent().settings.begin_group(self.parent().settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_WIDTHS)
                self.parent().settings.settings.setValue(column_def.id, current_width)
                self.parent().settings.end_group()

        # Update the visibility state on the table view (not header view)
        self.parent().setColumnHidden(column_index, not checked)
        
        # Save current column order before any reloading
        self.parent().save_column_order()
        
        # Save both visibility and widths after the change
        self.parent().save_column_visibility()
        self.parent().save_column_widths()
        
        # Ensure settings are synchronized
        self.parent().settings.settings.sync()
        
        # Force a complete refresh of the table
        self.parent().viewport().update()
        self.parent().repaint()
        self.parent().horizontalHeader().update()
        self.parent().updateGeometry()
        self.parent().adjustSize()
        
        # Force a model refresh to ensure the view is updated
        self.parent().model().layoutChanged.emit()

        self.parent().reload_column_order_from_settings()

class TranscriptionTasksTableWidget(QTableView):
    return_clicked = pyqtSignal()
    delete_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.transcription_service = None
        self.setObjectName("TaskTable")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setShowGrid(False)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)

        self.setHorizontalHeader(TranscriptionTasksTableHeaderView(Qt.Orientation.Horizontal, self))

        self._model = QSqlTableModel()
        self._model.setTable("transcription")
        self._model.setEditStrategy(QSqlTableModel.EditStrategy.OnManualSubmit)
        self._model.setSort(Column.TIME_QUEUED.value, Qt.SortOrder.DescendingOrder)

        self.setModel(self._model)

        for i in range(self.model().columnCount()):
            self.hideColumn(i)

        self.settings = Settings()

        # Set up column headers and delegates
        for definition in column_definitions:
            self.model().setHeaderData(
                definition.column.value,
                Qt.Orientation.Horizontal,
                definition.header,
            )
            if definition.delegate is not None:
                self.setItemDelegateForColumn(
                    definition.column.value, definition.delegate
                )

        # Load column visibility
        self.load_column_visibility()

        self.model().select()
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(48)
        self.setAlternatingRowColors(True)
        
        # Enable column sorting and moving
        self.setSortingEnabled(True)
        self.horizontalHeader().setSectionsMovable(True)
        self.horizontalHeader().setSectionsClickable(True)
        self.horizontalHeader().setSortIndicatorShown(True)
        self.horizontalHeader().setHighlightSections(False)
        self.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        # Connect signals for column resize and move
        self.horizontalHeader().sectionResized.connect(self.on_column_resized)
        self.horizontalHeader().sectionMoved.connect(self.on_column_moved)
        self.horizontalHeader().sortIndicatorChanged.connect(self.on_sort_indicator_changed)

        # Load saved column order, widths, and sort state
        self.load_column_order()
        self.load_column_widths()
        self.load_sort_state()


        # Reload column visibility after all reordering is complete
        self.load_column_visibility()

    def _fit_columns_to_viewport(self):
        available_width = self.viewport().width()
        visible_columns = [
            definition.column.value
            for definition in column_definitions
            if not self.isColumnHidden(definition.column.value)
        ]
        if available_width <= 0 or not visible_columns:
            return

        current_widths = {
            column: self.columnWidth(column) for column in visible_columns
        }
        if sum(current_widths.values()) <= available_width:
            return

        minimum_widths = {
            definition.column.value: definition.min_width
            for definition in column_definitions
        }
        scale = available_width / sum(current_widths.values())
        compact_widths = {
            column: max(
                minimum_widths.get(column, 120),
                int(width * scale),
            )
            for column, width in current_widths.items()
        }

        overflow = sum(compact_widths.values()) - available_width
        while overflow > 0:
            candidates = [
                column
                for column in visible_columns
                if compact_widths[column] > minimum_widths.get(column, 120)
            ]
            if not candidates:
                break
            column = max(
                candidates,
                key=lambda item: compact_widths[item] - minimum_widths.get(item, 120),
            )
            reduction = min(
                overflow,
                compact_widths[column] - minimum_widths.get(column, 120),
            )
            compact_widths[column] -= reduction
            overflow -= reduction

        with QSignalBlocker(self.horizontalHeader()):
            for column, width in compact_widths.items():
                self.setColumnWidth(column, width)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_columns_to_viewport()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._fit_columns_to_viewport()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        
        # Add transcription actions if a row is selected
        selected_rows = self.selectionModel().selectedRows()
        if selected_rows:
            transcription = self.transcription(selected_rows[0])

            # Add restart/continue action for failed/canceled/skipped tasks
            if transcription.status in ["failed", "canceled", "skipped"]:
                restart_action = menu.addAction(_("Restart Transcription"))
                restart_action.triggered.connect(self.on_restart_transcription_action)
                menu.addSeparator()
            
            rename_action = menu.addAction(_("Rename"))
            rename_action.triggered.connect(self.on_rename_action)
            
            notes_action = menu.addAction(_("Add/Edit Notes"))
            notes_action.triggered.connect(self.on_notes_action)
        
        menu.exec(event.globalPos())

    def save_column_visibility(self):
        self.settings.begin_group(
            Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_VISIBILITY
        )
        for definition in column_definitions:
            self.settings.settings.setValue(
                definition.id, not self.isColumnHidden(definition.column.value)
            )
        self.settings.end_group()

    def on_column_resized(self, logical_index: int, old_size: int, new_size: int):
        """Handle column resize events"""
        self.save_column_widths()

    def on_column_moved(self, logical_index: int, old_visual_index: int, new_visual_index: int):
        """Handle column move events"""
        self.save_column_order()
        # Refresh visibility after column move to ensure it's maintained
        self.load_column_visibility()

    def on_sort_indicator_changed(self, logical_index: int, order: Qt.SortOrder):
        """Handle sort indicator change events"""
        self.save_sort_state()

    def on_double_click(self, index: QModelIndex):
        """Handle double-click events - trigger notes edit for notes column"""
        if index.column() == Column.NOTES.value:
            self.on_notes_action()

    def save_column_widths(self):
        """Save current column widths to settings"""
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_WIDTHS)
        for definition in column_definitions:
            # Only save width if column is visible and has a meaningful width
            if not self.isColumnHidden(definition.column.value):
                width = self.columnWidth(definition.column.value)
                if width > 0:  # Only save if there's a meaningful width
                    self.settings.settings.setValue(definition.id, width)
        self.settings.end_group()

    def save_column_order(self):
        """Save current column order to settings"""
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_ORDER)
        header = self.horizontalHeader()
        for visual_index in range(header.count()):
            logical_index = header.logicalIndex(visual_index)
            # Find the column definition for this logical index
            for definition in column_definitions:
                if definition.column.value == logical_index:
                    self.settings.settings.setValue(definition.id, visual_index)
                    break
        self.settings.end_group()

    def load_column_widths(self):
        """Load saved column widths from settings"""
        # Read every saved width before touching the header. setColumnWidth fires
        # sectionResized -> save_column_widths(), which rewrites *all* widths from
        # the header's current state; reading inside that loop would hand back
        # values the previous iteration had already overwritten with defaults.
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_WIDTHS)
        saved_widths = {
            definition.column.value: self.settings.settings.value(
                definition.id, definition.width
            )
            for definition in column_definitions
            if definition.width is not None
        }
        self.settings.end_group()

        minimum_widths = {
            definition.column.value: definition.min_width
            for definition in column_definitions
        }
        # Blocked so restoring widths is not mistaken for the user dragging them.
        with QSignalBlocker(self.horizontalHeader()):
            for column, saved_width in saved_widths.items():
                if saved_width is None:
                    continue
                # Clamp up: a width saved while the table was squeezed into a
                # narrow window should not permanently elide the column.
                self.setColumnWidth(
                    column,
                    max(minimum_widths.get(column, 120), int(saved_width)),
                )

    def save_sort_state(self):
        """Save current sort state to settings"""
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_SORT_STATE)
        header = self.horizontalHeader()
        self.settings.settings.setValue("column", header.sortIndicatorSection())
        self.settings.settings.setValue("order", header.sortIndicatorOrder().value)
        self.settings.end_group()

    def load_sort_state(self):
        """Load saved sort state from settings"""
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_SORT_STATE)
        column = self.settings.settings.value("column")
        order = self.settings.settings.value("order")
        self.settings.end_group()

        if column is not None and order is not None:
            sort_order = Qt.SortOrder(int(order))
            self.sortByColumn(int(column), sort_order)

    def load_column_visibility(self):
        """Load saved column visibility from settings"""
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_VISIBILITY)
        for definition in column_definitions:
            visible = True
            if definition.hidden_toggleable:
                value = self.settings.settings.value(definition.id, "true")
                visible = value in {"true", "True", True}
            
            self.setColumnHidden(definition.column.value, not visible)
        self.settings.end_group()
        
        # Force a refresh of the table layout
        self.horizontalHeader().update()
        self.viewport().update()
        self.updateGeometry()

    def load_column_order(self):
        """Load saved column order from settings"""
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_ORDER)
        
        # Create a mapping of column IDs to their saved visual positions
        column_positions = {}
        for definition in column_definitions:
            saved_position = self.settings.settings.value(definition.id)
            if saved_position is not None:
                column_positions[definition.column.value] = int(saved_position)
        
        self.settings.end_group()
        
        # Apply the saved order
        if column_positions:
            header = self.horizontalHeader()
            for logical_index, visual_position in column_positions.items():
                if 0 <= visual_position < header.count():
                    header.moveSection(header.visualIndex(logical_index), visual_position)
    
    def reset_column_order(self):
        """Reset column order to default"""

        # Reset column widths to defaults
        for definition in column_definitions:
            if definition.width is not None:
                self.setColumnWidth(definition.column.value, definition.width)

        # Show all columns
        for definition in column_definitions:
            self.setColumnHidden(definition.column.value, False)

        # Restore default column order
        header = self.horizontalHeader()
        # Move each section to its default position in order
        # To avoid index shifting, move from left to right
        for target_visual_index, definition in enumerate(column_definitions):
            logical_index = definition.column.value
            current_visual_index = header.visualIndex(logical_index)
            if current_visual_index != target_visual_index:
                header.moveSection(current_visual_index, target_visual_index)

        # Reset sort to default (TIME_QUEUED descending)
        self.sortByColumn(Column.TIME_QUEUED.value, Qt.SortOrder.DescendingOrder)

        # Clear saved settings
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_ORDER)
        self.settings.settings.remove("")
        self.settings.end_group()

        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_WIDTHS)
        self.settings.settings.remove("")
        self.settings.end_group()

        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_SORT_STATE)
        self.settings.settings.remove("")
        self.settings.end_group()

        # Save the reset state for visibility, widths, and sort
        self.save_column_visibility()
        self.save_column_widths()
        self.save_sort_state()

        # Force a refresh of the table layout
        self.horizontalHeader().update()
        self.viewport().update()
        self.updateGeometry()

    def reload_column_order_from_settings(self):
        """Reload column order, width, and visibility from settings"""

        # --- Load column visibility ---
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_VISIBILITY)
        visibility_settings = {}
        for definition in column_definitions:
            vis = self.settings.settings.value(definition.id)
            if vis is not None:
                visibility_settings[definition.id] = str(vis).lower() not in ("0", "false", "no")
        self.settings.end_group()

        # --- Load column widths ---
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_WIDTHS)
        width_settings = {}
        for definition in column_definitions:
            width = self.settings.settings.value(definition.id)
            if width is not None:
                try:
                    width_settings[definition.id] = int(width)
                except Exception:
                    pass
        self.settings.end_group()

        # --- Load column order ---
        self.settings.begin_group(Settings.Key.TRANSCRIPTION_TASKS_TABLE_COLUMN_ORDER)
        order_settings = {}
        for definition in column_definitions:
            pos = self.settings.settings.value(definition.id)
            if pos is not None:
                try:
                    order_settings[definition.column.value] = int(pos)
                except Exception:
                    pass
        self.settings.end_group()

        # --- Apply visibility, widths, and order ---
        header = self.horizontalHeader()

        # First, set visibility and width for each column. Block the header so
        # each setColumnWidth does not fire sectionResized -> save_column_widths
        # and write half-applied state back over the settings we just read.
        with QSignalBlocker(header):
            for definition in column_definitions:
                is_visible = visibility_settings.get(definition.id, True)
                width = width_settings.get(definition.id, definition.width)
                self.setColumnHidden(definition.column.value, not is_visible)
                if width is not None:
                    self.setColumnWidth(
                        definition.column.value, max(width, definition.min_width)
                    )

        # Then, apply column order
        # Build a list of (logical_index, visual_position) for ALL columns (including hidden ones)
        all_columns = [
            (definition.column.value, order_settings.get(definition.column.value, idx))
            for idx, definition in enumerate(column_definitions)
        ]
        # Sort by saved visual position
        all_columns.sort(key=lambda x: x[1])

        # Move sections to match the saved order
        for target_visual, (logical_index, saved_visual) in enumerate(all_columns):
            current_visual = header.visualIndex(logical_index)
            if current_visual != target_visual:
                header.moveSection(current_visual, target_visual)

    def copy_selected_fields(self):
        selected_text = ""
        for row in self.selectionModel().selectedRows():
            row_index = row.row()
            file_name = self.model().data(self.model().index(row_index, Column.FILE.value))
            url = self.model().data(self.model().index(row_index, Column.URL.value))

            selected_text += f"{file_name}{url}\n"

        selected_text = selected_text.rstrip("\n")
        QApplication.clipboard().setText(selected_text)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Override double-click to prevent default behavior when clicking on notes column"""
        index = self.indexAt(event.pos())
        if index.isValid() and index.column() == Column.NOTES.value:
            # Handle our custom double-click action without triggering default behavior
            self.on_double_click(index)
            event.accept()
        else:
            # For other columns, use default behavior
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Return:
            self.return_clicked.emit()

        if event.key() == Qt.Key.Key_Delete:
            if self.selectionModel().selectedRows():
                self.delete_requested.emit()
            return

        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selected_fields()
            return

        super().keyPressEvent(event)

    def selected_transcriptions(self) -> List[Transcription]:
        selected = self.selectionModel().selectedRows()
        return [self.transcription(row) for row in selected]

    def delete_transcriptions(self, rows: List[QModelIndex]):
        for row in rows:
            self.model().removeRow(row.row())
        self.model().submitAll()

    def transcription(self, index: QModelIndex) -> Transcription:
        return Transcription.from_record(self.model().record(index.row()))

    def refresh_all(self):
        self.model().select()

    @staticmethod
    def _escape_like(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("'", "''")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    def set_filter(self, text: str = "", status: str = ""):
        """Filter the existing SQL model without adding a second data model."""
        conditions = []
        search = self._escape_like(text.strip().lower())
        if search:
            fields = (
                "name",
                "file",
                "url",
                "model_type",
                "task",
                "status",
                "language",
                "notes",
                "error_message",
            )
            conditions.append(
                "(" + " OR ".join(
                    f"LOWER(COALESCE({field}, '')) LIKE '%{search}%' ESCAPE '\\'"
                    for field in fields
                ) + ")"
            )

        status = status.strip()
        if status:
            conditions.append("status = '" + status.replace("'", "''") + "'")

        self.model().setFilter(" AND ".join(conditions))
        self.model().select()

    def refresh_row(self, id: UUID):
        for i in range(self.model().rowCount()):
            record = self.model().record(i)
            if record.value("id") == str(id):
                self.model().selectRow(i)
                return

    @staticmethod
    def format_timedelta(delta: timedelta):
        mm, ss = divmod(delta.seconds, 60)
        result = _("{}s").format(ss)
        if mm == 0:
            return result
        hh, mm = divmod(mm, 60)
        result = _("{}m {}").format(mm, result)
        if hh == 0:
            return result
        return _("{}h {}").format(hh, result)

    def on_rename_action(self):
        selected_rows = self.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Get the first selected transcription
        transcription = self.transcription(selected_rows[0])
        
        # Get current name or fallback to file name
        current_name = transcription.name or (
            transcription.url if transcription.url 
            else os.path.basename(transcription.file) if transcription.file 
            else ""
        )
        
        # Show input dialog
        from PyQt6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, 
            _("Rename Transcription"), 
            _("Enter new name:"), 
            text=current_name
        )
        
        if ok and new_name.strip():
            # Update the transcription name
            from uuid import UUID
            self.transcription_service.update_transcription_name(
                UUID(transcription.id), 
                new_name.strip()
            )
            self.refresh_all()

    def on_notes_action(self):
        selected_rows = self.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Get the first selected transcription
        transcription = self.transcription(selected_rows[0])
        
        # Show input dialog for notes
        from PyQt6.QtWidgets import QInputDialog
        current_notes = transcription.notes or ""
        new_notes, ok = QInputDialog.getMultiLineText(
            self, 
            _("Notes"),
            _("Enter some relevant notes for this transcription:"),
            text=current_notes
        )
        
        if ok:
            # Update the transcription notes
            from uuid import UUID
            self.transcription_service.update_transcription_notes(
                UUID(transcription.id), 
                new_notes
            )
            self.refresh_all()

    def on_restart_transcription_action(self):
        """Restart transcription for failed or canceled tasks"""
        selected_rows = self.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Get the first selected transcription
        transcription = self.transcription(selected_rows[0])
        
        # Check if the task can be restarted
        if transcription.status not in ["failed", "canceled", "skipped"]:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                _("Cannot Restart"),
                _("Only failed, canceled, or skipped transcriptions can be restarted.")
            )
            return
        
        try:
            self.transcription_service.reset_transcription_for_restart(UUID(transcription.id))
            transcription.progress = 0.0
            transcription.download_progress = 0.0
            transcription.segment_checkpoint_json = None
            self._restart_transcription_task(transcription)
            self.refresh_all()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                _("Error"),
                _("Failed to restart transcription: {}").format(str(e))
            )
    
    def _restart_transcription_task(self, transcription):
        """Create a new FileTranscriptionTask and add it to the queue worker"""
        from buzz.store.keyring_store import Key, get_password
        from buzz.transcriber.transcriber import (
            FileTranscriptionTask,
            deserialize_segment_checkpoint,
            deserialize_task_metadata,
            deserialize_task_options,
        )

        transcription_options, file_transcription_options = deserialize_task_options(
            getattr(transcription, "task_options_json", None),
            fallback=transcription,
            openai_access_token=get_password(Key.OPENAI_API_KEY),
        )
        
        # Get the model path from the transcription options
        model_path = transcription_options.model.get_local_model_path()
        if model_path is None:
            # If model is not available locally, we need to download it
            from buzz.model_loader import ModelDownloader
            ModelDownloader(model=transcription_options.model).run()
            model_path = transcription_options.model.get_local_model_path()
        
        if model_path is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, 
                _("Error"), 
                _("Could not restart transcription: model not available and could not be downloaded.")
            )
            return

        checkpoint_segments, checkpoint_next_chunk = deserialize_segment_checkpoint(
            getattr(transcription, "segment_checkpoint_json", None)
        )
        task_metadata = deserialize_task_metadata(
            getattr(transcription, "task_options_json", None)
        )
        source_value = transcription.source or task_metadata.get("source")
        try:
            source = FileTranscriptionTask.Source(source_value)
        except (TypeError, ValueError):
            source = FileTranscriptionTask.Source.FILE_IMPORT

        # Create the new task
        task = FileTranscriptionTask(
            transcription_options=transcription_options,
            file_transcription_options=file_transcription_options,
            model_path=model_path,
            file_path=transcription.file or task_metadata.get("file_path"),
            original_file_path=task_metadata.get("original_file_path"),
            delete_source_file=bool(task_metadata.get("delete_source_file")),
            url=transcription.url or task_metadata.get("url"),
            output_directory=(
                transcription.output_folder or task_metadata.get("output_directory")
            ),
            source=source,
            uid=UUID(transcription.id),
            segments=checkpoint_segments,
            fraction_completed=getattr(transcription, "progress", 0.0) or 0.0,
            fraction_downloaded=getattr(transcription, "download_progress", 0.0) or 0.0,
            checkpoint_next_chunk=checkpoint_next_chunk,
        )
        
        # Add the task to the queue worker
        # We need to access the main window's transcriber worker
        # This is a bit of a hack, but it's the cleanest way given the current architecture
        main_window = self.parent()
        while main_window and not hasattr(main_window, 'transcriber_worker'):
            main_window = main_window.parent()
        
        if main_window and hasattr(main_window, 'transcriber_worker'):
            main_window.transcriber_worker.add_task(task)
        else:
            # Fallback: show error if we can't find the transcriber worker
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, 
                _("Error"), 
                _("Could not restart transcription: transcriber worker not found.")
            )

import html
from typing import Callable

from PyQt6.QtCore import QEvent
from PyQt6.QtSql import QSqlRecord, QSqlTableModel
from PyQt6.QtWidgets import QStyledItemDelegate, QToolTip

# Above this length a plain tooltip becomes one unreadable long line, so wrap it
# in rich text to let Qt word-wrap it.
TOOLTIP_WRAP_THRESHOLD = 80


class RecordDelegate(QStyledItemDelegate):
    def __init__(self, text_getter: Callable[[QSqlRecord], str]):
        super().__init__()
        self.callback = text_getter

    def cell_text(self, index) -> str:
        model: QSqlTableModel = index.model()
        return self.callback(model.record(index.row()))

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.text = self.cell_text(index)

    def helpEvent(self, event, view, option, index):
        """Show the full cell text on hover. Cells are elided when the column is
        narrower than its content, so without this the tail is unreachable
        without resizing the column."""
        if event is None or event.type() != QEvent.Type.ToolTip:
            return super().helpEvent(event, view, option, index)

        text = self.cell_text(index)
        if not text:
            QToolTip.hideText()
            return False

        if len(text) > TOOLTIP_WRAP_THRESHOLD:
            text = f"<div style='white-space: pre-wrap'>{html.escape(text)}</div>"

        QToolTip.showText(event.globalPos(), text, view)
        return True

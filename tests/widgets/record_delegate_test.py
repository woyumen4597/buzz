from unittest.mock import MagicMock, patch

from PyQt6.QtCore import QEvent, QPoint

from buzz.widgets.record_delegate import RecordDelegate, TOOLTIP_WRAP_THRESHOLD


def _index(row=0):
    index = MagicMock()
    index.row.return_value = row
    index.model.return_value.record.return_value = MagicMock()
    return index


def _tooltip_event():
    event = MagicMock()
    event.type.return_value = QEvent.Type.ToolTip
    event.globalPos.return_value = QPoint(10, 20)
    return event


class TestRecordDelegateTooltip:
    def test_shows_full_text_on_hover(self):
        delegate = RecordDelegate(text_getter=lambda record: "Failed (boom)")

        with patch("buzz.widgets.record_delegate.QToolTip") as tooltip:
            handled = delegate.helpEvent(
                _tooltip_event(), MagicMock(), MagicMock(), _index()
            )

        assert handled is True
        shown_text = tooltip.showText.call_args[0][1]
        assert shown_text == "Failed (boom)"

    def test_long_text_is_wrapped_and_escaped(self):
        # A long single-line tooltip is unreadable, and error messages can carry
        # characters Qt would otherwise treat as markup.
        long_text = "Failed (<b>" + "x" * TOOLTIP_WRAP_THRESHOLD + ")"
        delegate = RecordDelegate(text_getter=lambda record: long_text)

        with patch("buzz.widgets.record_delegate.QToolTip") as tooltip:
            delegate.helpEvent(_tooltip_event(), MagicMock(), MagicMock(), _index())

        shown_text = tooltip.showText.call_args[0][1]
        assert shown_text.startswith("<div")
        assert "&lt;b&gt;" in shown_text
        assert "<b>" not in shown_text

    def test_empty_cell_hides_tooltip(self):
        delegate = RecordDelegate(text_getter=lambda record: "")

        with patch("buzz.widgets.record_delegate.QToolTip") as tooltip:
            handled = delegate.helpEvent(
                _tooltip_event(), MagicMock(), MagicMock(), _index()
            )

        assert handled is False
        tooltip.showText.assert_not_called()
        tooltip.hideText.assert_called_once()

    def test_non_tooltip_events_are_not_intercepted(self):
        delegate = RecordDelegate(text_getter=lambda record: "text")
        event = MagicMock()
        event.type.return_value = QEvent.Type.MouseMove

        with patch("buzz.widgets.record_delegate.QToolTip") as tooltip:
            with patch.object(
                RecordDelegate.__bases__[0], "helpEvent", return_value=False
            ) as base_help_event:
                delegate.helpEvent(event, MagicMock(), MagicMock(), _index())

        tooltip.showText.assert_not_called()
        base_help_event.assert_called_once()

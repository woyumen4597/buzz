from pathlib import Path

from PyQt6.QtWidgets import QCheckBox, QSpinBox, QWidget

from buzz.widgets.highlight_candidates_widget import HighlightCandidatesWidget
from buzz.widgets.main_window import MainWindow


def test_main_window_has_highlight_tab(qtbot, transcription_service):
    window = MainWindow(transcription_service)
    qtbot.add_widget(window)
    assert window.workspace_tabs.count() == 2
    assert window.workspace_tabs.tabText(0) == "转录"
    assert window.workspace_tabs.tabText(1) == "视频高光"
    assert window.workspace_tabs.widget(1).__class__ is HighlightCandidatesWidget
    window.close()


def test_highlight_widget_builds_cli_arguments(qtbot, tmp_path):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    assert widget.windowTitle() == ""
    assert widget.run_button.text() == "生成高光合集"
    assert widget.open_button.text() == "打开结果"
    assert widget.video_input.placeholderText() == "请选择视频文件"
    assert widget.target_ratio.value() == 0.3
    assert widget.target_ratio.minimum() == 0.0
    assert widget.target_ratio.maximum() == 1.0
    assert widget.findChildren(QSpinBox) == []
    assert widget.findChildren(QCheckBox) == []
    form_container = widget.findChild(QWidget, "HighlightFormContainer")
    assert form_container is not None
    assert form_container.maximumWidth() == 820
    video = tmp_path / "video file.mp4"
    video.write_bytes(b"video")
    widget.video_input.setText(str(video))
    widget.srt_input.setText(str(tmp_path / "captions.srt"))
    widget.target_ratio.setValue(0.5)

    args = widget._build_args()
    assert args.video == Path(video)
    assert args.srt == tmp_path / "captions.srt"
    assert args.output_dir is None
    assert args.window_seconds == 20.0
    assert args.stride_seconds == 10.0
    assert args.max_candidates == 0
    assert args.target_ratio == 0.5
    assert args.max_auto_clips == 0
    assert args.no_scene_detection is False
    assert args.keep_static_scenes is False
    # The GUI has no CLI flags, so verification must be requested explicitly
    # rather than relying on the CLI default.
    assert args.verify == "fast"


def test_highlight_widget_surfaces_verification_warnings(qtbot, tmp_path):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    output = tmp_path / "video_highlight.mp4"
    output.write_bytes(b"video")
    report = tmp_path / "video_highlights" / "verification.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "[ok  ] video_stream: video: h264 320x240\n"
        "[WARN] audio_silence: 2 silent stretch(es)\n",
        encoding="utf-8",
    )
    widget._output_dir = output
    assert "1" in widget._verification_note()


def test_highlight_widget_surfaces_verification_failures(qtbot, tmp_path):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    output = tmp_path / "video_highlight.mp4"
    output.write_bytes(b"video")
    report = tmp_path / "video_highlights" / "verification.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "[FAIL] subtitle_bounds: cues end past the reel duration\n",
        encoding="utf-8",
    )
    widget._output_dir = output
    note = widget._verification_note()
    assert "cues end past the reel duration" in note


def test_highlight_widget_verification_note_is_empty_without_a_report(qtbot, tmp_path):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    output = tmp_path / "video_highlight.mp4"
    output.write_bytes(b"video")
    widget._output_dir = output
    assert widget._verification_note() == ""


def test_highlight_widget_has_a_manual_check_button(qtbot):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    # Disabled until a reel exists, since there is nothing to check yet.
    assert widget.verify_button.isEnabled() is False
    assert widget.verify_button.text() == "检查视频"


def test_highlight_widget_enables_check_button_after_generation(qtbot, tmp_path):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    reel = tmp_path / "video_highlight.mp4"
    reel.write_bytes(b"video")
    widget._generation_finished(str(reel))
    assert widget.verify_button.isEnabled() is True


def test_highlight_widget_check_button_ignores_a_missing_reel(qtbot):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    # No reel at all: clicking must be a no-op rather than an error.
    widget.verify_result()
    assert widget._verify_thread is None


def test_highlight_widget_rejects_missing_video(qtbot):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    widget.start_generation()
    assert "存在的视频文件" in widget.status_label.text()

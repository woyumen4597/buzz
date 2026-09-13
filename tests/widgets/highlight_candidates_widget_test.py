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
    assert args.no_previews is False
    assert args.keep_static_scenes is False


def test_highlight_widget_rejects_missing_video(qtbot):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    widget.start_generation()
    assert "存在的视频文件" in widget.status_label.text()

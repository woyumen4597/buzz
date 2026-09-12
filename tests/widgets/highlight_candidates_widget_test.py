from pathlib import Path

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
    assert widget.run_button.text() == "生成候选"
    assert widget.open_button.text() == "打开结果"
    assert widget.video_input.placeholderText() == "请选择视频文件"
    video = tmp_path / "video file.mp4"
    video.write_bytes(b"video")
    widget.video_input.setText(str(video))
    widget.srt_input.setText(str(tmp_path / "captions.srt"))
    widget.output_input.setText(str(tmp_path / "results"))
    widget.window_seconds.setValue(30)
    widget.stride_seconds.setValue(15)
    widget.max_candidates.setValue(50)
    widget.no_scene_detection.setChecked(True)
    widget.no_previews.setChecked(True)

    args = widget._build_args()
    assert args.video == Path(video)
    assert args.srt == tmp_path / "captions.srt"
    assert args.output_dir == tmp_path / "results"
    assert args.window_seconds == 30
    assert args.stride_seconds == 15
    assert args.max_candidates == 50
    assert args.no_scene_detection is True
    assert args.no_previews is True


def test_highlight_widget_rejects_missing_video(qtbot):
    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    widget.start_generation()
    assert "存在的视频文件" in widget.status_label.text()

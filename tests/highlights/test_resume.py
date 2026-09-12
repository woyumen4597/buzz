from types import SimpleNamespace

from buzz.highlights.exporter import load_checkpoint, write_checkpoint
from buzz.highlights.models import Candidate, HighlightConfig, VideoInfo


def test_checkpoint_round_trip(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    candidate = Candidate("candidate-0001", 0, 5000, 0, 5000, thumbnail="thumbnails/candidate-0001.jpg")
    config = HighlightConfig(no_previews=True)
    write_checkpoint(tmp_path, str(video_path), VideoInfo(5000), config, [candidate])
    checkpoint = load_checkpoint(tmp_path)
    assert checkpoint is not None
    assert checkpoint["completed_ids"] == [candidate.id]
    assert checkpoint["input"]["size"] == 5


def test_resume_args_include_keep_existing(qtbot):
    from buzz.widgets.highlight_candidates_widget import HighlightCandidatesWidget

    widget = HighlightCandidatesWidget()
    qtbot.add_widget(widget)
    assert widget._build_args().keep_existing is True

import json

from buzz.highlights.exporter import export_outputs
from buzz.highlights.models import Candidate, HighlightConfig, VideoInfo


def test_export_outputs(tmp_path):
    candidate = Candidate("candidate-0001", 0, 5000, 0, 5000, transcript='<script>&', selected=True, status="keep")
    export_outputs(tmp_path, [candidate], VideoInfo(10_000), HighlightConfig(no_previews=True), "/tmp/video name.mp4")
    assert (tmp_path / "index.html").exists()
    selected = json.loads((tmp_path / "selected.json").read_text())
    assert selected[0]["id"] == candidate.id
    html = (tmp_path / "index.html").read_text()
    assert "<script>&" not in html
    assert "video name.mp4" in html
    assert "-ss 0.000" in (tmp_path / "clips.txt").read_text()

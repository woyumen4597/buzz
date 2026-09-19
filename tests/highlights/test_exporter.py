import json

from buzz.highlights.exporter import _verify_reel_summary, export_outputs
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


def test_export_html_render_handler_reports_verification(tmp_path):
    candidate = Candidate("candidate-0001", 0, 5000, 0, 5000, selected=True, status="keep")
    export_outputs(
        tmp_path, [candidate], VideoInfo(10_000), HighlightConfig(no_previews=True),
        "/tmp/video.mp4", render_endpoint="http://127.0.0.1:1/render",
    )
    html = (tmp_path / "index.html").read_text()
    # The manual render path must surface the check result on the page.
    assert "result.verification" in html


def test_verify_reel_summary_is_best_effort_for_a_missing_file(tmp_path):
    # A missing reel must not raise: verification never breaks a render.
    summary = _verify_reel_summary(tmp_path / "missing.mp4")
    assert summary["ok"] is False
    assert summary["summary"]
    assert summary["details"]

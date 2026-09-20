from buzz.highlights.cli import build_parser


def test_cli_target_ratio_defaults_to_point_three():
    args = build_parser().parse_args(["video.mp4"])
    assert args.target_ratio == 0.3
    assert args.target_duration is None


def test_cli_target_ratio_accepts_fraction():
    args = build_parser().parse_args(["video.mp4", "--target-ratio", "0.5"])
    assert args.target_ratio == 0.5


def test_cli_keeps_legacy_target_duration_option():
    args = build_parser().parse_args(["video.mp4", "--target-duration", "600"])
    assert args.target_duration == 600


def test_cli_verify_defaults_to_fast():
    args = build_parser().parse_args(["video.mp4"])
    assert args.verify == "fast"


def test_cli_verify_accepts_modes():
    assert build_parser().parse_args(["video.mp4", "--verify", "off"]).verify == "off"
    assert build_parser().parse_args(["video.mp4", "--verify", "full"]).verify == "full"


def test_cli_rejects_an_unknown_verify_mode():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args(["video.mp4", "--verify", "sometimes"])


def test_verification_report_is_written_inside_the_work_directory(tmp_path):
    """The report must not land beside the reel in the user's source folder.

    A one-click run removes its work directory but the reel itself sits next to
    the source video, so writing the report to the reel's parent left a stray
    file in the library that no cleanup step owned.
    """
    from buzz.highlights.output import (
        automatic_output_path,
        verification_report_path,
        work_dir_for,
    )

    source = tmp_path / "movie.mp4"
    reel = automatic_output_path(source)
    report = verification_report_path(reel)

    assert report == work_dir_for(source) / "verification.txt"
    assert report.parent != reel.parent


def test_verification_report_path_handles_a_renamed_reel(tmp_path):
    from buzz.highlights.output import verification_report_path

    # A reel that does not follow the automatic naming keeps its own stem.
    assert (
        verification_report_path(tmp_path / "custom.mp4").parent
        == tmp_path / "custom_highlights"
    )

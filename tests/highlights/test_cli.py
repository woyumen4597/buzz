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

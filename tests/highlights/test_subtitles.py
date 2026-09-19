from buzz.highlights.models import Candidate
from buzz.highlights.subtitles import (
    Subtitle,
    associate_subtitles,
    clip_subtitles,
    format_srt,
    parse_srt,
)


def test_parse_multiline_chinese_srt():
    cues, warnings = parse_srt("""1\n00:00:01,000 --> 00:00:03,000\n你好，\n世界\n\n2\n00:00:03.500 --> 00:00:04.000\n结束\n""")
    assert not warnings
    assert [cue.text for cue in cues] == ["你好， 世界", "结束"]


def test_invalid_srt_record_is_warning():
    cues, warnings = parse_srt("1\nbad --> time\ntext")
    assert not cues
    assert warnings


def test_associate_uses_overlap_only():
    candidate = Candidate("x", 2000, 3000, 0, 4000)
    cues, _ = parse_srt("1\n00:00:01,000 --> 00:00:02,000\nno\n\n2\n00:00:02,500 --> 00:00:04,000\nyes")
    assert associate_subtitles(candidate, cues) == "yes"
    assert "transcript_density" in candidate.reasons


def test_associate_rescues_dialogue_from_static_scene_ignore():
    candidate = Candidate("x", 2000, 5000, 0, 5000, status="ignore", reasons=["static_scene"])
    cues, _ = parse_srt("1\n00:00:02,000 --> 00:00:04,000\n重要内容")
    associate_subtitles(candidate, cues)
    assert candidate.status == "unprocessed"
    assert "dialogue_rescue" in candidate.reasons


def test_clip_subtitles_trims_and_rebases_to_clip_origin():
    cues = [Subtitle(1_000, 2_000, "a"), Subtitle(12_000, 16_000, "b")]
    clipped = clip_subtitles(cues, 10_000, 20_000)
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in clipped] == [
        (2_000, 6_000, "b"),
    ]


def test_clip_subtitles_truncates_cue_spanning_the_boundary():
    clipped = clip_subtitles([Subtitle(9_000, 12_000, "x")], 10_000, 20_000)
    assert [(cue.start_ms, cue.end_ms) for cue in clipped] == [(0, 2_000)]


def test_clip_subtitles_drops_non_overlapping_cues():
    assert clip_subtitles([Subtitle(0, 1_000, "x")], 10_000, 20_000) == []


def test_format_srt_numbers_cues_and_uses_comma_separator():
    text = format_srt([Subtitle(0, 1_000, "first"), Subtitle(2_000, 3_500, "second")])
    assert text == (
        "1\n00:00:00,000 --> 00:00:01,000\nfirst\n\n"
        "2\n00:00:02,000 --> 00:00:03,500\nsecond\n"
    )


def test_clipped_srt_round_trips_through_the_parser():
    cues = [Subtitle(1_000, 2_000, "a"), Subtitle(12_000, 16_000, "b")]
    reparsed, warnings = parse_srt(format_srt(clip_subtitles(cues, 10_000, 20_000)))
    assert not warnings
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in reparsed] == [
        (2_000, 6_000, "b"),
    ]

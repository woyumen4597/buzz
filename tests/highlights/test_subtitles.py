from buzz.highlights.models import Candidate
from buzz.highlights.subtitles import associate_subtitles, parse_srt


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

from buzz.transcriber.hallucinations import (
    drop_hallucinated_segments,
    normalize_text,
)
from buzz.transcriber.transcriber import Segment


def _seg(text, start=0, end=2000):
    return Segment(start=start, end=end, text=text, translation="")


def _texts(segments):
    return [segment.text for segment in segments]


class TestNormalizeText:
    def test_ignores_spacing_and_punctuation(self):
        assert normalize_text(" ご視聴ありがとうございました。 ") == normalize_text(
            "ご視聴ありがとうございました"
        )
        assert normalize_text("Thanks for watching!") == normalize_text(
            "thanks for watching"
        )

    def test_handles_empty_text(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""


class TestDropHallucinatedSegments:
    def test_drops_a_signoff_seen_only_once(self):
        # Whisper emits this over silence even when the audio says it nowhere.
        segments = [
            _seg("ご視聴ありがとうございました", 0, 29980),
            _seg("ゆうきです。", 30000, 32000),
        ]

        assert _texts(drop_hallucinated_segments(segments)) == ["ゆうきです。"]

    def test_drops_punctuated_and_spaced_variants(self):
        segments = [
            _seg(" ご視聴ありがとうございました。 ", 0, 29980),
            _seg("感谢您的观看！", 30000, 32000),
            _seg("Thanks for watching!", 32000, 34000),
        ]

        assert drop_hallucinated_segments(segments) == []

    def test_drops_subtitle_credits_carrying_a_name(self):
        segments = [
            _seg("字幕由 某某字幕组 提供", 0, 2000),
            _seg("Subtitles by John Doe", 2000, 4000),
        ]

        assert drop_hallucinated_segments(segments) == []

    def test_keeps_an_ambiguous_greeting_said_once(self):
        # A character going to bed must survive; only repetition marks it as an
        # artifact, because that is what Whisper does over silence.
        segments = [_seg("おやすみなさい。", 0, 2000), _seg("また明日。", 2000, 4000)]

        assert _texts(drop_hallucinated_segments(segments)) == [
            "おやすみなさい。",
            "また明日。",
        ]

    def test_drops_an_ambiguous_greeting_when_it_repeats(self):
        segments = [
            _seg("おやすみなさい。", 0, 2000),
            _seg("はい。", 2000, 4000),
            _seg("おやすみなさい。", 4000, 6000),
            _seg("おやすみなさい", 6000, 8000),
        ]

        assert _texts(drop_hallucinated_segments(segments)) == ["はい。"]

    def test_drops_the_chinese_translations_of_those_artifacts(self):
        segments = [
            _seg("欢迎收看", 0, 2000),
            _seg("晚安。", 2000, 4000),
            _seg("晚安", 4000, 6000),
        ]

        assert drop_hallucinated_segments(segments) == []

    def test_drops_a_repeated_cue_spanning_a_decode_window(self):
        # An invented sentence replayed for a whole 30 s window: repetition plus
        # the window-length cue is the signature.
        segments = [
            _seg("チョコレートを作ることができます。", 0, 29980),
            _seg("はい", 30000, 31000),
            _seg("チョコレートを作ることができます。", 60000, 89980),
        ]

        assert _texts(drop_hallucinated_segments(segments)) == ["はい"]

    def test_keeps_a_long_cue_that_does_not_repeat(self):
        # A genuinely long stretch of speech is not an artifact on its own.
        segments = [_seg("長い説明をここでしています。", 0, 30000)]

        assert _texts(drop_hallucinated_segments(segments)) == [
            "長い説明をここでしています。"
        ]

    def test_keeps_repeated_short_dialogue(self):
        # Interjections repeat constantly in real speech and are never artifacts
        # by themselves.
        segments = [
            _seg("はい", 0, 1000),
            _seg("ん", 1000, 2000),
            _seg("はい", 2000, 3000),
            _seg("ちょっとやめてください", 3000, 5000),
            _seg("ちょっとやめてください", 5000, 7000),
        ]

        assert _texts(drop_hallucinated_segments(segments)) == [
            "はい",
            "ん",
            "はい",
            "ちょっとやめてください",
            "ちょっとやめてください",
        ]

    def test_keeps_segment_order(self):
        segments = [
            _seg("ゆうきさんですよね。", 0, 1000),
            _seg("ご視聴ありがとうございました", 1000, 30980),
            _seg("お願いします。", 31000, 32000),
        ]

        kept = drop_hallucinated_segments(segments)

        assert [segment.start for segment in kept] == [0, 31000]

    def test_returns_an_empty_list_when_everything_is_an_artifact(self):
        segments = [_seg("ご視聴ありがとうございました", 0, 29980)]

        assert drop_hallucinated_segments(segments) == []

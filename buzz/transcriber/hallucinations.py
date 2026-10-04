"""Remove cues Whisper invents over silence.

Whisper does not return nothing for a decode window that holds no speech: it
falls back on whatever its training data made likely there. That is a small,
well-known set of sign-offs and credits ("ご視聴ありがとうございました",
"Thank you for watching", "感谢您的观看") and, less often, an invented sentence
replayed for the whole window. None of it is in the audio, so none of it belongs
in a subtitle.

Recognising the artifacts from the transcript alone covers every backend
(whisper.cpp, faster-whisper, Whisper, the OpenAI API) and every output path,
without decoding the audio a second time. The filter is deliberately narrow:
it only rejects text that is on the artifact list, or a cue that spans a whole
decode window *and* repeats, so ordinary dialogue -- including a character
saying good night once -- is left alone.
"""

import logging
import re
from collections import Counter
from typing import List, Sequence

from buzz.transcriber.transcriber import Segment

LOG = logging.getLogger(__name__)

#: Spacing and punctuation carry no signal here and Whisper re-punctuates the
#: same artifact differently between runs, so both sides are compared with them
#: removed.
_NOISE_RE = re.compile(
    r"[\s\u3000。、．，,.!！?？…‥「」『』（）()\[\]【】｛｝{}<>《》\"'“”‘’·・:：;；—–~～\-_*#]+"
)


def normalize_text(text: str) -> str:
    """Fold a cue's text to the form the artifact lists are written in."""
    return _NOISE_RE.sub("", str(text or "")).casefold()


def _normalized(*phrases: str) -> frozenset:
    return frozenset(normalize_text(phrase) for phrase in phrases)


#: Text Whisper emits from its training data when a window holds no speech.
#: These are station sign-offs and subtitle credits: they are never dialogue, so
#: a single occurrence is already an artifact.
_ALWAYS_HALLUCINATED = _normalized(
    # Japanese
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご視聴いただきありがとうございました",
    "ご視聴いただきありがとうございます",
    "最後までご視聴いただきありがとうございます",
    "ご覧いただきありがとうございます",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    "高評価とチャンネル登録お願いします",
    # Chinese
    "欢迎收看",
    "感谢收看",
    "谢谢收看",
    "感谢观看",
    "谢谢观看",
    "感谢您的观看",
    "谢谢您的观看",
    "请不吝点赞订阅转发打赏支持明镜与点点栏目",
    "请订阅",
    "请点赞",
    "点赞订阅",
    "点赞订阅转发",
    "订阅频道",
    "订阅我们的频道",
    "关注我们的频道",
    # English
    "Thank you for watching",
    "Thanks for watching",
    "Thank you for watching this video",
    "Please subscribe",
    "Like and subscribe",
    "Subscribe to my channel",
    "Don't forget to subscribe",
    "Amara.org",
    # Korean
    "시청해주셔서 감사합니다",
    "구독과 좋아요 부탁드립니다",
    # Russian
    "Спасибо за просмотр",
    "Продолжение следует",
)

#: Prefixes of artifacts that carry a credit after them ("字幕由 ... 提供",
#: "Subtitles by ..."), so an exact match cannot catch them.
_HALLUCINATION_PREFIXES = tuple(
    normalize_text(phrase)
    for phrase in (
        "字幕",
        "subtitles by",
        "subtitle by",
        "subtitled by",
        "transcription by",
        "translated by",
        "synced by",
        "редактор субтитров",
    )
)

#: Phrases that are also ordinary dialogue. Whisper repeats them during
#: silence, so they are rejected only when the same text occurs more than once
#: in one transcription: a character saying good night once is kept.
_WHEN_REPEATED = _normalized(
    # Japanese
    "おやすみなさい",
    "おやすみ",
    "お疲れ様でした",
    "お疲れ様です",
    "次回お楽しみに",
    # Chinese
    "晚安",
    "再见",
    "大家好",
    "谢谢大家",
    # English
    "Good night",
    "Good night everyone",
    "Bye bye",
    "Goodbye",
    "See you next time",
)

#: Whisper splits a transcription at sentence boundaries, so a single cue that
#: spans a whole 30 s decode window is not dialogue: it is the window itself.
#: The shorter of the two decode lengths whisper.cpp uses is 25 s, and the
#: observed durations cluster at 29.98 s.
_DECODE_WINDOW_MS = 25_000


def _is_hallucination(text: str, occurrences: int, duration_ms: int) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if normalized in _ALWAYS_HALLUCINATED:
        return True
    if normalized.startswith(_HALLUCINATION_PREFIXES):
        return True
    if occurrences > 1:
        if normalized in _WHEN_REPEATED:
            return True
        # An invented sentence replayed window after window: only repetition
        # distinguishes it from a genuinely long (25 s+) stretch of speech.
        if duration_ms >= _DECODE_WINDOW_MS:
            return True
    return False


def drop_hallucinated_segments(segments: Sequence[Segment]) -> List[Segment]:
    """Return ``segments`` without the cues Whisper invented over silence.

    Repetition is counted per transcription, so callers must pass the whole
    recognized result rather than one file's worth of a batch at a time.
    """
    counts = Counter(normalize_text(segment.text) for segment in segments)
    kept: List[Segment] = []
    dropped: List[Segment] = []
    for segment in segments:
        normalized = normalize_text(segment.text)
        duration_ms = (segment.end or 0) - (segment.start or 0)
        if _is_hallucination(segment.text, counts[normalized], duration_ms):
            dropped.append(segment)
            continue
        kept.append(segment)

    if dropped:
        LOG.warning(
            "Dropped %d of %d segment(s) that Whisper invented over silence, "
            "e.g. %r",
            len(dropped),
            len(segments),
            dropped[0].text,
        )
        for segment in dropped[:10]:
            LOG.debug(
                "Hallucinated segment at %s-%sms: %r",
                segment.start,
                segment.end,
                segment.text,
            )
    return kept

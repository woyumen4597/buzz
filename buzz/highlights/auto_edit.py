"""Automatic highlight selection from scored candidates."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable

from .models import Candidate, HighlightConfig, interval_iou


ALGORITHM_VERSION = "weighted-interval-budget-v6"
_BUDGET_QUANTUM_MS = 1000
# Upper bound on DP table cells. Long sources are handled by coarsening the
# budget quantum instead of growing the table, which keeps memory flat.
_MAX_DP_STATES = 4_000_000


@dataclass(frozen=True)
class AutoSelection:
    selected: list[Candidate]
    #: Target reel duration derived from the source length and ratio.
    budget_ms: int
    #: Summed duration of the clips that were actually selected.
    total_duration_ms: int


def _candidate_key(candidate: Candidate) -> tuple[int, int, str]:
    return candidate.start_ms, candidate.end_ms, candidate.id


def resolve_target_duration_seconds(
    video_duration_ms: int,
    configured_seconds: float | None = None,
    *,
    configured_ratio: float | None = None,
) -> float:
    """Resolve the target, preserving the old seconds API and adding ratios."""
    source_seconds = max(0.0, video_duration_ms / 1000)
    if configured_ratio is None:
        # Legacy positional/keyword calls use seconds; zero means automatic.
        if configured_seconds is not None and configured_seconds > 0:
            return min(configured_seconds, source_seconds)
        return source_seconds * 0.3
    if configured_seconds is not None and configured_seconds > 0:
        return min(configured_seconds, source_seconds)
    ratio = max(0.0, min(1.0, configured_ratio))
    return min(source_seconds, source_seconds * ratio)


def _quantum_ms(budget_ms: int, rows: int, layers: int) -> int:
    """Pick a budget quantum that keeps the DP table within a fixed budget.

    A multi-hour source at the default ratio can request a budget of thousands
    of seconds. One-second quanta would then create a table with millions of
    cells per row. Coarsening the quantum trades a little precision for a table
    that always fits comfortably in memory.
    """
    if budget_ms <= 0:
        return _BUDGET_QUANTUM_MS
    max_units = max(1, _MAX_DP_STATES // (max(1, rows) * max(1, layers)) - 1)
    if max_units >= budget_ms // _BUDGET_QUANTUM_MS:
        return _BUDGET_QUANTUM_MS
    return max(_BUDGET_QUANTUM_MS, int(math.ceil(budget_ms / max_units / 1000.0)) * 1000)


def _dedupe_by_iou(valid: list[Candidate]) -> list[Candidate]:
    """Drop near-duplicate intervals, strongest score first.

    Kept intervals are bucketed by start time so each candidate only compares
    against the handful of intervals that can actually overlap it. The naive
    all-pairs scan is quadratic and dominates runtime on long sources.
    """
    ordered = sorted(valid, key=lambda item: (-item.score, _candidate_key(item)))
    if not ordered:
        return []
    max_len = max(candidate.duration_ms for candidate in ordered)
    bucket = max(1_000, max_len)
    index: dict[int, list[Candidate]] = {}
    kept: list[Candidate] = []
    for candidate in ordered:
        duplicate = False
        for key in range((candidate.start_ms - max_len) // bucket, candidate.end_ms // bucket + 1):
            for existing in index.get(key, ()):
                if interval_iou(
                    existing.start_ms, existing.end_ms, candidate.start_ms, candidate.end_ms
                ) >= 0.6:
                    duplicate = True
                    break
            if duplicate:
                break
        if duplicate:
            continue
        kept.append(candidate)
        index.setdefault(candidate.start_ms // bucket, []).append(candidate)
    return kept


def _reconstruct(
    take: list[bytearray] | bytearray,
    durations: list[int],
    predecessors: list[int],
    count: int,
    units: int,
    clips: int | None,
) -> list[int]:
    """Walk the recorded take/skip decisions back into chosen indices.

    ``clips`` is ``None`` when the selection has no clip-count limit; in that
    case a single decision layer is indexed directly.
    """
    layers = take if isinstance(take, list) else [take]
    width = len(layers[0]) // (len(predecessors) + 1)
    indices: list[int] = []
    remaining_clips = clips if clips is not None else len(durations)
    while count > 0 and units > 0 and remaining_clips > 0:
        layer = layers[remaining_clips if clips is not None else 0]
        if layer[count * width + units]:
            index = count - 1
            indices.append(index)
            units -= durations[index]
            count = predecessors[index] + 1
            remaining_clips -= 1
        else:
            count -= 1
    indices.reverse()
    return indices


def _select_unbounded(
    scores: list[float],
    durations: list[int],
    predecessors: list[int],
    budget_units: int,
) -> list[int]:
    """Budgeted weighted interval scheduling without a clip-count limit.

    Only the running best score is stored per state; the selection is rebuilt
    from the recorded take/skip decisions. Storing the chosen intervals inside
    every state is what previously made long videos exhaust memory.
    """
    n = len(scores)
    width = budget_units + 1
    best = [0.0] * ((n + 1) * width)
    take = bytearray((n + 1) * width)
    for count in range(1, n + 1):
        index = count - 1
        duration = durations[index]
        row = count * width
        skip_row = (count - 1) * width
        take_row = (predecessors[index] + 1) * width
        value = scores[index]
        for units in range(1, budget_units + 1):
            candidate = best[skip_row + units]
            if duration <= units:
                taken = best[take_row + units - duration] + value
                if taken > candidate:
                    candidate = taken
                    take[row + units] = 1
            best[row + units] = candidate
    return _reconstruct(take, durations, predecessors, n, budget_units, None)


def _select_bounded(
    scores: list[float],
    durations: list[int],
    predecessors: list[int],
    budget_units: int,
    max_clips: int,
) -> list[int]:
    """Same recurrence as :func:`_select_unbounded` with a clip-count cap."""
    n = len(scores)
    width = budget_units + 1
    size = (n + 1) * width
    best = [[0.0] * size for _ in range(max_clips + 1)]
    take = [bytearray(size) for _ in range(max_clips + 1)]
    for clips in range(1, max_clips + 1):
        layer = best[clips]
        previous_layer = best[clips - 1]
        decisions = take[clips]
        for count in range(1, n + 1):
            index = count - 1
            duration = durations[index]
            row = count * width
            skip_row = (count - 1) * width
            take_row = (predecessors[index] + 1) * width
            value = scores[index]
            for units in range(1, budget_units + 1):
                candidate = layer[skip_row + units]
                if duration <= units:
                    taken = previous_layer[take_row + units - duration] + value
                    if taken > candidate:
                        candidate = taken
                        decisions[row + units] = 1
                layer[row + units] = candidate
    return _reconstruct(take, durations, predecessors, n, budget_units, max_clips)


def select_auto_candidates(
    candidates: Iterable[Candidate],
    config: HighlightConfig,
    source_duration_ms: int | None = None,
) -> AutoSelection:
    """Select a deterministic, non-overlapping reel within a time budget.

    Candidate durations are the primary intervals, never preview padding. A
    weighted interval scheduling recurrence adds a duration budget and clip
    count, avoiding the common failure where one long clip blocks several
    shorter, higher-value clips.
    """
    all_candidates = list(candidates)
    source_duration_ms = source_duration_ms or max(
        (candidate.end_ms for candidate in all_candidates), default=0
    )
    target_seconds = resolve_target_duration_seconds(
        source_duration_ms,
        config.target_duration_seconds,
        configured_ratio=config.target_duration_ratio,
    )
    budget_ms = round(target_seconds * 1000)
    for candidate in all_candidates:
        candidate.selected = False
        if candidate.status == "keep":
            candidate.status = "unprocessed"

    valid = [
        candidate
        for candidate in all_candidates
        if candidate.status != "ignore"
        and candidate.score >= config.score_threshold
        and candidate.duration_ms > 0
    ]
    deduped = _dedupe_by_iou(valid)

    if budget_ms <= 0 or not deduped:
        return AutoSelection([], budget_ms=budget_ms, total_duration_ms=0)

    ordered = sorted(deduped, key=lambda item: (item.end_ms, item.start_ms, item.id))
    scores = [candidate.score for candidate in ordered]
    ends = [candidate.end_ms for candidate in ordered]
    # Binary search beats the previous backward scan: it is O(n log n) overall
    # instead of O(n^2) on sources that produce thousands of candidates.
    predecessors = [bisect_right(ends, candidate.start_ms) - 1 for candidate in ordered]

    max_clips = min(config.max_auto_clips, len(ordered)) if config.max_auto_clips else 0
    layers = max_clips if max_clips else 1
    quantum = _quantum_ms(budget_ms, len(ordered) + 1, layers + 1)
    budget_units = max(1, budget_ms // quantum)
    durations = [
        max(1, (candidate.duration_ms + quantum - 1) // quantum) for candidate in ordered
    ]

    if max_clips:
        chosen_indices = _select_bounded(
            scores, durations, predecessors, budget_units, max_clips
        )
    else:
        chosen_indices = _select_unbounded(scores, durations, predecessors, budget_units)

    chosen = [ordered[index] for index in chosen_indices]
    chosen.sort(key=_candidate_key)
    chosen_objects = {id(candidate) for candidate in chosen}
    for candidate in all_candidates:
        if id(candidate) in chosen_objects:
            candidate.selected = True
            candidate.status = "keep"
    selected_duration = sum(candidate.duration_ms for candidate in chosen)
    return AutoSelection(chosen, budget_ms=budget_ms, total_duration_ms=selected_duration)


def selection_summary(
    candidates: Iterable[Candidate],
    config: HighlightConfig,
    source_duration_ms: int | None = None,
) -> dict[str, object]:
    """Return a serializable explanation of the automatic selection."""
    all_candidates = list(candidates)
    selected = sorted(
        (candidate for candidate in all_candidates if candidate.selected), key=_candidate_key
    )
    return {
        "algorithm": ALGORITHM_VERSION,
        "target_duration_ratio": config.target_duration_ratio,
        "target_duration_ms": round(
            resolve_target_duration_seconds(
                source_duration_ms
                or max((candidate.end_ms for candidate in all_candidates), default=0),
                config.target_duration_seconds,
                configured_ratio=config.target_duration_ratio,
            )
            * 1000
        ),
        "selected_count": len(selected),
        "selected_duration_ms": sum(candidate.duration_ms for candidate in selected),
        "candidate_count": len(all_candidates),
        "ignored_count": sum(candidate.status == "ignore" for candidate in all_candidates),
        # Ranking blends an embedded-subtitle signal, so record how much of the
        # candidate set it actually reached. A zero here means the source had no
        # usable cues and the reel was ranked on audio and video alone.
        "transcript_candidate_count": sum(
            "transcript_density" in candidate.reasons for candidate in all_candidates
        ),
        "transcript_rescued_count": sum(
            "dialogue_rescue" in candidate.reasons for candidate in all_candidates
        ),
        "selected": [
            {
                "id": candidate.id,
                "start_ms": candidate.start_ms,
                "end_ms": candidate.end_ms,
                "score": candidate.score,
            }
            for candidate in selected
        ],
    }

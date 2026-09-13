"""Automatic highlight selection from scored candidates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .models import Candidate, HighlightConfig, interval_iou


ALGORITHM_VERSION = "weighted-interval-budget-v1"
_BUDGET_QUANTUM_MS = 1000


@dataclass(frozen=True)
class AutoSelection:
    selected: list[Candidate]
    budget_ms: int
    total_duration_ms: int


def _candidate_key(candidate: Candidate) -> tuple[int, int, str]:
    return candidate.start_ms, candidate.end_ms, candidate.id


def _better(
    left: tuple[float, int, tuple[tuple[int, int, str], ...], tuple[int, ...]],
    right: tuple[float, int, tuple[tuple[int, int, str], ...], tuple[int, ...]],
) -> tuple[float, int, tuple[tuple[int, int, str], ...], tuple[int, ...]]:
    """Compare states by budget utilization, score, then stable timeline order."""
    if left[1] != right[1]:
        return left if left[1] > right[1] else right
    if left[0] != right[0]:
        return left if left[0] > right[0] else right
    return left if left[2] < right[2] else right


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
    source_duration_ms = source_duration_ms or max((candidate.end_ms for candidate in all_candidates), default=0)
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
    deduped: list[Candidate] = []
    for candidate in sorted(valid, key=lambda item: (-item.score, _candidate_key(item))):
        if any(
            interval_iou(
                existing.start_ms,
                existing.end_ms,
                candidate.start_ms,
                candidate.end_ms,
            ) >= 0.6
            for existing in deduped
        ):
            continue
        deduped.append(candidate)

    if budget_ms <= 0 or not deduped:
        return AutoSelection([], budget_ms, 0)

    ordered = sorted(deduped, key=lambda item: (item.end_ms, item.start_ms, item.id))
    max_clips = config.max_auto_clips or len(ordered)
    budget_units = budget_ms // _BUDGET_QUANTUM_MS
    predecessors: list[int] = []
    for index, candidate in enumerate(ordered):
        predecessor = -1
        for earlier in range(index - 1, -1, -1):
            if ordered[earlier].end_ms <= candidate.start_ms:
                predecessor = earlier
                break
        predecessors.append(predecessor)

    zero_state: tuple[float, int, tuple[tuple[int, int, str], ...], tuple[int, ...]] = (0.0, 0, (), ())

    def take_or_skip(
        candidate_index: int,
        previous: tuple[float, int, tuple[tuple[int, int, str], ...], tuple[int, ...]],
    ) -> tuple[float, int, tuple[tuple[int, int, str], ...], tuple[int, ...]]:
        candidate = ordered[candidate_index]
        return (
            previous[0] + candidate.score,
            previous[1] + candidate.duration_ms,
            previous[2] + (_candidate_key(candidate),),
            previous[3] + (candidate_index,),
        )

    if config.max_auto_clips:
        @lru_cache(maxsize=None)
        def solve(
            count: int, remaining_units: int, remaining_clips: int
        ) -> tuple[float, int, tuple[tuple[int, int, str], ...], tuple[int, ...]]:
            if count <= 0 or remaining_units <= 0 or remaining_clips <= 0:
                return zero_state
            candidate_index = count - 1
            candidate = ordered[candidate_index]
            best = solve(count - 1, remaining_units, remaining_clips)
            duration_units = (candidate.duration_ms + _BUDGET_QUANTUM_MS - 1) // _BUDGET_QUANTUM_MS
            if duration_units <= remaining_units:
                previous = solve(
                    predecessors[candidate_index] + 1,
                    remaining_units - duration_units,
                    remaining_clips - 1,
                )
                best = _better(take_or_skip(candidate_index, previous), best)
            return best

        best = solve(len(ordered), budget_units, max_clips)
    else:
        @lru_cache(maxsize=None)
        def solve_unlimited(
            count: int, remaining_units: int
        ) -> tuple[float, int, tuple[tuple[int, int, str], ...], tuple[int, ...]]:
            if count <= 0 or remaining_units <= 0:
                return zero_state
            candidate_index = count - 1
            candidate = ordered[candidate_index]
            best = solve_unlimited(count - 1, remaining_units)
            duration_units = (candidate.duration_ms + _BUDGET_QUANTUM_MS - 1) // _BUDGET_QUANTUM_MS
            if duration_units <= remaining_units:
                previous = solve_unlimited(
                    predecessors[candidate_index] + 1,
                    remaining_units - duration_units,
                )
                best = _better(take_or_skip(candidate_index, previous), best)
            return best

        best = solve_unlimited(len(ordered), budget_units)
    chosen = [ordered[index] for index in best[3]]
    chosen.sort(key=_candidate_key)
    chosen_objects = {id(candidate) for candidate in chosen}
    for candidate in all_candidates:
        if id(candidate) in chosen_objects:
            candidate.selected = True
            candidate.status = "keep"
    return AutoSelection(chosen, best[1], budget_ms)


def selection_summary(
    candidates: Iterable[Candidate],
    config: HighlightConfig,
    source_duration_ms: int | None = None,
) -> dict[str, object]:
    """Return a serializable explanation of the automatic selection."""
    all_candidates = list(candidates)
    selected = sorted((candidate for candidate in all_candidates if candidate.selected), key=_candidate_key)
    return {
        "algorithm": ALGORITHM_VERSION,
        "target_duration_ratio": config.target_duration_ratio,
        "target_duration_ms": round(
            resolve_target_duration_seconds(
                source_duration_ms or max((candidate.end_ms for candidate in all_candidates), default=0),
                config.target_duration_seconds,
                configured_ratio=config.target_duration_ratio,
            )
            * 1000
        ),
        "selected_count": len(selected),
        "selected_duration_ms": sum(candidate.duration_ms for candidate in selected),
        "candidate_count": len(all_candidates),
        "ignored_count": sum(candidate.status == "ignore" for candidate in all_candidates),
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

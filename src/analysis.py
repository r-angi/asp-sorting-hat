"""Analysis and console reporting helpers for crew assignments.

All score helpers are zero-safe (empty rosters / centers return well-defined
values rather than crashing) and avoid the previously quadratic
``is_person_at_center`` calls inside their inner loops by precomputing a
``name -> center`` map once per invocation.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol, TypedDict

import polars as pl
from ortools.sat.python import cp_model

from src.models import Center, Crew, Leader, Youth

type PersonCrew = (
    dict[tuple[str, str, str], cp_model.IntVar] | dict[tuple[str, str, str], int]
)


class SolverLike(Protocol):
    """Subset of ``cp_model.CpSolver`` used by reporting code; satisfied by the
    re-analysis ``AssignmentsLookup`` adapter as well.

    ``var`` is typed ``Any`` because OR-Tools' ``CpSolver.Value`` accepts a
    broader ``LinearExprT`` than our re-analysis adapter; the protocol stays
    structurally compatible with both.
    """

    def Value(self, var: Any) -> int: ...


class CenterSummary(TypedDict):
    total_youth: int
    total_adults: int
    years: dict[str, int]
    gender: dict[str, int]
    history: dict[str, int]
    friend_score: float


def build_name_to_center(
    solver: SolverLike,
    person_crew: PersonCrew,
    centers: list[Center],
) -> dict[str, str]:
    """Map each placed person (youth or pre-assigned leader) to their center.

    Pre-assigned leaders come from ``crew.adults``. Solver-placed youth come
    from ``person_crew`` Booleans set to 1. Flexible (solver-placed) adults are
    not included — same semantics as the legacy :func:`is_person_at_center`.
    """
    mapping: dict[str, str] = {}
    for center in centers:
        for crew in center.crews:
            for name in crew.adult_names:
                mapping[name] = center.name
    for (name, center_name, _crew_name), var in person_crew.items():
        if solver.Value(var) == 1:
            mapping[name] = center_name
    return mapping


def synthesize_centers_from_assignments(
    person_crew: dict[tuple[str, str, str], int],
) -> list[Center]:
    """Reconstruct a minimal ``list[Center]`` topology from a placed ``person_crew``.

    Used by the ``--no-reassignment`` analysis path when the crews-CSV scaffold is
    absent (e.g. every leader is ``CENTER_ONLY`` so :func:`get_centers_from_adults_df`
    returns ``[]``) but a saved assignments workbook supplies the placements.
    The returned crews carry no leaders; only ``center.name`` and
    ``crew.name`` strings are needed for downstream reporting and clustering.
    """
    by_center: dict[str, set[str]] = defaultdict(set)
    for (_name, center_name, crew_name), value in person_crew.items():
        if value == 1 and center_name and crew_name:
            by_center[center_name].add(crew_name)
    return [
        Center(name=cname, crews=[Crew(name=crew_name) for crew_name in sorted(crews)])
        for cname, crews in sorted(by_center.items())
    ]


def is_person_at_center(
    solver: SolverLike,
    person_crew: PersonCrew,
    person_name: str,
    center: Center,
) -> bool:
    """Return ``True`` if ``person_name`` lands at ``center`` in the solved model.

    Prefer :func:`build_name_to_center` for tight loops; this single-shot helper
    is kept for clustering visualization which probes ad-hoc names.
    """
    for crew in center.crews:
        if person_name in crew.adult_names:
            return True
        key = (person_name, center.name, crew.name)
        if key in person_crew and solver.Value(person_crew[key]) == 1:
            return True
    return False


_CHOICES_AND_WEIGHTS: tuple[tuple[str, int], ...] = (
    ('first_choice', 3),
    ('second_choice', 2),
    ('third_choice', 1),
)


def calculate_friend_scores(
    solver: SolverLike,
    person_crew: PersonCrew,
    youth_list: list[Youth],
    centers: list[Center],
) -> tuple[dict[str, float], float]:
    """Per-center normalized friend score and overall average.

    Weights: 1st / 2nd / 3rd choice = +3 / +2 / +1 when both youth land at the
    same center. Non-roster (leader) picks are ignored here; the solver
    objective already accounts for them.
    """
    empty_centers = {c.name: 0.0 for c in centers}
    if not youth_list:
        return empty_centers, 0.0

    name_to_center = build_name_to_center(solver, person_crew, centers)
    youth_dict = {y.name: y for y in youth_list}
    center_scores: dict[str, float] = dict(empty_centers)
    center_count: dict[str, int] = {c.name: 0 for c in centers}

    for youth in youth_list:
        my_center = name_to_center.get(youth.name)
        if my_center is None:
            continue
        center_count[my_center] += 1
        for choice_attr, weight in _CHOICES_AND_WEIGHTS:
            friend = getattr(youth, choice_attr)
            if friend and friend in youth_dict and name_to_center.get(friend) == my_center:
                center_scores[my_center] += weight

    normalized = {
        name: round(center_scores[name] / count, 2) if count > 0 else 0.0
        for name, count in center_count.items()
    }
    avg_score = round(sum(center_scores.values()) / len(youth_list), 2)
    return normalized, avg_score


def calculate_historical_friend_scores(
    centers: list[Center], year: int
) -> tuple[dict[str, float], float]:
    """Compute friend-preference scores for the *manual* roster in ``crews_{year}.csv``.

    Used to baseline a hand-edited assignment against the solver's output. Same
    weights as :func:`calculate_friend_scores` so per-center deltas are
    directly comparable to the solver's printed numbers.
    """
    youth_df = pl.read_csv(f'./data/clean/crews_{year}.csv').filter(pl.col('role') != 'Adult')
    buddies_df = pl.read_csv(f'./data/clean/buddies_{year}.csv')
    joined = youth_df.join(buddies_df, on='name', how='left')

    valid_centers = {c.name for c in centers}
    name_to_center: dict[str, str] = {
        row['name']: row['Center']
        for row in joined.iter_rows(named=True)
        if row['Center'] in valid_centers
    }
    if not name_to_center:
        return {c.name: 0.0 for c in centers}, 0.0

    center_scores: dict[str, float] = {c.name: 0.0 for c in centers}
    center_count: dict[str, int] = {c.name: 0 for c in centers}
    overall_score = 0.0

    for row in joined.iter_rows(named=True):
        my_center = row['Center']
        if my_center not in valid_centers:
            continue
        center_count[my_center] += 1
        for choice_attr, weight in _CHOICES_AND_WEIGHTS:
            friend = row.get(choice_attr)
            if friend and name_to_center.get(friend) == my_center:
                center_scores[my_center] += weight
                overall_score += weight

    normalized = {
        name: round(center_scores[name] / count, 2) if count > 0 else 0.0
        for name, count in center_count.items()
    }
    avg_score = round(overall_score / len(name_to_center), 2)
    return normalized, avg_score


def calculate_friend_choice_stats(
    solver: SolverLike,
    person_crew: PersonCrew,
    youth_list: list[Youth],
    centers: list[Center],
) -> dict[str, float]:
    """Percent of youth whose 1st / 2nd / 3rd choice friend lands at the same center."""
    total = len(youth_list)
    keys = ('first_choice_pct', 'second_choice_pct', 'third_choice_pct', 'multiple_friends_pct')
    if total == 0:
        return dict.fromkeys(keys, 0.0)

    name_to_center = build_name_to_center(solver, person_crew, centers)
    youth_dict = {y.name: y for y in youth_list}
    counts = {'first_choice': 0, 'second_choice': 0, 'third_choice': 0, 'multiple': 0}

    for youth in youth_list:
        my_center = name_to_center.get(youth.name)
        if my_center is None:
            continue
        matches = 0
        for choice_attr, _weight in _CHOICES_AND_WEIGHTS:
            friend = getattr(youth, choice_attr)
            if friend and friend in youth_dict and name_to_center.get(friend) == my_center:
                counts[choice_attr] += 1
                matches += 1
        if matches > 1:
            counts['multiple'] += 1

    return {
        'first_choice_pct': round(counts['first_choice'] / total * 100, 1),
        'second_choice_pct': round(counts['second_choice'] / total * 100, 1),
        'third_choice_pct': round(counts['third_choice'] / total * 100, 1),
        'multiple_friends_pct': round(counts['multiple'] / total * 100, 1),
    }


BUDDY_MATCH_BUCKETS: tuple[int, ...] = (0, 1, 2, 3)


def calculate_friend_match_buckets(
    solver: SolverLike,
    person_crew: PersonCrew,
    youth_list: list[Youth],
    centers: list[Center],
) -> dict[str, dict[int, int]]:
    """Per-center distribution of how many friend picks land at each youth's center.

    Returns ``{center_name: {0: n, 1: n, 2: n, 3: n}}`` — the count of youth at
    that center whose 0/1/2/3 same-center buddy matches sum to that bucket. Only
    roster-youth picks count (leader picks are ignored, matching
    :func:`calculate_friend_choice_stats`). Youth without an assigned center are
    skipped, so summed bucket counts equal the placed-youth headcount.
    """
    result: dict[str, dict[int, int]] = {
        c.name: dict.fromkeys(BUDDY_MATCH_BUCKETS, 0) for c in centers
    }
    if not youth_list or not centers:
        return result

    name_to_center = build_name_to_center(solver, person_crew, centers)
    youth_dict = {y.name: y for y in youth_list}

    for youth in youth_list:
        my_center = name_to_center.get(youth.name)
        if my_center is None or my_center not in result:
            continue
        matches = 0
        for choice_attr, _weight in _CHOICES_AND_WEIGHTS:
            friend = getattr(youth, choice_attr)
            if friend and friend in youth_dict and name_to_center.get(friend) == my_center:
                matches += 1
        result[my_center][matches] += 1
    return result


def print_crew_assignments(
    solver: SolverLike,
    person_crew: PersonCrew,
    youth_list: list[Youth],
    centers: list[Center],
    adult_crew: dict[tuple[str, str, str], cp_model.IntVar] | None = None,
    center_only_adults: Sequence[Leader] | None = None,
    unassigned_adults: Sequence[Leader] | None = None,
) -> None:
    """Print full assignment / diversity / friend-choice breakdown to stdout."""
    youth_dict = {y.name: y for y in youth_list}
    friend_scores, _ = calculate_friend_scores(solver, person_crew, youth_list, centers)
    flex_pool: list[Leader] = list(center_only_adults or []) + list(unassigned_adults or [])

    center_stats: dict[str, CenterSummary] = {}

    for center in centers:
        print(f'\nCenter {center.name}:')
        summary: CenterSummary = {
            'total_youth': 0,
            'total_adults': 0,
            'years': {'Fr': 0, 'So': 0, 'Jr': 0, 'Sr': 0},
            'gender': {'M': 0, 'F': 0},
            'history': {'V': 0, 'N': 0},
            'friend_score': friend_scores[center.name],
        }
        center_stats[center.name] = summary

        for crew in center.crews:
            crew_youth = [
                youth.name
                for youth in youth_list
                if (youth.name, center.name, crew.name) in person_crew
                and solver.Value(person_crew[youth.name, center.name, crew.name]) == 1
            ]
            preassigned_adults = [a.name for a in crew.adults]
            flex_adults: list[str] = []
            if adult_crew is not None:
                for leader in flex_pool:
                    key = (leader.name, center.name, crew.name)
                    if key in adult_crew and solver.Value(adult_crew[key]) == 1:
                        flex_adults.append(leader.name)
            crew_adults = preassigned_adults + flex_adults

            year_counts: dict[str, int] = {}
            gender_counts: dict[str, int] = {}
            history_counts: dict[str, int] = {}
            for person in crew_youth:
                y = youth_dict[person]
                year_counts[y.year] = year_counts.get(y.year, 0) + 1
                gender_counts[y.gender] = gender_counts.get(y.gender, 0) + 1
                history_counts[y.history] = history_counts.get(y.history, 0) + 1
                summary['years'][y.year] = summary['years'].get(y.year, 0) + 1
                summary['gender'][y.gender] = summary['gender'].get(y.gender, 0) + 1
                summary['history'][y.history] = summary['history'].get(y.history, 0) + 1

            print(f'  {crew.name}:')
            print(f'    Youth: {crew_youth}')
            print(f'    Adults: {crew_adults}')
            print(f'    Years: {year_counts}')
            print(f'    Gender (M/F): {gender_counts}')
            print(f'    History (vet/new): {history_counts}')

            summary['total_youth'] += len(crew_youth)
            summary['total_adults'] += len(crew_adults)

    print('\n=== Summary Statistics ===')
    total_youth = sum(s['total_youth'] for s in center_stats.values())
    total_adults = sum(s['total_adults'] for s in center_stats.values())

    print('\nOverall Totals:')
    print(f'Total Youth: {total_youth}')
    print(f'Total Adults: {total_adults}')
    print(f'Total Participants: {total_youth + total_adults}')

    friend_stats = calculate_friend_choice_stats(solver, person_crew, youth_list, centers)
    print('\nFriend Choice Statistics:')
    print(f'Youth with first choice friend: {friend_stats["first_choice_pct"]}%')
    print(f'Youth with second choice friend: {friend_stats["second_choice_pct"]}%')
    print(f'Youth with third choice friend: {friend_stats["third_choice_pct"]}%')
    print(f'Youth with multiple friends: {friend_stats["multiple_friends_pct"]}%')

    print('\nCenter-by-Center Statistics:')
    for center_name, stats in center_stats.items():
        print(f'\n{center_name}:')
        total = stats['total_youth'] + stats['total_adults']
        print(f'  Total Participants: {total}')
        print(f'  Youth: {stats["total_youth"]}, Adults: {stats["total_adults"]}')
        print(f'  Friend Score: {stats["friend_score"]:.2f}')
        print(f'  Years: {stats["years"]}')
        print(f'  Gender: {stats["gender"]}')
        print(f'  History: {stats["history"]}')


def status_to_string(status: int) -> str:
    """Map an OR-Tools CP-SAT status code to its human-readable name."""
    if status == cp_model.OPTIMAL:
        return 'OPTIMAL'
    if status == cp_model.FEASIBLE:
        return 'FEASIBLE'
    if status == cp_model.INFEASIBLE:
        return 'INFEASIBLE'
    if status == cp_model.MODEL_INVALID:
        return 'MODEL_INVALID'
    return 'UNKNOWN'

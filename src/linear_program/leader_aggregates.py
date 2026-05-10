"""Per-crew leader aggregation helpers.

Both ``constraints.py`` (driver, new+vet) and ``objectives.py`` (gender +
history balance) need the same two operations on each crew:

1. **Pre-count**: how many leaders are *fixed* on this crew with
   ``getattr(leader, attribute) == value``? Plain ``int``.
2. **Flex indicators**: the ``adult_crew`` BoolVars for solver-placed leaders
   matching ``getattr(adult, attribute) == value`` and (optionally)
   ``adult.fixed_center == fixed_center_only``. Returned as a list of vars to
   ``sum(...)`` into a constraint or objective.

Sourcing both from the typed ``Adult`` / ``YoungAdult`` instances on
``crew.adults`` and ``ctx.center_only_adults`` / ``ctx.unassigned_adults``
keeps the helpers four-line transparent and removes the previous duplication.
"""

from ortools.sat.python import cp_model

from src.linear_program.context import AdultCrewVars, ModelContext
from src.models import Crew, Leader


def pre_count(crew: Crew, attribute: str, value: str) -> int:
    """Count pre-assigned leaders on ``crew`` whose ``attribute == value``."""
    return sum(1 for a in crew.adults if getattr(a, attribute) == value)


def flex_indicators(
    adult_crew: AdultCrewVars,
    center_name: str,
    crew_name: str,
    flex_leaders: list[Leader],
    *,
    attribute: str,
    value: str,
    fixed_center_only: str | None = None,
) -> list[cp_model.IntVar]:
    """BoolVars for solver-placed leaders matching ``attribute == value`` on a crew.

    ``fixed_center_only=center.name`` filters CENTER_ONLY leaders to their
    fixed center; pass ``None`` for UNASSIGNED leaders that may land anywhere.
    """
    indicators: list[cp_model.IntVar] = []
    for leader in flex_leaders:
        if fixed_center_only is not None and leader.fixed_center != fixed_center_only:
            continue
        if getattr(leader, attribute) != value:
            continue
        var = adult_crew.get((leader.name, center_name, crew_name))
        if var is not None:
            indicators.append(var)
    return indicators


def crew_attribute_total(
    ctx: ModelContext,
    crew: Crew,
    center_name: str,
    *,
    attribute: str,
    value: str,
) -> tuple[int, list[cp_model.IntVar]]:
    """Convenience: ``(pre_count, flex_indicators_for_both_pools)`` on a crew.

    The combined ``flex`` list mixes CENTER_ONLY (filtered by center) and
    UNASSIGNED adults so callers can ``int_pre + sum(flex_list)``.
    """
    flex = flex_indicators(
        ctx.adult_crew, center_name, crew.name, ctx.center_only_adults,
        attribute=attribute, value=value, fixed_center_only=center_name,
    ) + flex_indicators(
        ctx.adult_crew, center_name, crew.name, ctx.unassigned_adults,
        attribute=attribute, value=value, fixed_center_only=None,
    )
    return pre_count(crew, attribute, value), flex

"""CP-SAT constraint helpers for the crew-assignment model.

Every helper takes a single ``ModelContext`` and posts hard constraints on
``ctx.model``. The context bundles the cached ``at_center`` map (so we never
recompute the per-crew sum), the sparse ``person_crew`` / ``adult_crew``
dicts, and the typed ``Adult`` / ``Leader`` lists.

Hard constraints already encoded by sparse variable construction (parent
center, a pre-placed parent's own crew, past leaders) intentionally have no
helper here — ``lp_model._compute_eligibility`` is the single source of truth.

The exception is :func:`enforce_parent_crew_separation_constraint`, which
covers parents whose crew is still a solver decision: the "youth not in
parent's crew" rule must be a runtime constraint pairing ``person_crew``
with ``adult_crew``.
"""

from typing import cast

from ortools.sat.python import cp_model

from src.linear_program.adult_placement import adult_presence_at_center
from src.linear_program.context import ModelContext
from src.linear_program.leader_aggregates import crew_attribute_total
from src.models import Leader, PlacementMode, Youth


# ----- Youth-side hard constraints --------------------------------------------


def add_one_crew_per_youth(ctx: ModelContext) -> None:
    """Each youth lands on exactly one of their eligible (center, crew) cells.

    An empty eligibility list yields ``sum([]) == 1`` (infeasible) — by design,
    so the solver reports a clear conflict instead of silently placing nobody.
    """
    for y in ctx.youth_list:
        terms = [ctx.person_crew[(y.name, c, k)] for c, k in ctx.eligibility[y.name]]
        ctx.model.Add(sum(terms) == 1)


def enforce_sibling_center_constraint(ctx: ModelContext) -> None:
    """Siblings live at the same center.

    Eligibility narrows to a shared subset via propagation; this constraint pins
    them when more than one shared option remains. Each unordered pair is posted
    once even when sibling lists are symmetric.
    """
    processed: set[tuple[str, str]] = set()
    for y in ctx.youth_list:
        for sibling in y.siblings_list:
            if sibling not in ctx.youth_dict:
                continue
            pair = cast(tuple[str, str], tuple(sorted([y.name, sibling])))
            if pair in processed:
                continue
            processed.add(pair)
            for center in ctx.centers:
                ctx.model.Add(
                    ctx.at_center[(y.name, center.name)]
                    == ctx.at_center[(sibling, center.name)]
                )


def enforce_sibling_crew_separation_constraint(ctx: ModelContext) -> None:
    """Siblings cannot share a crew. Each unordered pair processed once."""
    processed: set[tuple[str, str]] = set()
    for y in ctx.youth_list:
        for sibling in y.siblings_list:
            if sibling not in ctx.youth_dict:
                continue
            pair = cast(tuple[str, str], tuple(sorted([y.name, sibling])))
            if pair in processed:
                continue
            processed.add(pair)
            for center in ctx.centers:
                for crew in center.crews:
                    a = ctx.person_crew.get((y.name, center.name, crew.name))
                    b = ctx.person_crew.get((sibling, center.name, crew.name))
                    if a is None or b is None:
                        continue
                    ctx.model.Add(a + b <= 1)


def enforce_parent_crew_separation_constraint(ctx: ModelContext) -> None:
    """Youth cannot share a crew with a parent whose crew is solver-assigned.

    For a parent already placed in a specific crew, the parent's exact
    ``(center, crew)`` is pruned from the youth's eligibility upstream (no
    ``person_crew`` Boolean exists for that cell). This helper handles the
    remaining parents — those with only a center on the crews CSV (or no
    placement at all) — by pairing
    ``person_crew[(youth, c, k)] + adult_crew[(parent, c, k)] <= 1`` over
    every (center, crew) the parent could land on.
    """
    for youth in ctx.youth_list:
        for parent_name in youth.parent_names_list:
            leader = ctx.leaders_by_name.get(parent_name)
            if leader is None or leader.placement == PlacementMode.FIXED:
                continue

            fixed_center = leader.fixed_center
            for center in ctx.centers:
                if fixed_center is not None and fixed_center != center.name:
                    continue
                for crew in center.crews:
                    picker = ctx.person_crew.get((youth.name, center.name, crew.name))
                    parent_var = ctx.adult_crew.get((parent_name, center.name, crew.name))
                    if picker is None or parent_var is None:
                        continue
                    ctx.model.Add(picker + parent_var <= 1)


def enforce_friend_separation_constraint(ctx: ModelContext) -> None:
    """Buddy picks may not share a crew with the picker.

    Applied uniformly to youth-youth and youth-adult pairs. For an adult buddy:
    pre-assigned (FIXED) → forbid the picker from that exact crew; flexible →
    pair the picker's BoolVar with the adult's adult_crew BoolVar.
    """
    processed_youth_pairs: set[tuple[str, str]] = set()
    processed_youth_adult: set[tuple[str, str]] = set()

    for youth in ctx.youth_list:
        choices: list[str] = [
            c for c in (youth.first_choice, youth.second_choice, youth.third_choice) if c is not None
        ]
        for friend in choices:
            if friend in ctx.youth_dict:
                pair = cast(tuple[str, str], tuple(sorted([youth.name, friend])))
                if pair in processed_youth_pairs:
                    continue
                processed_youth_pairs.add(pair)
                for center in ctx.centers:
                    for crew in center.crews:
                        a = ctx.person_crew.get((youth.name, center.name, crew.name))
                        b = ctx.person_crew.get((friend, center.name, crew.name))
                        if a is None or b is None:
                            continue
                        ctx.model.Add(a + b <= 1)
            elif friend in ctx.leaders_by_name:
                key = (youth.name, friend)
                if key in processed_youth_adult:
                    continue
                processed_youth_adult.add(key)

                leader = ctx.leaders_by_name[friend]
                fixed_center = leader.fixed_center
                for center in ctx.centers:
                    if fixed_center is not None and fixed_center != center.name:
                        continue
                    for crew in center.crews:
                        picker = ctx.person_crew.get((youth.name, center.name, crew.name))
                        if picker is None:
                            continue
                        if leader.placement == PlacementMode.FIXED:
                            if friend in crew.adult_names:
                                ctx.model.Add(picker == 0)
                            continue
                        adult_var = ctx.adult_crew.get((friend, center.name, crew.name))
                        if adult_var is not None:
                            ctx.model.Add(picker + adult_var <= 1)


def enforce_friend_center_constraint(ctx: ModelContext) -> None:
    """At least one buddy choice (roster youth or adult/YA leader) must end up
    at the youth's center.

    Pairs with friend_separation to give "same center, different crew" semantics.
    """
    for youth in ctx.youth_list:
        choices = [youth.first_choice, youth.second_choice, youth.third_choice]
        youth_choices = [c for c in choices if c is not None and c in ctx.youth_dict]
        adult_choices = [
            c for c in choices if c is not None and c not in ctx.youth_dict and c in ctx.leaders_by_name
        ]
        if not youth_choices and not adult_choices:
            continue

        for center in ctx.centers:
            youth_at_center = ctx.at_center[(youth.name, center.name)]

            friend_terms: list[cp_model.IntVar | cp_model.LinearExpr | int] = [
                ctx.at_center[(friend, center.name)] for friend in youth_choices
            ]
            for adult_name in adult_choices:
                presence = adult_presence_at_center(
                    ctx.leaders_by_name[adult_name], center, ctx.adult_crew
                )
                if presence is None:
                    continue
                friend_terms.append(presence)

            ctx.model.Add(youth_at_center <= sum(friend_terms))


def enforce_supervision_group_limit(ctx: ModelContext, max_per_center: int = 2) -> None:
    """Each supervision group (A, B, C, ...) is independently capped per center."""
    groups: dict[str, list[Youth]] = {}
    for youth in ctx.youth_list:
        if youth.supervision_group:
            groups.setdefault(youth.supervision_group, []).append(youth)

    for group_youth in groups.values():
        for center in ctx.centers:
            total = sum(ctx.at_center[(y.name, center.name)] for y in group_youth)
            ctx.model.Add(total <= max_per_center)


def enforce_anti_buddy_constraint(ctx: ModelContext) -> None:
    """Prevent anti-buddies from being at the same center."""
    processed: set[tuple[str, str]] = set()
    for youth in ctx.youth_list:
        for anti in youth.anti_buddy_list:
            if anti not in ctx.youth_dict:
                continue
            pair = cast(tuple[str, str], tuple(sorted([youth.name, anti])))
            if pair in processed:
                continue
            processed.add(pair)
            for center in ctx.centers:
                ctx.model.Add(
                    ctx.at_center[(youth.name, center.name)]
                    + ctx.at_center[(anti, center.name)]
                    <= 1
                )


# ----- Adult-side: placement, headcount, driver, new+vet ----------------------


def assign_center_only_adults(ctx: ModelContext) -> None:
    """Each center-only leader lands on exactly one crew within their fixed center."""
    for leader in ctx.center_only_adults:
        assert leader.fixed_center is not None
        center = ctx.centers_by_name[leader.fixed_center]
        ctx.model.Add(
            sum(ctx.adult_crew[(leader.name, center.name, crew.name)] for crew in center.crews) == 1
        )


def assign_unassigned_adults(ctx: ModelContext) -> None:
    """Each unassigned leader lands on exactly one crew across all centers."""
    for leader in ctx.unassigned_adults:
        ctx.model.Add(
            sum(
                ctx.adult_crew[(leader.name, center.name, crew.name)]
                for center in ctx.centers
                for crew in center.crews
            )
            == 1
        )


def enforce_crew_headcount(ctx: ModelContext) -> None:
    """Combined min/max bounds on crew headcount and per-crew adult count.

    - Total headcount: youth + pre-assigned leaders + flexible leaders the solver
      places, bounded by ``min_crew_size`` / ``max_crew_size``.
    - Adult headcount: pre-assigned leaders + flexible leaders (Adult or YA),
      bounded by ``min_adults_per_crew`` / ``max_adults_per_crew``.

    Both sides reuse the same precomputed flex indicator list per crew, so the
    iteration walks each ``(center, crew)`` once.
    """
    cfg = ctx.cfg
    for center in ctx.centers:
        for crew in center.crews:
            preassigned = len(crew.adults)
            flex_indicators: list[cp_model.IntVar] = []
            for leader in ctx.center_only_adults:
                if leader.fixed_center != center.name:
                    continue
                var = ctx.adult_crew.get((leader.name, center.name, crew.name))
                if var is not None:
                    flex_indicators.append(var)
            for leader in ctx.unassigned_adults:
                var = ctx.adult_crew.get((leader.name, center.name, crew.name))
                if var is not None:
                    flex_indicators.append(var)

            youth_terms = [
                ctx.person_crew[(y.name, center.name, crew.name)]
                for y in ctx.regular_youth
                if (y.name, center.name, crew.name) in ctx.person_crew
            ]

            adult_total = preassigned + sum(flex_indicators)
            total_headcount = sum(youth_terms) + adult_total

            ctx.model.Add(total_headcount >= cfg.min_crew_size)
            ctx.model.Add(total_headcount <= cfg.max_crew_size)
            ctx.model.Add(adult_total >= cfg.min_adults_per_crew)
            ctx.model.Add(adult_total <= cfg.max_adults_per_crew)


def enforce_driver_per_crew(ctx: ModelContext) -> None:
    """At least one ``role == 'Adult'`` (driver) on every crew.

    Young Adults are excluded — they cannot drive.
    """
    for center in ctx.centers:
        for crew in center.crews:
            pre, flex = crew_attribute_total(ctx, crew, center.name, attribute='role', value='Adult')
            ctx.model.Add(pre + sum(flex) >= 1)


def enforce_new_requires_vet(ctx: ModelContext) -> None:
    """If a crew has a New leader, it must also have a Vet leader.

    Reified per crew so the constraint is vacuous on crews with no New leaders.
    Counts both Adults and Young Adults on each side.
    """
    for center in ctx.centers:
        for crew in center.crews:
            pre_new, flex_new = crew_attribute_total(ctx, crew, center.name, attribute='history', value='N')
            pre_vet, flex_vet = crew_attribute_total(ctx, crew, center.name, attribute='history', value='V')

            sum_new = pre_new + sum(flex_new)
            sum_vet = pre_vet + sum(flex_vet)

            if isinstance(sum_new, int):
                if sum_new == 0:
                    continue
                ctx.model.Add(sum_vet >= 1)
                continue

            has_new = ctx.model.NewBoolVar(f'has_new_leader_{center.name}_{crew.name}')
            ctx.model.Add(sum_new >= 1).OnlyEnforceIf(has_new)
            ctx.model.Add(sum_new == 0).OnlyEnforceIf(has_new.Not())
            ctx.model.Add(sum_vet >= 1).OnlyEnforceIf(has_new)


def break_symmetry_unassigned_adults(ctx: ModelContext) -> None:
    """Lex-order interchangeable unassigned leaders (same role/gender/history).

    Anchors each interchangeable group by forcing the alphabetically-earlier
    name to land on a lower-or-equal flat crew index than the next one. Adults
    and Young Adults form separate groups via the ``role`` key, so the rule
    never mixes them.
    """
    if not ctx.unassigned_adults:
        return

    flat_crews: list[tuple[str, str]] = [
        (center.name, crew.name) for center in ctx.centers for crew in center.crews
    ]
    if len(flat_crews) <= 1:
        return

    groups: dict[tuple[str, str | None, str | None], list[Leader]] = {}
    for leader in ctx.unassigned_adults:
        groups.setdefault((leader.role, leader.gender, leader.history), []).append(leader)

    for leaders in groups.values():
        if len(leaders) < 2:
            continue
        ordered = sorted(leaders, key=lambda a: a.name)
        index_terms: list[cp_model.LinearExpr | int | None] = []
        for leader in ordered:
            terms: list[cp_model.LinearExpr] = [
                cast(cp_model.LinearExpr, idx * ctx.adult_crew[(leader.name, center_name, crew_name)])
                for idx, (center_name, crew_name) in enumerate(flat_crews)
                if (leader.name, center_name, crew_name) in ctx.adult_crew
            ]
            index_terms.append(cast(cp_model.LinearExpr, sum(terms)) if terms else None)

        for prev, curr in zip(index_terms, index_terms[1:]):
            if prev is None or curr is None:
                continue
            ctx.model.Add(prev <= curr)

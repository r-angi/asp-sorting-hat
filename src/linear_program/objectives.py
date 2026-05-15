"""Objective-term builders for the crew-assignment model.

Soft preferences only — hard rules live in ``constraints.py``. Every helper
takes a single ``ModelContext`` and returns a list of weighted ``LinearExpr``
terms that ``lp_model`` sums into the ``Maximize`` objective.

Like the constraint helpers, these reuse the cached ``at_center`` map and the
sparse ``person_crew`` dict on ``ctx``. ``person_crew.get(key, 0)`` is the
safe lookup.
"""

from collections.abc import Sequence
from typing import Final, cast

from ortools.sat.python import cp_model

from src.linear_program.adult_placement import adult_presence_at_center
from src.linear_program.context import ModelContext, PersonCrewVars
from src.linear_program.leader_aggregates import crew_attribute_total
from src.models import Youth

ObjectiveTerm = cp_model.LinearExpr | int

YEARS: Final[tuple[str, ...]] = ('Fr', 'So', 'Jr', 'Sr')

# lcm(2, 4): scale per-(center, bucket) deviation so gender/history and year knobs match.
CENTER_DIVERSITY_BUCKET_NORM: Final[int] = 4


def proportional_integer_targets(bucket_total: int, center_crew_counts: Sequence[int]) -> list[int]:
    """Split ``bucket_total`` across centers proportional to crew counts.

    Uses largest remainder (Hamilton) so targets are integers summing exactly to ``bucket_total``.
    """
    n_centers = len(center_crew_counts)
    total_crews = sum(center_crew_counts)
    if n_centers == 0:
        return []
    if bucket_total <= 0 or total_crews == 0:
        return [0] * n_centers

    floors: list[int] = [(bucket_total * crews) // total_crews for crews in center_crew_counts]
    remainder = bucket_total - sum(floors)
    # Highest remainder `(bucket_total * crews_i) mod total_crews` receives +1 slots first.
    order = sorted(range(n_centers), key=lambda i: (bucket_total * center_crew_counts[i]) % total_crews, reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def _is_zero(value: cp_model.LinearExpr | int) -> bool:
    return isinstance(value, int) and value == 0


def _is_one(value: cp_model.LinearExpr | int) -> bool:
    return isinstance(value, int) and value == 1


def _same_center_term(
    model: cp_model.CpModel,
    youth_at_center: cp_model.LinearExpr | int,
    friend_at_center: cp_model.LinearExpr | int,
    weight: int,
    label: str,
) -> ObjectiveTerm | None:
    """Return ``weight * AND(youth_at_center, friend_at_center)`` as an objective term.

    Short-circuits when either side is the constant 0 or 1 to avoid building
    redundant Booleans for forced placements (e.g. Young Adult fixed crews).
    """
    if _is_zero(youth_at_center) or _is_zero(friend_at_center):
        return None
    if _is_one(youth_at_center):
        return cast(ObjectiveTerm, weight * friend_at_center)
    if _is_one(friend_at_center):
        return cast(ObjectiveTerm, weight * youth_at_center)

    same_center = model.NewBoolVar(label)
    model.Add(same_center <= youth_at_center)
    model.Add(same_center <= friend_at_center)
    model.Add(same_center >= youth_at_center + friend_at_center - 1)
    return cast(ObjectiveTerm, weight * same_center)


def add_friend_preference_objectives(ctx: ModelContext) -> list[ObjectiveTerm]:
    """Reward each youth landing at a center where their picked friend is also placed.

    Friend choices may name another youth (buddy roster) or an Adult / Young Adult.
    Youth-youth same-center bonuses use ``cfg.friend_weight``; youth-adult bonuses
    use ``cfg.adult_friend_weight`` (must be set).

    Weights:
        First choice: 3, Second: 2, Third: 1.
    """
    cfg = ctx.cfg
    adult_w = cfg.adult_friend_weight
    assert adult_w is not None

    objective_terms: list[ObjectiveTerm] = []
    weights = (3, 2, 1)

    for youth in ctx.youth_list:
        choices = (youth.first_choice, youth.second_choice, youth.third_choice)
        for friend, weight in zip(choices, weights):
            if friend is None:
                continue

            if friend in ctx.youth_dict:
                youth_w = cfg.friend_weight * weight
                for center in ctx.centers:
                    term = _same_center_term(
                        ctx.model,
                        ctx.at_center[(youth.name, center.name)],
                        ctx.at_center[(friend, center.name)],
                        youth_w,
                        f'same_center_{youth.name}_{friend}_{center.name}',
                    )
                    if term is not None:
                        objective_terms.append(term)
            elif friend in ctx.leaders_by_name:
                leader = ctx.leaders_by_name[friend]
                w = adult_w * weight
                for center in ctx.centers:
                    presence = adult_presence_at_center(leader, center, ctx.adult_crew)
                    if presence is None:
                        continue
                    term = _same_center_term(
                        ctx.model,
                        ctx.at_center[(youth.name, center.name)],
                        presence,
                        w,
                        f'same_center_adult_{youth.name}_{friend}_{center.name}',
                    )
                    if term is not None:
                        objective_terms.append(term)

    return objective_terms


def _crew_attribute_sum(
    person_crew: PersonCrewVars,
    youth_list: list[Youth],
    center_name: str,
    crew_name: str,
    attribute: str,
    value: str,
) -> cp_model.LinearExpr | int:
    """Sum of ``person_crew`` Booleans for youth with ``attribute == value`` on a crew."""
    terms = [
        person_crew[(y.name, center_name, crew_name)]
        for y in youth_list
        if getattr(y, attribute) == value and (y.name, center_name, crew_name) in person_crew
    ]
    if not terms:
        return 0
    return sum(terms)


def add_gender_diversity_objectives(ctx: ModelContext) -> list[ObjectiveTerm]:
    """Reward each crew having both M and F youth (``min(F_count, M_count)`` term)."""
    objective_terms: list[ObjectiveTerm] = []
    for center in ctx.centers:
        for crew in center.crews:
            females = _crew_attribute_sum(ctx.person_crew, ctx.regular_youth, center.name, crew.name, 'gender', 'F')
            males = _crew_attribute_sum(ctx.person_crew, ctx.regular_youth, center.name, crew.name, 'gender', 'M')
            if _is_zero(females) or _is_zero(males):
                continue
            balance = ctx.model.NewIntVar(0, ctx.cfg.max_crew_size, f'gender_balance_{center.name}_{crew.name}')
            ctx.model.Add(balance <= females)
            ctx.model.Add(balance <= males)
            objective_terms.append(cast(ObjectiveTerm, ctx.cfg.gender_weight * balance))
    return objective_terms


def add_year_diversity_objectives(ctx: ModelContext) -> list[ObjectiveTerm]:
    """Reward each crew having members from multiple years (Fr/So/Jr/Sr).

    One ``IntVar`` per crew counts how many of the four year-buckets are present.
    """
    objective_terms: list[ObjectiveTerm] = []
    for center in ctx.centers:
        for crew in center.crews:
            year_booleans: list[cp_model.IntVar] = []
            for year in YEARS:
                year_count = _crew_attribute_sum(ctx.person_crew, ctx.regular_youth, center.name, crew.name, 'year', year)
                if _is_zero(year_count):
                    continue
                has_year = ctx.model.NewBoolVar(f'has_{year}_{center.name}_{crew.name}')
                ctx.model.Add(year_count >= 1).OnlyEnforceIf(has_year)
                ctx.model.Add(year_count == 0).OnlyEnforceIf(has_year.Not())
                year_booleans.append(has_year)

            if not year_booleans:
                continue
            years_present = ctx.model.NewIntVar(0, len(YEARS), f'years_present_{center.name}_{crew.name}')
            ctx.model.Add(years_present == sum(year_booleans))
            objective_terms.append(cast(ObjectiveTerm, ctx.cfg.year_weight * years_present))

    return objective_terms


def add_history_diversity_objectives(ctx: ModelContext) -> list[ObjectiveTerm]:
    """Reward each crew having both Vet and New youth (``min(V, N)`` per crew)."""
    objective_terms: list[ObjectiveTerm] = []
    for center in ctx.centers:
        for crew in center.crews:
            vets = _crew_attribute_sum(ctx.person_crew, ctx.regular_youth, center.name, crew.name, 'history', 'V')
            new = _crew_attribute_sum(ctx.person_crew, ctx.regular_youth, center.name, crew.name, 'history', 'N')
            if _is_zero(vets) or _is_zero(new):
                continue
            balance = ctx.model.NewIntVar(0, ctx.cfg.max_crew_size, f'history_balance_{center.name}_{crew.name}')
            ctx.model.Add(balance <= vets)
            ctx.model.Add(balance <= new)
            objective_terms.append(cast(ObjectiveTerm, ctx.cfg.history_weight * balance))
    return objective_terms


def _center_bucket_sum(
    ctx: ModelContext,
    *,
    attribute: str,
    value: str,
    center_name: str,
) -> cp_model.LinearExpr | int:
    """Count regular youth at ``center_name`` with ``attribute == value``."""
    terms: list[cp_model.LinearExpr | int] = [ctx.at_center[(y.name, center_name)] for y in ctx.regular_youth if getattr(y, attribute) == value]
    if not terms:
        return 0
    if len(terms) == 1:
        return terms[0]
    return cast(cp_model.LinearExpr, sum(terms))


def _add_center_proportional_balance_terms(
    ctx: ModelContext,
    *,
    attribute: str,
    values: tuple[str, ...],
    weight: int,
    var_prefix: str,
) -> list[ObjectiveTerm]:
    """Penalize absolute deviation from crew-proportional targets per center and demographic bucket."""
    if weight <= 0:
        return []

    centers = ctx.centers
    bucket_norm = CENTER_DIVERSITY_BUCKET_NORM // len(values)
    if CENTER_DIVERSITY_BUCKET_NORM % len(values) != 0:
        raise ValueError(f'CENTER_DIVERSITY_BUCKET_NORM must be divisible by bucket count ({CENTER_DIVERSITY_BUCKET_NORM} % {len(values)} != 0)')

    objective_terms: list[ObjectiveTerm] = []
    center_crew_counts: list[int] = [len(c.crews) for c in centers]

    for v in values:
        total_bucket = sum(1 for y in ctx.regular_youth if getattr(y, attribute) == v)
        if total_bucket == 0:
            continue

        targets = proportional_integer_targets(total_bucket, center_crew_counts)

        for center, target_v_c in zip(centers, targets, strict=True):
            base_coef = weight * bucket_norm
            actual_v_c = _center_bucket_sum(ctx, attribute=attribute, value=v, center_name=center.name)
            deviation_ub = max(target_v_c, total_bucket - target_v_c)

            if deviation_ub <= 0:
                continue

            if isinstance(actual_v_c, int):
                dev_int = abs(actual_v_c - target_v_c)
                if dev_int <= 0:
                    continue
                objective_terms.append(cast(ObjectiveTerm, -base_coef * dev_int))
                continue

            deviation = ctx.model.NewIntVar(0, deviation_ub, f'{var_prefix}_{center.name}_{v}')
            delta = cast(cp_model.LinearExpr, actual_v_c - target_v_c)
            ctx.model.AddAbsEquality(deviation, delta)
            objective_terms.append(cast(ObjectiveTerm, -base_coef * deviation))

    return objective_terms


def add_center_gender_diversity_objectives(ctx: ModelContext) -> list[ObjectiveTerm]:
    """Penalize youth gender imbalance across centers versus proportional crew share."""
    return _add_center_proportional_balance_terms(
        ctx,
        attribute='gender',
        values=('M', 'F'),
        weight=ctx.cfg.center_gender_weight,
        var_prefix='center_gender_dev',
    )


def add_center_year_diversity_objectives(ctx: ModelContext) -> list[ObjectiveTerm]:
    """Penalize year-bucket imbalance across centers versus proportional crew share."""
    return _add_center_proportional_balance_terms(
        ctx,
        attribute='year',
        values=YEARS,
        weight=ctx.cfg.center_year_weight,
        var_prefix='center_year_dev',
    )


def add_center_history_diversity_objectives(ctx: ModelContext) -> list[ObjectiveTerm]:
    """Penalize vet/new imbalance across centers versus proportional crew share."""
    return _add_center_proportional_balance_terms(
        ctx,
        attribute='history',
        values=('V', 'N'),
        weight=ctx.cfg.center_history_weight,
        var_prefix='center_history_dev',
    )


# ----- Adult-side leader balance ----------------------------------------------


def _add_leader_min_balance_terms(
    ctx: ModelContext,
    *,
    attribute: str,
    value_a: str,
    value_b: str,
    weight: int,
    var_prefix: str,
    upper_bound: int,
) -> list[ObjectiveTerm]:
    """Generic ``min(count_a, count_b)`` per crew over leaders, weighted into the objective.

    Counts pre-assigned leaders as integers and flexible leaders via their
    ``adult_crew`` Booleans, so the solver can earn diversity by routing flexible
    adults to crews lacking the missing attribute.
    """
    objective_terms: list[ObjectiveTerm] = []
    for center in ctx.centers:
        for crew in center.crews:
            pre_a, flex_a = crew_attribute_total(ctx, crew, center.name, attribute=attribute, value=value_a)
            pre_b, flex_b = crew_attribute_total(ctx, crew, center.name, attribute=attribute, value=value_b)

            sum_a = pre_a + sum(flex_a)
            sum_b = pre_b + sum(flex_b)

            if isinstance(sum_a, int) and sum_a == 0:
                continue
            if isinstance(sum_b, int) and sum_b == 0:
                continue

            balance = ctx.model.NewIntVar(0, upper_bound, f'{var_prefix}_{center.name}_{crew.name}')
            ctx.model.Add(balance <= sum_a)
            ctx.model.Add(balance <= sum_b)
            objective_terms.append(cast(ObjectiveTerm, weight * balance))
    return objective_terms


def add_adult_leader_gender_objectives(ctx: ModelContext) -> list[ObjectiveTerm]:
    """Reward crews whose leadership has both M and F representation."""
    return _add_leader_min_balance_terms(
        ctx,
        attribute='gender',
        value_a='M',
        value_b='F',
        weight=ctx.cfg.adult_gender_weight,
        var_prefix='adult_gender_balance',
        upper_bound=ctx.cfg.max_adults_per_crew,
    )


def add_adult_leader_history_objectives(ctx: ModelContext) -> list[ObjectiveTerm]:
    """Reward crews whose leadership has both Vet and New representation."""
    return _add_leader_min_balance_terms(
        ctx,
        attribute='history',
        value_a='V',
        value_b='N',
        weight=ctx.cfg.adult_history_weight,
        var_prefix='adult_history_balance',
        upper_bound=ctx.cfg.max_adults_per_crew,
    )

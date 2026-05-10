"""Build the CP-SAT crew-assignment model.

The model is constructed sparsely: ``person_crew`` only contains Booleans for
``(youth, center, crew)`` triples a youth could legally land on, given parent
/ past-leader / sibling constraints. Constraints that would otherwise post
``person_crew[...] == 0`` for impossible cells (parent_center, past_leader)
are encoded directly in the variable set, so those constraint helpers no
longer exist.

A cached ``at_center`` map (``(name, center) -> LinearExpr | int``) is reused
across every constraint and objective that asks "is this person at this
center?". This removes the repeated ``sum(person_crew[name, c, k] for k ...)``
pattern that previously appeared in seven places.

All helpers receive a single ``ModelContext`` (built once here, passed through
unchanged); no helper needs to know about siblings of its parameters.
"""

from collections.abc import Sequence

from ortools.sat.python import cp_model

from src.config import Config
from src.linear_program.adult_placement import build_adult_buddy_placement_map
from src.linear_program.constraints import (
    add_one_crew_per_youth,
    assign_center_only_adults,
    assign_unassigned_adults,
    break_symmetry_unassigned_adults,
    enforce_anti_buddy_constraint,
    enforce_crew_headcount,
    enforce_driver_per_crew,
    enforce_friend_center_constraint,
    enforce_friend_separation_constraint,
    enforce_new_requires_vet,
    enforce_parent_crew_separation_constraint,
    enforce_sibling_center_constraint,
    enforce_sibling_crew_separation_constraint,
    enforce_supervision_group_limit,
)
from src.linear_program.context import (
    AdultCrewVars,
    AtCenterMap,
    EligibilityMap,
    ModelContext,
    PersonCrewVars,
)
from src.linear_program.objectives import (
    ObjectiveTerm,
    add_adult_leader_gender_objectives,
    add_adult_leader_history_objectives,
    add_friend_preference_objectives,
    add_gender_diversity_objectives,
    add_history_diversity_objectives,
    add_year_diversity_objectives,
)
from src.models import Center, Leader, Youth


def _ya_fixed_crew(youth_name: str, centers: list[Center]) -> tuple[str, str] | None:
    """Find the (center, crew) where a Young Adult is pre-assigned, or ``None``."""
    for center in centers:
        for crew in center.crews:
            if youth_name in crew.adult_names:
                return (center.name, crew.name)
    return None


def _build_adult_to_center(
    centers: list[Center],
    center_only_adults: Sequence[Leader] | None = None,
) -> dict[str, str]:
    """Map leader names to a center: pre-placed crew adults, then center-only adults.

    Parents on the crews CSV often have a fixed :class:`Center` but empty ``Crew``
    (algorithm assigns their crew). Those leaders are not in ``crew.adults`` until
    after the solve, but their ``fixed_center`` still pins where their children
    may be placed.
    """
    mapping: dict[str, str] = {}
    for center in centers:
        for crew in center.crews:
            for leader in crew.adults:
                mapping[leader.name] = center.name
    for leader in center_only_adults or []:
        if leader.fixed_center is None:
            continue
        mapping.setdefault(leader.name, leader.fixed_center)
    return mapping


def _compute_eligibility(
    youth_list: list[Youth],
    centers: list[Center],
    center_only_adults: Sequence[Leader] | None = None,
) -> EligibilityMap:
    """Return ``{youth_name: [(center, crew), ...]}`` of legal placements per youth.

    Eligibility prunes ``(center, crew)`` pairs that hard constraints would force
    to zero anyway:

    - **Parent center**: youth restricted to their parent's ``fixed_center``,
      whether the parent is pre-placed in a specific crew or only pinned to a
      center (the solver picks the crew).
    - **Pre-placed parent's crew**: pruned here because the parent already lives
      in ``crew.adults``. For parents whose crew is still a solver decision the
      same rule is enforced at runtime by
      :func:`enforce_parent_crew_separation_constraint`.
    - **Past leaders**: any crew containing one of ``past_leaders`` is excluded.
    - **Siblings**: each youth's eligible centers are intersected (transitively)
      with every sibling's, so the sibling-center constraint never has to compare
      across centers neither youth could reach.
    - **Young Adult** (``role == 'Young Adult'``): exactly one entry, the YA's
      pre-placed crew. Production buddy CSVs do not include YAs, so this branch
      mostly serves test fixtures that promote a Youth to a YA.

    An empty list is allowed (the solver then reports infeasibility via the
    ``add_one_crew_per_youth`` constraint).
    """
    youth_dict = {y.name: y for y in youth_list}
    adult_to_center = _build_adult_to_center(centers, center_only_adults)
    all_centers: frozenset[str] = frozenset(c.name for c in centers)

    allowed_centers: dict[str, set[str]] = {}
    for y in youth_list:
        if y.role == 'Young Adult':
            placement = _ya_fixed_crew(y.name, centers)
            if placement is None:
                raise ValueError(f'Young Adult {y.name!r} not found in any crew')
            allowed_centers[y.name] = {placement[0]}
            continue

        if y.parent_names_list:
            parent_centers: set[str] = set()
            for parent in y.parent_names_list:
                if parent not in adult_to_center:
                    raise ValueError(
                        f'Parent {parent!r} not found in any center for {y.name!r}'
                    )
                parent_centers.add(adult_to_center[parent])
            allowed_centers[y.name] = parent_centers
        else:
            allowed_centers[y.name] = set(all_centers)

    changed = True
    while changed:
        changed = False
        for y in youth_list:
            sibs = [s for s in y.siblings_list if s in youth_dict]
            if not sibs:
                continue
            shared = set(allowed_centers[y.name])
            for sib in sibs:
                shared &= allowed_centers[sib]
            if shared != allowed_centers[y.name]:
                allowed_centers[y.name] = shared
                changed = True

    eligibility: EligibilityMap = {}
    for y in youth_list:
        if y.role == 'Young Adult':
            placement = _ya_fixed_crew(y.name, centers)
            assert placement is not None
            eligibility[y.name] = [placement]
            continue

        parents = set(y.parent_names_list)
        past = set(y.past_leaders)
        keys: list[tuple[str, str]] = []
        for center in centers:
            if center.name not in allowed_centers[y.name]:
                continue
            for crew in center.crews:
                names = crew.adult_names
                if past & names:
                    continue
                if parents & names:
                    continue
                keys.append((center.name, crew.name))
        eligibility[y.name] = keys

    return eligibility


def _build_at_center_cache(
    person_crew: PersonCrewVars,
    youth_list: list[Youth],
    centers: list[Center],
) -> AtCenterMap:
    """Cache ``at_center[(name, center)]`` as a sum (or constant) of eligible vars.

    For a Young Adult with role flag set, the result at their fixed center is the
    integer ``1`` (presence is forced by ``add_one_crew_per_youth``); at every
    other center it is ``0``. For regular youth, it is either a ``LinearExpr``
    summing eligible Booleans at that center, or ``0`` when the center is fully
    excluded by eligibility.
    """
    by_center: dict[str, dict[str, list[cp_model.IntVar]]] = {
        c.name: {y.name: [] for y in youth_list} for c in centers
    }
    for (name, center_name, crew_name), var in person_crew.items():
        by_center[center_name][name].append(var)

    cache: AtCenterMap = {}
    for center in centers:
        for y in youth_list:
            terms = by_center[center.name][y.name]
            if not terms:
                cache[(y.name, center.name)] = 0
            elif len(terms) == 1:
                cache[(y.name, center.name)] = terms[0]
            else:
                cache[(y.name, center.name)] = sum(terms)
    return cache


def _build_person_crew(
    model: cp_model.CpModel,
    youth_list: list[Youth],
    eligibility: EligibilityMap,
) -> PersonCrewVars:
    """Allocate one BoolVar per eligible (youth, center, crew) triple."""
    person_crew: PersonCrewVars = {}
    for y in youth_list:
        for center_name, crew_name in eligibility[y.name]:
            person_crew[(y.name, center_name, crew_name)] = model.NewBoolVar(
                f'person_{y.name}_center_{center_name}_crew_{crew_name}'
            )
    return person_crew


def _build_adult_crew(
    model: cp_model.CpModel,
    centers: list[Center],
    centers_by_name: dict[str, Center],
    center_only_leaders: list[Leader],
    unassigned_leaders: list[Leader],
) -> AdultCrewVars:
    """Allocate adult_crew BoolVars: center-only restricted to one center, others everywhere."""
    adult_crew: AdultCrewVars = {}
    for leader in center_only_leaders:
        assert leader.fixed_center is not None
        center = centers_by_name[leader.fixed_center]
        for crew in center.crews:
            adult_crew[(leader.name, center.name, crew.name)] = model.NewBoolVar(
                f'adult_{leader.name}_center_{center.name}_crew_{crew.name}'
            )
    for leader in unassigned_leaders:
        for center in centers:
            for crew in center.crews:
                adult_crew[(leader.name, center.name, crew.name)] = model.NewBoolVar(
                    f'adult_{leader.name}_center_{center.name}_crew_{crew.name}'
                )
    return adult_crew


def create_crew_assignment_model(
    cfg: Config,
    youth_list: list[Youth],
    centers: list[Center],
    center_only_adults: Sequence[Leader] | None = None,
    unassigned_adults: Sequence[Leader] | None = None,
) -> tuple[cp_model.CpModel, PersonCrewVars, AdultCrewVars]:
    """Build the CP-SAT model for a crew-assignment run.

    ``Adult`` and ``YoungAdult`` instances are the source of truth for leader
    metadata (role / gender / history); the previous ``leader_info`` sidecar is
    gone. Pre-assigned leaders live inside ``crew.adults``; flexible leaders are
    passed via ``center_only_adults`` (CENTER_ONLY placement) and
    ``unassigned_adults`` (UNASSIGNED placement). Both pools may contain Adult
    or Young Adult rows; only ``role == 'Adult'`` satisfies the per-crew driver
    minimum.

    Returns ``(model, person_crew, adult_crew)``. ``person_crew`` is sparse: only
    eligible ``(youth, center, crew)`` triples have a Boolean — callers should
    use ``person_crew.get(key, 0)`` (or ``key in person_crew``) when iterating
    arbitrary triples.
    """
    flex_center_only: list[Leader] = list(center_only_adults) if center_only_adults else []
    flex_unassigned: list[Leader] = list(unassigned_adults) if unassigned_adults else []

    print(f'Youth count: {len(youth_list)}')
    print(f'Centers: {[c.name for c in centers]}')
    print(f'Total crews: {sum(len(c.crews) for c in centers)}')
    if flex_center_only:
        print(f'Center-only leaders: {len(flex_center_only)}')
    if flex_unassigned:
        print(f'Unassigned leaders: {len(flex_unassigned)}')

    model = cp_model.CpModel()

    centers_by_name = {c.name: c for c in centers}
    eligibility = _compute_eligibility(youth_list, centers, flex_center_only)
    person_crew = _build_person_crew(model, youth_list, eligibility)
    adult_crew = _build_adult_crew(
        model, centers, centers_by_name, flex_center_only, flex_unassigned
    )
    at_center = _build_at_center_cache(person_crew, youth_list, centers)

    ctx = ModelContext(
        cfg=cfg,
        model=model,
        youth_list=youth_list,
        regular_youth=[y for y in youth_list if y.role == 'Youth'],
        youth_dict={y.name: y for y in youth_list},
        centers=centers,
        centers_by_name=centers_by_name,
        center_only_adults=flex_center_only,
        unassigned_adults=flex_unassigned,
        leaders_by_name=build_adult_buddy_placement_map(centers, flex_center_only, flex_unassigned),
        person_crew=person_crew,
        adult_crew=adult_crew,
        at_center=at_center,
        eligibility=eligibility,
    )

    add_one_crew_per_youth(ctx)
    enforce_parent_crew_separation_constraint(ctx)
    enforce_sibling_center_constraint(ctx)
    enforce_sibling_crew_separation_constraint(ctx)
    enforce_friend_separation_constraint(ctx)
    enforce_friend_center_constraint(ctx)
    enforce_supervision_group_limit(ctx)
    enforce_anti_buddy_constraint(ctx)

    if flex_center_only:
        assign_center_only_adults(ctx)
    if flex_unassigned:
        assign_unassigned_adults(ctx)

    enforce_crew_headcount(ctx)
    enforce_driver_per_crew(ctx)
    enforce_new_requires_vet(ctx)
    break_symmetry_unassigned_adults(ctx)

    objective_terms: list[ObjectiveTerm] = [
        *add_friend_preference_objectives(ctx),
        *add_gender_diversity_objectives(ctx),
        *add_year_diversity_objectives(ctx),
        *add_history_diversity_objectives(ctx),
        *add_adult_leader_gender_objectives(ctx),
        *add_adult_leader_history_objectives(ctx),
    ]

    model.Maximize(sum(objective_terms))

    return model, person_crew, adult_crew


__all__: tuple[str, ...] = ('create_crew_assignment_model',)

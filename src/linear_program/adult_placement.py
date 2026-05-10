"""Map adult / Young Adult names to how their (center, crew) is decided in CP-SAT."""

from typing import cast

from ortools.sat.python import cp_model

from src.models import Center, Leader, PlacementMode


def build_adult_buddy_placement_map(
    centers: list[Center],
    center_only_leaders: list[Leader],
    unassigned_leaders: list[Leader],
) -> dict[str, Leader]:
    """Name-keyed map of every Adult / Young Adult appearing in any placement bucket.

    Replaces the old ``AdultBuddyPlacement`` sidecar — :class:`Adult` and
    :class:`YoungAdult` already encode their placement (``placement``,
    ``fixed_center``, ``fixed_crew``), so we just key the same instance by name.

    Raises:
        ValueError: Same name appears in incompatible buckets / fixed crews.
    """
    out: dict[str, Leader] = {}

    def merge(leader: Leader, *, bucket: str) -> None:
        existing = out.get(leader.name)
        if existing is None:
            out[leader.name] = leader
            return
        if (
            existing.placement == leader.placement
            and existing.fixed_center == leader.fixed_center
            and getattr(existing, 'fixed_crew', None) == getattr(leader, 'fixed_crew', None)
        ):
            return
        raise ValueError(
            f"Leader {leader.name!r} appears twice with different placement "
            f"(prev={existing.placement}/{existing.fixed_center}; new={bucket})"
        )

    for center in centers:
        for crew in center.crews:
            for leader in crew.adults:
                merge(leader, bucket=f"crew {crew.name!r}")

    for leader in center_only_leaders:
        merge(leader, bucket="center_only_leaders")

    for leader in unassigned_leaders:
        merge(leader, bucket="unassigned_leaders")

    return out


def adult_presence_at_center(
    leader: Leader,
    center: Center,
    adult_crew: dict[tuple[str, str, str], cp_model.IntVar],
) -> cp_model.LinearExpr | int | None:
    """Return whether ``leader`` is present at ``center``.

    Returns:
        ``None``: leader cannot be at ``center`` (fixed elsewhere).
        ``int(1)``: tautologically present (FIXED placement at this center).
        ``LinearExpr``: sum of ``adult_crew`` Booleans placing the flexible adult here.
        ``int(0)``: flexible adult has no Booleans at this center (impossible cell).
    """
    placement = leader.placement
    fixed_center = leader.fixed_center

    if placement == PlacementMode.FIXED:
        return 1 if center.name == fixed_center else None

    if fixed_center is not None and center.name != fixed_center:
        return None

    terms = [
        adult_crew[(leader.name, center.name, crew.name)]
        for crew in center.crews
        if (leader.name, center.name, crew.name) in adult_crew
    ]
    if not terms:
        return 0
    return cast(cp_model.LinearExpr, sum(terms))

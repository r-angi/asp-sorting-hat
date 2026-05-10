"""Load cleaned CSV data into typed domain objects.

Three public entry points consumed by ``main.py``, the test suite, and any
other production code:

- :func:`get_youth_from_buddy_form_df` — buddy roster CSV → ``list[Youth]``.
- :func:`get_centers_from_adults_df` — crews CSV → ``(centers, center_only,
  unassigned)`` with typed :class:`Adult` / :class:`YoungAdult` instances
  embedded inside ``crew.adults`` and partitioned by :class:`PlacementMode`.
- :func:`get_historical_youth_leaders` — historical-crews df → past-leader
  map keyed by youth name.

The validation helpers :func:`all_parents_are_valid` /
:func:`all_friends_are_valid` live alongside; they are called once per run
in ``main`` to fail fast on dangling references.

Year-specific raw-CSV cleaners (``clean_asp_buddies`` etc.) live in
``scripts/clean_raw.py``; historical merge utilities live in ``src/historical.py``.
"""

from collections.abc import Iterable

import polars as pl

from src.config import CenterConfig
from src.models import Adult, Center, Crew, Leader, PlacementMode, Youth, YoungAdult
from src.schema import ALLOWED_HISTORY, GENDER_NORMALIZATION


def _normalize_gender(raw: object) -> str | None:
    if raw is None:
        return None
    return GENDER_NORMALIZATION.get(str(raw).strip())


def _normalize_history(raw: object) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    return s if s in ALLOWED_HISTORY else None


def _row_role(row: dict[str, object]) -> str:
    return str(row.get('role') or '').strip()


def _build_leader(
    row: dict[str, object],
    *,
    placement: PlacementMode,
    fixed_center: str | None,
    fixed_crew: str | None,
) -> Leader:
    """Construct an :class:`Adult` or :class:`YoungAdult` from a crews-CSV row."""
    name = str(row['name'])
    gender = _normalize_gender(row.get('gender'))
    history = _normalize_history(row.get('history'))
    role = _row_role(row)

    if role == 'Young Adult':
        return YoungAdult(
            name=name,
            placement=placement,
            fixed_center=fixed_center,
            fixed_crew=fixed_crew,
            gender=gender,  # type: ignore[arg-type]
            history=history,  # type: ignore[arg-type]
        )

    return Adult(
        name=name,
        placement=placement,
        fixed_center=fixed_center,
        fixed_crew=fixed_crew,
        gender=gender,  # type: ignore[arg-type]
        history=history,  # type: ignore[arg-type]
    )


CrewLeaderIndex = dict[tuple[str, str], list[Leader]]


def _normalize_placement_columns(adult_crews: pl.DataFrame) -> pl.DataFrame:
    """Cast Center/Crew to strings so CSV numeric inference matches CLI names (e.g. ``1`` ↔ ``\"1\"``)."""
    return adult_crews.with_columns(
        pl.col('Center').cast(pl.Utf8, strict=False),
        pl.col('Crew').cast(pl.Utf8, strict=False),
    )


def _index_preassigned_leaders(adult_crews: pl.DataFrame) -> CrewLeaderIndex:
    """Group fully-pre-assigned (Center, Crew) leaders into a single dict.

    One pass over the dataframe replaces the previous O(centers * crews)
    repeated ``filter`` calls in :func:`_build_crew_leaders`.
    """
    fully_assigned = adult_crews.filter(
        pl.col('Center').is_not_null()
        & (pl.col('Center') != '')
        & pl.col('Crew').is_not_null()
        & (pl.col('Crew') != '')
    )
    index: CrewLeaderIndex = {}
    for row in fully_assigned.iter_rows(named=True):
        key = (str(row['Center']), str(row['Crew']))
        leader = _build_leader(
            row,
            placement=PlacementMode.FIXED,
            fixed_center=key[0],
            fixed_crew=key[1],
        )
        index.setdefault(key, []).append(leader)
    return index


def _build_centers_from_index(index: CrewLeaderIndex) -> list[Center]:
    """Build centers / crews from the precomputed (center, crew) -> leaders index."""
    by_center: dict[str, list[tuple[str, list[Leader]]]] = {}
    for (center_name, crew_name), leaders in index.items():
        by_center.setdefault(center_name, []).append((crew_name, leaders))
    return [
        Center(name=name, crews=[Crew(name=cn, adults=ls) for cn, ls in crews])
        for name, crews in by_center.items()
    ]


def _build_crews_from_index(index: CrewLeaderIndex, center_name: str) -> list[Crew]:
    """All crews for a center using the prebuilt index, preserving CSV order."""
    return [
        Crew(name=crew_name, adults=leaders)
        for (idx_center, crew_name), leaders in index.items()
        if idx_center == center_name
    ]


def _build_crews_for_center(
    index: CrewLeaderIndex,
    center_name: str,
    crew_count: int,
) -> list[Crew]:
    """Build ``crew_count`` crews under ``center_name`` (e.g. ``F01``, ``F02``, ...).

    Crews missing from the CSV get empty ``adults`` lists; the solver fills them.
    """
    prefix = center_name[0].upper()
    return [
        Crew(name=cn, adults=index.get((center_name, cn), []))
        for cn in (f'{prefix}{i:02d}' for i in range(1, crew_count + 1))
    ]


def get_centers_from_adults_df(
    adult_crews: pl.DataFrame,
    center_configs: list[CenterConfig] | None = None,
) -> tuple[list[Center], list[Leader], list[Leader]]:
    """Extract centers and flexible leaders from the crews df.

    Returns:
        ``(centers, center_only_leaders, unassigned_leaders)`` where:

        - ``centers``: pre-assigned (FIXED) leaders embedded in ``crew.adults`` as typed
          :class:`Adult` / :class:`YoungAdult` instances.
        - ``center_only_leaders``: :class:`Adult` / :class:`YoungAdult` instances with
          ``placement=CENTER_ONLY`` and ``fixed_center`` populated; the solver picks
          their crew.
        - ``unassigned_leaders``: :class:`Adult` / :class:`YoungAdult` instances with
          ``placement=UNASSIGNED``; the solver picks both center and crew.

    Raises:
        ValueError: If df centers are not a subset of configured centers.
    """
    adult_crews = _normalize_placement_columns(adult_crews)
    center_only_df = adult_crews.filter(
        pl.col('Center').is_not_null()
        & (pl.col('Center') != '')
        & (pl.col('Crew').is_null() | (pl.col('Crew') == ''))
    )
    center_only_list: list[Leader] = [
        _build_leader(
            row,
            placement=PlacementMode.CENTER_ONLY,
            fixed_center=str(row['Center']),
            fixed_crew=None,
        )
        for row in center_only_df.iter_rows(named=True)
    ]

    unassigned_df = adult_crews.filter(
        (pl.col('Center').is_null() | (pl.col('Center') == ''))
        & (pl.col('Crew').is_null() | (pl.col('Crew') == ''))
    )
    unassigned_list: list[Leader] = [
        _build_leader(
            row,
            placement=PlacementMode.UNASSIGNED,
            fixed_center=None,
            fixed_crew=None,
        )
        for row in unassigned_df.iter_rows(named=True)
    ]

    df_centers = {
        c for c in adult_crews['Center'].unique().to_list() if c is not None and c != ''
    }

    leader_index = _index_preassigned_leaders(adult_crews)

    if center_configs is None:
        return _build_centers_from_index(leader_index), center_only_list, unassigned_list

    config_names = {c.name for c in center_configs}
    invalid = df_centers - config_names
    if invalid:
        raise ValueError(f'Centers {invalid} in adults df not in configured list: {config_names}')

    centers: list[Center] = []
    for cfg in center_configs:
        crews = (
            _build_crews_for_center(leader_index, cfg.name, cfg.crew_count)
            if cfg.crew_count is not None
            else _build_crews_from_index(leader_index, cfg.name)
        )
        centers.append(Center(name=cfg.name, crews=crews))

    return centers, center_only_list, unassigned_list


def get_youth_from_buddy_form_df(youth: pl.DataFrame) -> list[Youth]:
    return [Youth(**row) for row in youth.iter_rows(named=True)]


def get_historical_youth_leaders(all_historical_crews: pl.DataFrame) -> dict[str, list[str]]:
    """Map each youth name to the list of adult leaders they crewed under historically.

    Deduplicates leader names per youth so a leader who crewed multiple years
    appears once in ``past_leaders`` (matters for eligibility filtering).
    """
    adult_df = all_historical_crews.filter(pl.col('is_adult')).drop('is_adult').rename({'name': 'adult_name'})
    youth_df = all_historical_crews.filter(~pl.col('is_adult')).drop('is_adult').rename({'name': 'youth_name'})
    youth_pairings_df = (
        youth_df.join(adult_df, on='crew_year', how='left')
        .group_by('youth_name')
        .agg(pl.col('adult_name').drop_nulls().unique().alias('adult_names'))
    )
    return {row['youth_name']: row['adult_names'] for row in youth_pairings_df.iter_rows(named=True)}


def all_parents_are_valid(youth_df: pl.DataFrame, adult_df: pl.DataFrame) -> bool:
    """Verify every youth's parent name exists in the adult crew list. Raises on missing."""
    adult_names = set(adult_df['name'].drop_nulls().to_list())
    missing_parents: list[str] = []
    for row in youth_df.iter_rows(named=True):
        parent_field = row['parent_name']
        if not parent_field:
            continue
        for parent_name in parent_field.split('|'):
            if parent_name and parent_name not in adult_names:
                missing_parents.append(f"{row['name']}'s parent {parent_name}")
    if missing_parents:
        raise ValueError(f'Missing parents in adult crews: {", ".join(missing_parents)}')
    return True


def all_friends_are_valid(youth_list: list[Youth], adult_leader_names: Iterable[str]) -> bool:
    """Verify every friend choice references a roster youth or pre-assigned leader."""
    valid_names = {youth.name for youth in youth_list} | set(adult_leader_names)
    missing_friends: list[str] = []

    for youth in youth_list:
        friend_choices = [youth.first_choice, youth.second_choice, youth.third_choice]
        for choice in friend_choices:
            if choice and choice not in valid_names:
                missing_friends.append(f"{youth.name}'s friend {choice}")

    if missing_friends:
        raise ValueError(f'Invalid friend choices found: {", ".join(missing_friends)}')
    return True

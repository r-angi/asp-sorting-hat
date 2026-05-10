"""Bundle of every input/derivation needed to build the CP-SAT model.

A single ``ModelContext`` flows through every constraint and objective helper,
replacing the previous 6-8 parameter signatures. All caches (``at_center``,
``leaders_by_name``, ``centers_by_name``, ``regular_youth``, ``youth_dict``)
are precomputed once during construction so helpers stay allocation-free.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from ortools.sat.python import cp_model

from src.config import Config
from src.models import Center, Leader, Youth

PersonCrewVars = dict[tuple[str, str, str], cp_model.IntVar]
AdultCrewVars = dict[tuple[str, str, str], cp_model.IntVar]
AtCenterMap = dict[tuple[str, str], cp_model.LinearExpr | int]
EligibilityMap = dict[str, list[tuple[str, str]]]


@dataclass(slots=True)
class ModelContext:
    """All state needed by constraint and objective helpers.

    Construct once at the top of ``create_crew_assignment_model`` after the
    sparse Booleans and ``at_center`` cache are built; pass to every helper.
    """

    cfg: Config
    model: cp_model.CpModel
    youth_list: list[Youth]
    regular_youth: list[Youth]
    youth_dict: dict[str, Youth]
    centers: list[Center]
    centers_by_name: dict[str, Center]
    center_only_adults: list[Leader]
    unassigned_adults: list[Leader]
    leaders_by_name: dict[str, Leader]
    person_crew: PersonCrewVars
    adult_crew: AdultCrewVars
    at_center: AtCenterMap
    eligibility: EligibilityMap

    def flex_adults(self) -> Iterator[Leader]:
        """Iterate every solver-placed leader (CENTER_ONLY then UNASSIGNED)."""
        yield from self.center_only_adults
        yield from self.unassigned_adults

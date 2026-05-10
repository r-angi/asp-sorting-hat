"""Domain models for the crew-assignment optimizer.

``Adult`` and ``YoungAdult`` carry their own placement metadata (role / gender /
history / fixed_center / fixed_crew); the previous ``LeaderInfo`` sidecar is
gone. ``Crew.adults`` is a typed list of ``Adult | YoungAdult`` rather than a
list of bare names, so constraint helpers can read ``.role`` /  ``.gender`` /
``.history`` directly off each leader without a name lookup.

``Youth`` only ever represents a regular roster youth; Young Adults from the
crews CSV are :class:`YoungAdult` instances and never appear in ``youth_list``.
"""

from enum import StrEnum
from functools import cached_property
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlacementMode(StrEnum):
    """How an adult's (center, crew) is decided.

    * FIXED: pre-assigned in the crews CSV (Center + Crew set).
    * CENTER_ONLY: Center set, Crew chosen by the solver.
    * UNASSIGNED: Center and Crew chosen by the solver.
    """

    FIXED = "fixed"
    CENTER_ONLY = "center_only"
    UNASSIGNED = "unassigned"


class Adult(BaseModel):
    """Crew adult / driver. ``placement`` describes how (center, crew) is decided."""

    model_config = ConfigDict(extra="ignore")

    name: str
    role: Literal["Adult"] = "Adult"
    placement: PlacementMode = PlacementMode.FIXED
    fixed_center: str | None = None
    fixed_crew: str | None = None
    gender: Literal["M", "F"] | None = None
    history: Literal["V", "N"] | None = None


class YoungAdult(BaseModel):
    """Buddy-form youth promoted to crew leader.

    Supports the same placement modes as :class:`Adult` (FIXED, CENTER_ONLY,
    UNASSIGNED). Carries the buddy-roster fields (year, friend choices) so they
    participate in same-center friend objectives, plus crew-leader metadata
    (gender / history) used by adult-side balance objectives. ``role == 'Young
    Adult'`` never satisfies the per-crew driver minimum (only ``role == 'Adult'``
    does).
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    role: Literal["Young Adult"] = "Young Adult"
    placement: PlacementMode = PlacementMode.FIXED
    fixed_center: str | None = None
    fixed_crew: str | None = None
    year: str | None = None
    gender: Literal["M", "F"] | None = None
    history: Literal["V", "N"] | None = None
    first_choice: str | None = None
    second_choice: str | None = None
    third_choice: str | None = None


type Leader = Adult | YoungAdult


class Youth(BaseModel):
    """Buddy-form roster entry for a regular youth participant."""

    model_config = ConfigDict(extra="ignore")

    name: str
    year: str
    gender: str
    history: str
    parent_name: str | None = None
    siblings: str | None = None
    first_choice: str | None = None
    second_choice: str | None = None
    third_choice: str | None = None
    past_leaders: list[str] = Field(default_factory=list)
    role: Literal["Youth"] = "Youth"
    supervision_group: str | None = None
    anti_buddy: str | None = None

    @cached_property
    def siblings_list(self) -> list[str]:
        return self.siblings.split('|') if self.siblings else []

    @cached_property
    def parent_names_list(self) -> list[str]:
        return self.parent_name.split('|') if self.parent_name else []

    @cached_property
    def anti_buddy_list(self) -> list[str]:
        return self.anti_buddy.split('|') if self.anti_buddy else []


class Crew(BaseModel):
    """A crew at a center, holding its (typed) leaders."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    adults: list[Leader] = Field(default_factory=list)

    @cached_property
    def adult_names(self) -> set[str]:
        """Set of leader names — useful for ``name in crew.adult_names`` membership."""
        return {a.name for a in self.adults}

    @cached_property
    def adult_by_name(self) -> dict[str, Leader]:
        """Name-to-leader lookup for metadata access without scanning the list."""
        return {a.name: a for a in self.adults}


class Center(BaseModel):
    """A trip center with one or more crews."""

    name: str
    crews: list[Crew] = Field(default_factory=list)

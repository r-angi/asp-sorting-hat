from dataclasses import dataclass
from typing import Self


@dataclass
class CenterConfig:
    """Configuration for a center's crew setup."""

    name: str
    crew_count: int | None = None  # None = extract from CSV

    @classmethod
    def parse(cls, spec: str) -> Self:
        """Parse 'CenterName' or 'CenterName:count' format."""
        if ':' in spec:
            name, count = spec.split(':', 1)
            return cls(name=name.strip(), crew_count=int(count.strip()))
        return cls(name=spec.strip())


@dataclass
class Config:
    """Configuration for crew assignment model."""

    # Crew size constraints
    min_crew_size: int = 5
    max_crew_size: int = 7
    min_adults_per_crew: int = 2
    max_adults_per_crew: int = 3

    # Objective weights
    friend_weight: int = 2  # Weight for friend preferences
    gender_weight: int = 1  # Weight for youth gender diversity
    year_weight: int = 1  # Weight for youth year diversity
    history_weight: int = 1  # Weight for youth vet/new diversity
    adult_gender_weight: int = 1  # Weight for adult-leader M/F balance per crew
    adult_history_weight: int = 1  # Weight for adult-leader vet/new balance per crew
    # Center-level proportional balance (absolute deviation penalties; see objectives module).
    center_gender_weight: int = 1
    center_year_weight: int = 1
    center_history_weight: int = 1
    # Scales friend + crew-/adult-objective terms vs center proportional penalties only; printed friend scores unchanged.
    center_balance_softness: int = 4
    # Same-center buddy preference toward an Adult/Young Adult on crews CSV (None mirrors friend_weight).
    adult_friend_weight: int | None = None

    # CP-SAT solver hyperparameters (used by main.py orchestration; tests can override).
    solver_max_time_seconds: float = 300.0
    solver_num_workers: int = 8
    solver_relative_gap_limit: float = 0.005
    solver_log_progress: bool = True

    def __post_init__(self) -> None:
        if self.adult_friend_weight is None:
            self.adult_friend_weight = self.friend_weight
        if self.center_balance_softness < 1:
            raise ValueError(f'center_balance_softness must be >= 1, got {self.center_balance_softness}')

    @classmethod
    def default(cls) -> Self:
        """Get default configuration."""
        return cls()

    @classmethod
    def with_high_friend_weight(cls) -> Self:
        """Configuration that prioritizes friend preferences."""
        return cls(
            friend_weight=4,
            adult_friend_weight=4,
            gender_weight=1,
            year_weight=1,
            history_weight=1,
        )

    @classmethod
    def with_high_diversity(cls) -> Self:
        """Configuration that prioritizes diversity metrics."""
        return cls(
            friend_weight=1,
            adult_friend_weight=1,
            gender_weight=2,
            year_weight=2,
            history_weight=2,
            center_gender_weight=2,
            center_year_weight=2,
            center_history_weight=2,
            adult_gender_weight=2,
            adult_history_weight=2,
        )

    @classmethod
    def with_fast(cls) -> Self:
        """Quick-feedback config for development; trades solution quality for wall-clock."""
        return cls(
            solver_max_time_seconds=60.0,
            solver_relative_gap_limit=0.05,
            solver_log_progress=False,
        )

    @classmethod
    def with_optimal(cls) -> Self:
        """Push solver toward proven optimality at the cost of wall-clock."""
        return cls(
            solver_max_time_seconds=1800.0,
            solver_relative_gap_limit=0.001,
        )

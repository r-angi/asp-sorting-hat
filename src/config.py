from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for crew assignment model."""

    # Crew size constraints
    min_crew_size: int = 5
    max_crew_size: int = 7

    # Objective weights
    friend_weight: int = 2  # Weight for friend preferences
    gender_weight: int = 1  # Weight for gender diversity
    year_weight: int = 1  # Weight for year diversity
    history_weight: int = 1  # Weight for vet/new diversity

    @classmethod
    def default(cls) -> 'Config':
        """Get default configuration."""
        return cls()

    @classmethod
    def with_high_friend_weight(cls) -> 'Config':
        """Configuration that prioritizes friend preferences."""
        return cls(
            friend_weight=4,
            gender_weight=1,
            year_weight=1,
            history_weight=1,
        )

    @classmethod
    def with_high_diversity(cls) -> 'Config':
        """Configuration that prioritizes diversity metrics."""
        return cls(
            friend_weight=1,
            gender_weight=2,
            year_weight=2,
            history_weight=2,
        )

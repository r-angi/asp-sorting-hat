"""Canonical schema constants shared across ingestion, validation, and loading.

Keeping role / gender / history / year vocabularies in one place avoids the
drift that previously existed between :mod:`src.crew_csv_normalize`,
:mod:`src.validate_clean_data`, and :mod:`src.data_loaders`.
"""

from typing import Final, Literal

Role = Literal["Adult", "Young Adult", "Youth"]
Gender = Literal["M", "F"]
History = Literal["V", "N"]
YearLevel = Literal["Fr", "So", "Jr", "Sr"]

ALLOWED_ROLES: Final[frozenset[str]] = frozenset({"Adult", "Young Adult", "Youth"})
ALLOWED_GENDER: Final[frozenset[str]] = frozenset({"M", "F"})
ALLOWED_HISTORY: Final[frozenset[str]] = frozenset({"V", "N"})
ALLOWED_YEAR: Final[frozenset[str]] = frozenset({"Fr", "So", "Jr", "Sr"})
ADULT_ROLES: Final[frozenset[str]] = frozenset({"Adult", "Young Adult"})

GENDER_NORMALIZATION: Final[dict[str, str]] = {
    "M": "M",
    "F": "F",
    "Male": "M",
    "Female": "F",
    "male": "M",
    "female": "F",
}

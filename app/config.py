import os
from dataclasses import dataclass


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MatchConfig:
    """All the knobs that control how forgiving book matching is.

    Every value can be overridden with an env var so thresholds can be tuned
    without touching code while we gather real-world evidence (see README).
    """

    # Composite match score (0..1) required to reach each status. A candidate
    # must also have at least one "distinctive" token overlap to reach
    # POSSIBLE_MATCH or HIGH_CONFIDENCE — see matcher.BookMatcher._score.
    high_confidence_threshold: float = _float_env("MATCH_HIGH_THRESHOLD", 0.75)
    possible_match_threshold: float = _float_env("MATCH_POSSIBLE_THRESHOLD", 0.55)
    low_confidence_threshold: float = _float_env("MATCH_LOW_THRESHOLD", 0.35)

    # Minimum status a Detection must reach before we create a Book/Sighting.
    min_status_to_persist: str = os.getenv("MATCH_MIN_STATUS_TO_PERSIST", "POSSIBLE_MATCH")

    # OCR-quality gate: below this, we never even call the metadata API.
    min_alpha_tokens: int = _int_env("OCR_MIN_ALPHA_TOKENS", 2)
    min_alpha_ratio: float = _float_env("OCR_MIN_ALPHA_RATIO", 0.35)

    # A token shorter than this (or in the common-word list) doesn't count as
    # "distinctive" evidence on its own.
    min_token_length_for_distinctive: int = _int_env("MATCH_DISTINCTIVE_MIN_LEN", 5)

    metadata_search_limit: int = _int_env("METADATA_SEARCH_LIMIT", 5)
    metadata_timeout_seconds: float = _float_env("METADATA_TIMEOUT_SECONDS", 4.0)


DEFAULT_CONFIG = MatchConfig()

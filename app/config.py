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


@dataclass(frozen=True)
class VisionConfig:
    """Knobs for shelf-row detection and per-shelf spine segmentation.

    All ratios are relative to the *shelf region*, not the whole image —
    that's the core fix for boxes spanning multiple shelves: spine detection
    runs independently inside each shelf's crop, so height constraints must
    be sized against that shelf's height, not the full photo's height.
    """

    # --- Shelf-row detection (see ShelfVision._detect_shelf_rows) ---
    # Stage A ("raw" candidates, red in debug overlay): loose row-strength
    # threshold on horizontal-edge density, for visual comparison only.
    shelf_raw_row_strength_threshold: float = _float_env("SHELF_RAW_ROW_THRESHOLD", 0.25)
    # Stage B ("filtered" candidates, blue): a row only counts as a shelf
    # boundary if it's part of a horizontal run at least this fraction of
    # the image width — kills book-cover/text edges, keeps shelf boards.
    shelf_min_line_width_ratio: float = _float_env("SHELF_MIN_LINE_WIDTH_RATIO", 0.5)
    shelf_line_cluster_gap_px: int = _int_env("SHELF_LINE_CLUSTER_GAP_PX", 10)
    # Minimum spacing enforced between two shelf boundaries (as a fraction of
    # image height); closer pairs get merged.
    shelf_min_spacing_ratio: float = _float_env("SHELF_MIN_SPACING_RATIO", 0.04)
    # A top/bottom edge region shorter than this fraction of the median
    # interior shelf height is treated as a partial/cropped sliver and
    # dropped rather than scanned for spines.
    shelf_min_edge_region_ratio: float = _float_env("SHELF_MIN_EDGE_REGION_RATIO", 0.5)
    shelf_count_warn_low: int = _int_env("SHELF_COUNT_WARN_LOW", 3)
    shelf_count_warn_high: int = _int_env("SHELF_COUNT_WARN_HIGH", 15)

    # --- Per-shelf spine detection (see ShelfVision._detect_spines_in_shelf) ---
    spine_min_height_ratio: float = _float_env("SPINE_MIN_HEIGHT_RATIO", 0.45)
    spine_max_height_ratio: float = _float_env("SPINE_MAX_HEIGHT_RATIO", 1.0)
    spine_min_width_ratio: float = _float_env("SPINE_MIN_WIDTH_RATIO", 0.012)
    spine_max_width_ratio: float = _float_env("SPINE_MAX_WIDTH_RATIO", 0.22)
    spine_min_aspect_ratio: float = _float_env("SPINE_MIN_ASPECT_RATIO", 1.8)
    # Vertical MORPH_CLOSE kernel height, as a fraction of shelf height
    # (bridges small gaps in a single spine's own edge trace).
    spine_morph_close_height_ratio: float = _float_env("SPINE_MORPH_CLOSE_HEIGHT_RATIO", 0.12)
    spine_morph_close_min_px: int = _int_env("SPINE_MORPH_CLOSE_MIN_PX", 8)
    # Diagnostic-only: warn (don't reject) when an accepted candidate's width
    # exceeds this fraction of its shelf's height — often means it's actually
    # multiple adjacent books merged into one box.
    spine_max_width_to_shelf_height_ratio: float = _float_env("SPINE_MAX_WIDTH_TO_SHELF_HEIGHT_RATIO", 0.4)


DEFAULT_VISION_CONFIG = VisionConfig()

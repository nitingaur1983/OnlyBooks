"""Two families of schemas:

1. Model-facing (`ModelBook`, `ModelOutput`): the structured-output schema sent
   to the VLM. Kept deliberately flat/simple so every provider can express it.
   We still re-validate every item ourselves — a provider "supporting" a schema
   is not a guarantee the output obeys it.
2. API-facing (`VLMDetection` etc.): what /analyze/vlm returns, always in
   original-image pixel coordinates. This contract is provider-independent.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Readability(str, Enum):
    CLEAR = "CLEAR"
    PARTIAL = "PARTIAL"
    UNREADABLE = "UNREADABLE"
    NOT_VISIBLE = "NOT_VISIBLE"  # e.g. no author printed / spine region hidden


class IdentificationStatus(str, Enum):
    VISIBLE_TEXT = "VISIBLE_TEXT"  # title read directly from printed text
    PARTIAL_TEXT = "PARTIAL_TEXT"  # some text read, title incomplete
    NO_READABLE_TEXT = "NO_READABLE_TEXT"  # book located, nothing legible


class Orientation(str, Enum):
    SPINE_OUT = "SPINE_OUT"
    COVER_OUT = "COVER_OUT"
    UNKNOWN = "UNKNOWN"


# --- 1. Model-facing schema -------------------------------------------------


class ModelBook(BaseModel):
    box_2d: list[int] | None = Field(
        description="[ymin, xmin, ymax, xmax] of this ONE physical book, integers normalized 0-1000 "
        "relative to the image you were given. null if you cannot locate it reliably."
    )
    visible_title: str | None = Field(description="Title exactly as printed and legible; null if not legible.")
    visible_author: str | None = Field(description="Author exactly as printed and legible; null if not legible.")
    raw_visible_text: str | None = Field(description="All legible text on this book, in reading order; null if none.")
    title_readability: Readability
    author_readability: Readability
    identification_status: IdentificationStatus
    orientation: Orientation
    inferred_title: str | None = Field(
        description="Only if the full identity is unambiguous from the visible text itself (e.g. a partially "
        "occluded but distinctive title). Never from neighbouring books or general appearance. Usually null."
    )
    uncertainty_notes: str | None = Field(description="Short note on ambiguity, occlusion, glare, cut-off; else null.")
    shelf_index: int | None = Field(description="0-based shelf row from the top of the image you were given; null if unclear.")


class ModelOutput(BaseModel):
    books: list[ModelBook]


# --- 2. API-facing schema ---------------------------------------------------


class BBox(BaseModel):
    """Original-image pixel coordinates: origin top-left, x right, y down."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height


class DuplicateRef(BaseModel):
    tile_id: int | None
    model_box_2d: list[int]
    bbox: BBox
    visible_title: str | None
    iou: float
    containment: float
    text_similarity: float | None


class VLMDetection(BaseModel):
    index: int
    bbox: BBox
    visible_title: str | None
    visible_author: str | None
    raw_visible_text: str | None
    title_readability: Readability
    author_readability: Readability
    identification_status: IdentificationStatus
    orientation: Orientation
    inferred_title: str | None
    uncertainty_notes: str | None
    # The model returns free-form numbers that are NOT calibrated
    # probabilities, so we never ask for one. Always null for now; use the
    # readability/identification categories instead.
    model_confidence: float | None = None
    # Shelf row as numbered by the model. Only meaningful in full mode; in
    # tiled mode a tile's "shelf 0" is just its own top row, so it's null
    # here and kept per-tile in `model_shelf_index_in_tile`.
    shelf_index: int | None
    model_shelf_index_in_tile: int | None = None
    tile_id: int | None  # tile that produced the kept detection (null in full mode)
    model_box_2d: list[int]  # untouched model coordinates (0..1000, tile/image-relative)
    warnings: list[str] = []
    merged_duplicates: list[DuplicateRef] = []
    # Tiled mode: every per-tile observation ("t<tile>:<index>") folded into
    # this physical book, and whether dedupe left an open question about it.
    # Full provenance is in dedupe_report.json.
    observation_ids: list[str] = []
    needs_review: bool = False
    review_reasons: list[str] = []

"""Pure geometry for the VLM pipeline: tiling, model-box → pixel conversion,
and cross-tile duplicate removal. No I/O, no provider calls — unit-tested
directly in tests/test_vlm.py.

Coordinate conventions
----------------------
- Original-image pixels: origin top-left, x right, y down, after EXIF
  orientation has been applied (same as what cv2.imread sees in vision.py).
- Model boxes: Gemini's `box_2d` = [ymin, xmin, ymax, xmax], integers 0..1000,
  relative to the exact image the model received (a tile, or the whole photo).
  Because they're normalized, downscaling an image before sending it doesn't
  change the mapping — only the region's original-pixel extent matters.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .schemas import BBox, DuplicateRef, Readability, VLMDetection

MODEL_COORD_MAX = 1000
# Models occasionally emit 1001/-1 style overshoot; clamp that, reject beyond.
COORD_TOLERANCE = 10
MIN_BOX_PX = 3


@dataclass(frozen=True)
class Tile:
    tile_id: int
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _axis_starts(length: int, tile: int, overlap: int) -> tuple[list[int], int]:
    """Evenly spaced starts so every tile has the same size and neighbours
    overlap by at least `overlap` px. Returns (starts, tile_extent)."""
    if length <= tile:
        return [0], length
    n = math.ceil((length - overlap) / (tile - overlap))
    if n == 1:
        return [0], length
    return [round(i * (length - tile) / (n - 1)) for i in range(n)], tile


def compute_tiles(image_width: int, image_height: int, tile_size: int, overlap: int) -> list[Tile]:
    """Deterministic overlapping grid over the image. Depends only on the
    image dimensions and the two parameters — never on image content."""
    if tile_size < 256:
        raise ValueError(f"tile_size must be >= 256 px (got {tile_size})")
    if not 0 <= overlap < tile_size:
        raise ValueError(f"tile_overlap must be in [0, tile_size) (got {overlap} for tile_size {tile_size})")

    xs, tile_w = _axis_starts(image_width, tile_size, overlap)
    ys, tile_h = _axis_starts(image_height, tile_size, overlap)
    tiles = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            tiles.append(Tile(len(tiles), row, col, x, y, tile_w, tile_h))
    return tiles


def full_image_region(image_width: int, image_height: int) -> Tile:
    return Tile(0, 0, 0, 0, 0, image_width, image_height)


class InvalidBox(ValueError):
    """Model box that can't be trusted as a location. `reason` is a short code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def model_box_to_pixels(box_2d: list[int] | None, region: Tile) -> tuple[BBox, list[str]]:
    """Convert a [ymin, xmin, ymax, xmax] 0..1000 box relative to `region`
    into original-image pixels. Raises InvalidBox instead of guessing."""
    if box_2d is None:
        raise InvalidBox("missing_box")
    if len(box_2d) != 4:
        raise InvalidBox("wrong_length", f"expected 4 values, got {len(box_2d)}")

    warnings: list[str] = []
    values = []
    for v in box_2d:
        if v < -COORD_TOLERANCE or v > MODEL_COORD_MAX + COORD_TOLERANCE:
            raise InvalidBox("out_of_range", f"{box_2d} outside 0..{MODEL_COORD_MAX}")
        clamped = min(max(v, 0), MODEL_COORD_MAX)
        if clamped != v:
            warnings.append("box_clamped_to_region")
        values.append(clamped)
    ymin, xmin, ymax, xmax = values
    if ymin >= ymax or xmin >= xmax:
        raise InvalidBox("inverted_or_empty", str(box_2d))

    x1 = region.x + round(xmin / MODEL_COORD_MAX * region.width)
    x2 = region.x + round(xmax / MODEL_COORD_MAX * region.width)
    y1 = region.y + round(ymin / MODEL_COORD_MAX * region.height)
    y2 = region.y + round(ymax / MODEL_COORD_MAX * region.height)
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        raise InvalidBox("too_small", f"{x2 - x1}x{y2 - y1}px")
    return BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1), sorted(set(warnings))


def validate_in_image(bbox: BBox, image_width: int, image_height: int) -> None:
    if bbox.x < 0 or bbox.y < 0 or bbox.x2 > image_width or bbox.y2 > image_height:
        raise InvalidBox("outside_image", f"{bbox.model_dump()} vs {image_width}x{image_height}")


# --- Duplicate removal ------------------------------------------------------


def _intersection(a: BBox, b: BBox) -> int:
    w = min(a.x2, b.x2) - max(a.x, b.x)
    h = min(a.y2, b.y2) - max(a.y, b.y)
    return max(w, 0) * max(h, 0)


def iou(a: BBox, b: BBox) -> float:
    inter = _intersection(a, b)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def containment(a: BBox, b: BBox) -> float:
    """Fraction of the smaller box covered by the larger one."""
    smaller = min(a.area, b.area)
    return _intersection(a, b) / smaller if smaller else 0.0


def _axis_iou(a1: int, a2: int, b1: int, b2: int) -> float:
    inter = max(min(a2, b2) - max(a1, b1), 0)
    union = max(a2, b2) - min(a1, b1)
    return inter / union if union else 0.0


def axis_alignment(a: BBox, b: BBox) -> float:
    """Best 1-D overlap along x or y. A spine cut by a tile edge keeps its
    x-extent (or y-extent) while losing length along the other axis."""
    return max(_axis_iou(a.x, a.x2, b.x, b.x2), _axis_iou(a.y, a.y2, b.y, b.y2))


def _norm_text(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def detection_text(d: VLMDetection) -> str:
    return _norm_text(d.visible_title) or _norm_text(d.raw_visible_text)


def text_similarity(a: VLMDetection, b: VLMDetection) -> float | None:
    """None when either side has no legible text (nothing to compare).
    Partial reads of one spine ("these tru" vs "these truths jill lepore")
    count as similar via the containment check."""
    ta, tb = detection_text(a), detection_text(b)
    if not ta or not tb:
        return None
    if ta in tb or tb in ta:
        return 1.0
    return SequenceMatcher(None, ta, tb).ratio()


_READABILITY_RANK = {Readability.CLEAR: 3, Readability.PARTIAL: 2, Readability.UNREADABLE: 1, Readability.NOT_VISIBLE: 0}


def _quality(d: VLMDetection) -> tuple[int, int]:
    # Prefer the better-read copy, then the larger (less truncated) box.
    return (_READABILITY_RANK[d.title_readability], d.bbox.area)


@dataclass(frozen=True)
class DedupeThresholds:
    iou: float = 0.5
    containment: float = 0.8
    axis_alignment: float = 0.7
    text_similarity: float = 0.5


def dedupe_across_tiles(
    detections: list[VLMDetection], thresholds: DedupeThresholds = DedupeThresholds()
) -> tuple[list[VLMDetection], list[dict]]:
    """Merge detections of the same physical book reported by overlapping tiles.

    Two detections merge only if ALL hold:
      - they come from different tiles (one tile's books are distinct books);
      - their boxes overlap: IoU >= iou, or containment >= containment with
        axis alignment >= axis_alignment (a book cut by a tile edge);
      - their text doesn't contradict: similarity >= text_similarity, or
        either side has no legible text.
    Same title alone never merges anything — two copies of a book on a shelf
    are two books, and they don't overlap in pixels.

    Returns (kept, decisions). `decisions` logs every merge and every
    overlapping pair that was deliberately NOT merged, for diagnostics.json.
    """
    ordered = sorted(detections, key=_quality, reverse=True)
    kept: list[VLMDetection] = []
    decisions: list[dict] = []

    for cand in ordered:
        best = None
        for keeper in kept:
            if cand.tile_id is None or keeper.tile_id is None or cand.tile_id == keeper.tile_id:
                continue
            ov = iou(cand.bbox, keeper.bbox)
            cont = containment(cand.bbox, keeper.bbox)
            align = axis_alignment(cand.bbox, keeper.bbox)
            geometric = ov >= thresholds.iou or (cont >= thresholds.containment and align >= thresholds.axis_alignment)
            if not geometric:
                continue
            sim = text_similarity(cand, keeper)
            info = {
                "candidate_tile": cand.tile_id,
                "candidate_bbox": cand.bbox.model_dump(),
                "candidate_text": detection_text(cand) or None,
                "keeper_tile": keeper.tile_id,
                "keeper_bbox": keeper.bbox.model_dump(),
                "keeper_text": detection_text(keeper) or None,
                "iou": round(ov, 3),
                "containment": round(cont, 3),
                "axis_alignment": round(align, 3),
                "text_similarity": None if sim is None else round(sim, 3),
            }
            if sim is not None and sim < thresholds.text_similarity:
                decisions.append({**info, "decision": "kept_both_text_differs"})
                continue
            if best is None or ov > best[1]["iou"]:
                best = (keeper, info)

        if best is None:
            kept.append(cand)
            continue
        keeper, info = best
        keeper.merged_duplicates.append(
            DuplicateRef(
                tile_id=cand.tile_id,
                model_box_2d=cand.model_box_2d,
                bbox=cand.bbox,
                visible_title=cand.visible_title,
                iou=info["iou"],
                containment=info["containment"],
                text_similarity=info["text_similarity"],
            )
        )
        keeper.merged_duplicates.extend(cand.merged_duplicates)
        decisions.append({**info, "decision": "merged"})

    kept.sort(key=lambda d: (d.bbox.y, d.bbox.x))
    for i, d in enumerate(kept):
        d.index = i
    return kept, decisions

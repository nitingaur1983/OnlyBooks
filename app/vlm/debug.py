"""Debug artifacts for /analyze/vlm, written under data/debug/vlm/<request_id>/.

Pillow is used for drawing and image I/O only — nothing here detects books.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .geometry import Tile
from .schemas import Readability, VLMDetection

logger = logging.getLogger(__name__)

GREEN = (0, 200, 0)
YELLOW = (255, 210, 0)
RED = (230, 30, 30)
TILE_GRID = (0, 160, 255)


def readability_color(r: Readability) -> tuple[int, int, int]:
    if r == Readability.CLEAR:
        return GREEN
    if r == Readability.PARTIAL:
        return YELLOW
    return RED


# Pillow's bundled default font lacks most non-ASCII glyphs (e.g. Polish
# "ł", "ż"), so prefer a system TrueType font with wide Unicode coverage.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _font(size: int):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list[VLMDetection],
    offset: tuple[int, int] = (0, 0),
    tiles: list[Tile] | None = None,
    label_tile: bool = True,
) -> Image.Image:
    """Draw each detection's box (shifted by -offset) colour-coded by title
    readability, labelled "<index> <title…> t<tile>". Font and line width
    scale with the image so narrow spines on a 4590px-wide photo stay
    inspectable without labels swamping neighbours."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    short_side = min(canvas.size)
    line_w = max(2, short_side // 900)
    font = _font(max(12, short_side // 160))
    ox, oy = offset

    for tile in tiles or []:
        draw.rectangle([tile.x, tile.y, tile.x + tile.width - 1, tile.y + tile.height - 1], outline=TILE_GRID, width=line_w)

    for d in detections:
        color = readability_color(d.title_readability)
        x1, y1 = d.bbox.x - ox, d.bbox.y - oy
        x2, y2 = x1 + d.bbox.width, y1 + d.bbox.height
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        label = str(d.index)
        if d.visible_title:
            title = d.visible_title if len(d.visible_title) <= 18 else d.visible_title[:17] + "…"
            label += f" {title}"
        if label_tile and d.tile_id is not None:
            label += f" t{d.tile_id}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lx = x1
        ly = y1 + line_w  # inside the top of the box: never spills onto the shelf above
        draw.rectangle([lx, ly, lx + tw + 4, ly + th + 4], fill=color)
        draw.text((lx + 2, ly + 2 - tb[1]), label, fill=(0, 0, 0), font=font)
    return canvas


class DebugWriter:
    """Writes files under `root`, or does nothing when root is None."""

    def __init__(self, root: Path | None):
        self.root = root
        self.files: dict[str, str] = {}
        if root:
            root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _path(self, rel: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_image(self, key: str, rel: str, image: Image.Image, quality: int = 90) -> None:
        """`key` ending in "[]" collects paths into a list (e.g. per-tile images)."""
        if not self.root:
            return
        try:
            path = self._path(rel)
            image.convert("RGB").save(path, "JPEG", quality=quality)
            if key.endswith("[]"):
                self.files.setdefault(key[:-2], []).append(str(path))
            else:
                self.files[key] = str(path)
        except Exception:  # debug output must never break the analysis
            logger.exception("failed to save debug image %s", rel)

    def save_json(self, key: str, rel: str, data) -> None:
        if not self.root:
            return
        try:
            path = self._path(rel)
            path.write_text(json.dumps(data, indent=2, default=str))
            self.files[key] = str(path)
        except Exception:
            logger.exception("failed to save debug json %s", rel)


# --- physical-book dedupe artifacts ------------------------------------------------

MERGED = (0, 200, 0)  # book confirmed from several observations
SINGLE = (0, 190, 230)  # book seen by one tile only
REVIEW = (255, 140, 0)  # book with an open dedupe question
AMBIGUOUS = (230, 0, 230)
SEPARATE = (230, 30, 30)
OBSERVATION = (150, 150, 150)


def _center(b: dict) -> tuple[float, float]:
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color, font) -> None:
    tb = draw.textbbox((0, 0), text, font=font)
    x, y = xy
    draw.rectangle([x, y, x + tb[2] - tb[0] + 4, y + tb[3] - tb[1] + 4], fill=color)
    draw.text((x + 2, y + 2 - tb[1]), text, fill=(0, 0, 0), font=font)


def _rect(draw, b: dict, color, width: int) -> None:
    draw.rectangle([b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]], outline=color, width=width)


def draw_deduplicated(image: Image.Image, books: list[VLMDetection], tiles: list[Tile] | None = None) -> Image.Image:
    """Final physical books: green = merged from several observations,
    cyan = single observation, orange = needs review. Label: "<index> <title…> ×<observations>"."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    short_side = min(canvas.size)
    line_w = max(2, short_side // 900)
    font = _font(max(12, short_side // 160))
    for tile in tiles or []:
        draw.rectangle([tile.x, tile.y, tile.x + tile.width - 1, tile.y + tile.height - 1], outline=TILE_GRID, width=1)
    for d in books:
        n = len(d.observation_ids) or 1
        color = REVIEW if d.needs_review else (MERGED if n > 1 else SINGLE)
        b = d.bbox.model_dump()
        _rect(draw, b, color, line_w)
        title = d.visible_title or "?"
        title = title if len(title) <= 18 else title[:17] + "…"
        _label(draw, (b["x"], b["y"] + line_w), f"{d.index} {title} ×{n}", color, font)
    return canvas


def draw_dedupe_review(image: Image.Image, report: dict, pair_decisions: list[dict], ambiguous_pairs: list[dict],
                       tiles: list[Tile] | None = None) -> Image.Image:
    """Every original observation (grey, labelled with its id) plus the
    decisions between them: green lines join observations merged into one
    book (green box = canonical box), magenta = ambiguous pair (not merged),
    red = heavily overlapping pair kept separate."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    short_side = min(canvas.size)
    line_w = max(2, short_side // 900)
    font_size = max(10, short_side // 220)
    font = _font(font_size)
    for tile in tiles or []:
        draw.rectangle([tile.x, tile.y, tile.x + tile.width - 1, tile.y + tile.height - 1], outline=TILE_GRID, width=1)

    boxes = {}
    for book in report["books"]:
        for o in book["observations"]:
            boxes[o["observation_id"]] = o["bbox"]
            _rect(draw, o["bbox"], OBSERVATION, 1)
    for p in pair_decisions:
        if p["decision"] == "separate" and p["intersection_over_smaller"] >= 0.5:
            draw.line([_center(boxes[p["a"]]), _center(boxes[p["b"]])], fill=SEPARATE, width=line_w)
    for book in report["books"]:
        ids = book["observation_ids"]
        if len(ids) > 1:
            _rect(draw, book["canonical_bbox"], MERGED, line_w)
            for x, y in zip(ids, ids[1:]):
                draw.line([_center(boxes[x]), _center(boxes[y])], fill=MERGED, width=line_w)
    for p in ambiguous_pairs:
        for x in (p["a"], p["b"]):
            _rect(draw, boxes[x], AMBIGUOUS, line_w)
        draw.line([_center(boxes[p["a"]]), _center(boxes[p["b"]])], fill=AMBIGUOUS, width=line_w * 2)
    for obs_id, b in sorted(boxes.items()):
        draw.text((b["x"] + 2, b["y"] + b["height"] - font_size - 4), obs_id, fill=(255, 255, 255), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return canvas


def draw_pair_crop(image: Image.Image, pair: dict, header: list[str], margin: int = 80) -> Image.Image:
    """Close-up of one pair decision for manual review: observation a in blue,
    b in orange, the zone both tiles saw in light grey, notes on top."""
    a, b = pair["a_bbox"], pair["b_bbox"]
    x1 = max(min(a["x"], b["x"]) - margin, 0)
    y1 = max(min(a["y"], b["y"]) - margin, 0)
    x2 = min(max(a["x"] + a["width"], b["x"] + b["width"]) + margin, image.width)
    y2 = min(max(a["y"] + a["height"], b["y"] + b["height"]) + margin, image.height)
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    draw = ImageDraw.Draw(crop)

    def shifted(box):
        return {"x": box["x"] - x1, "y": box["y"] - y1, "width": box["width"], "height": box["height"]}

    if pair.get("shared_zone"):
        _rect(draw, shifted(pair["shared_zone"]), (220, 220, 220), 1)
    _rect(draw, shifted(a), (40, 120, 255), 3)
    _rect(draw, shifted(b), (255, 140, 0), 3)

    font = _font(16)
    line_h = 22
    panel_w = max(crop.width, 640)
    out = Image.new("RGB", (panel_w, crop.height + line_h * len(header) + 8), (255, 255, 255))
    out.paste(crop, (0, line_h * len(header) + 8))
    hd = ImageDraw.Draw(out)
    for i, line in enumerate(header):
        hd.text((6, 4 + i * line_h), line[:110], fill=(0, 0, 0), font=font)
    return out


def write_dedupe_artifacts(debug: DebugWriter, image: Image.Image | None, books: list[VLMDetection], report: dict,
                           pair_decisions: list[dict], ambiguous_pairs: list[dict], tiles: list[Tile]) -> None:
    debug.save_json("dedupe_report_path", "dedupe_report.json", report)
    debug.save_json("ambiguous_pairs_path", "ambiguous_pairs.json", ambiguous_pairs)
    debug.save_json("pair_decisions_path", "pair_decisions.json", pair_decisions)
    if image is not None:
        debug.save_image("deduplicated_annotated_path", "deduplicated_annotated.jpg", draw_deduplicated(image, books, tiles))
        debug.save_image("dedupe_review_path", "dedupe_review.jpg",
                         draw_dedupe_review(image, report, pair_decisions, ambiguous_pairs, tiles))

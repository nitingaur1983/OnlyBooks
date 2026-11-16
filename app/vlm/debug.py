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

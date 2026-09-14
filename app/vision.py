from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .config import DEFAULT_VISION_CONFIG, VisionConfig

logger = logging.getLogger(__name__)


@dataclass
class SpineCandidate:
    """Public output shape — unchanged by the shelf-aware rewrite below."""
    index: int
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float


class RejectionReason(str, Enum):
    CROSSES_SHELF_BOUNDARY = "CROSSES_SHELF_BOUNDARY"
    TOO_TALL_FOR_SHELF = "TOO_TALL_FOR_SHELF"
    TOO_SHORT_FOR_SHELF = "TOO_SHORT_FOR_SHELF"
    TOO_WIDE = "TOO_WIDE"
    TOO_NARROW = "TOO_NARROW"
    BAD_ASPECT_RATIO = "BAD_ASPECT_RATIO"


@dataclass
class ShelfRegion:
    index: int
    y0: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass
class SpineBoxMeta:
    """An accepted spine candidate in GLOBAL image coordinates.

    Carries shelf_index/local_index purely for debug crop filenames and
    logging — SpineCandidate (the public/API shape) is unaffected.
    """
    x: int
    y: int
    width: int
    height: int
    shelf_index: int
    local_index: int


@dataclass
class ShelfDetectionResult:
    regions: list[ShelfRegion]
    boundaries: list[int]           # clustered/filtered boundary y-coords used to build regions
    raw_line_ys: list[int]          # loose/raw candidate line y-coords, pre-filter (debug only)
    dropped_edge_regions: list[str]
    warnings: list[str]


@dataclass
class DebugSummary:
    """Always populated (cheap: counts/lists), independent of whether debug
    images were written to disk. `debug_dir`/`saved_files` are only
    non-empty when a debug_dir was passed in."""

    debug_dir: str | None = None
    total_raw_contours: int = 0
    final_candidate_count: int = 0
    final_spine_candidates: int = 0
    shelf_row_count: int = 0
    shelf_boundaries: list[int] = field(default_factory=list)
    candidates_per_shelf: dict[int, int] | None = None
    rejection_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    saved_files: list[str] = field(default_factory=list)


@dataclass
class VisionResult:
    candidates: list[SpineCandidate]
    debug: DebugSummary


class _NullDebugWriter:
    """No-op sink used when debug output wasn't requested — saves nothing."""

    enabled = False
    debug_dir: Path | None = None

    def save(self, name: str, image: np.ndarray) -> None:
        pass


class _FileDebugWriter:
    """Writes each named intermediate/overlay/crop image under debug_dir.

    `name` may contain "/" (e.g. "shelves/shelf_0_gray", "crops/shelf_0_spine_000")
    — the subdirectory is created automatically.
    """

    enabled = True

    def __init__(self, debug_dir: Path):
        self.debug_dir = debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.saved_files: list[str] = []

    def save(self, name: str, image: np.ndarray) -> None:
        path = self.debug_dir / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        self.saved_files.append(str(path.relative_to(self.debug_dir)))


# --------------------------------------------------------------------------
# Overlay drawing helpers (pure functions — take images, return new images)
# --------------------------------------------------------------------------

def _add_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def _shade_regions(image: np.ndarray, regions: list[ShelfRegion]) -> np.ndarray:
    band_colors = [(60, 60, 200), (200, 150, 60)]
    shaded = image.copy()
    for i, region in enumerate(regions):
        cv2.rectangle(shaded, (0, region.y0), (image.shape[1], region.y1), band_colors[i % 2], -1)
    return cv2.addWeighted(shaded, 0.12, image, 0.88, 0)


def _draw_shelf_row_overlay(
    image: np.ndarray, raw_line_ys: list[int], boundaries: list[int], regions: list[ShelfRegion]
) -> np.ndarray:
    overlay = _shade_regions(image, regions)
    w = image.shape[1]

    for y in raw_line_ys:
        cv2.line(overlay, (0, y), (w, y), (0, 0, 255), 1)  # thin red: every raw candidate

    for y in boundaries:
        cv2.line(overlay, (0, y), (w, y), (255, 120, 0), 3)  # thick blue: filtered boundaries

    for region in regions:
        y_label = min(image.shape[0] - 10, region.y0 + 26)
        cv2.putText(overlay, f"shelf_{region.index}", (10, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def _draw_shelf_spine_overlay(
    image: np.ndarray, regions: list[ShelfRegion], accepted: list[SpineBoxMeta], rejected: list[dict]
) -> np.ndarray:
    overlay = _shade_regions(image, regions)

    for region in regions:
        cv2.line(overlay, (0, region.y0), (image.shape[1], region.y0), (255, 255, 255), 1)
        y_label = min(image.shape[0] - 10, region.y0 + 24)
        cv2.putText(overlay, f"shelf_{region.index}", (10, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    for r in rejected:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 1)

    for box in accepted:
        x, y, w, h = box.x, box.y, box.width, box.height
        aspect = h / max(w, 1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 0), 2)
        lines = [f"s{box.shelf_index}.{box.local_index}", f"({x},{y},{w},{h})", f"ar={aspect:.2f}"]
        text_y = max(12, y - 6 - 14 * (len(lines) - 1))
        for line in lines:
            cv2.putText(overlay, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
            text_y += 14
    return overlay


def _draw_annotated_overlay(image: np.ndarray, entries: list[dict]) -> np.ndarray:
    """Final OCR-result deliverable: every box the pipeline OCR'd, labeled
    with what it found (dims, aspect ratio, OCR text/confidence)."""
    overlay = image.copy()
    for e in entries:
        x, y, w, h = e["x"], e["y"], e["w"], e["h"]
        has_text = bool(e["text"])
        color = (0, 200, 0) if has_text else (0, 0, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        snippet = e["text"][:40].replace("\n", " ") if has_text else "<no text>"
        lines = [
            f"#{e['index']}",
            f"({x},{y},{w},{h})",
            f"ar={e['aspect']:.2f}",
            f"conf={e['confidence']:.0f}",
            snippet,
        ]
        text_y = max(12, y - 8 - 14 * (len(lines) - 1))
        for line in lines:
            cv2.putText(overlay, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            text_y += 14
    return overlay


class ShelfVision:
    """POC vision pipeline.

    Hierarchy: whole image -> detect shelf rows -> crop per shelf -> detect
    spines independently inside each shelf crop -> convert to global
    coordinates -> OCR each spine. Vertical-edge detection never runs across
    the whole image, so a spine candidate can't span more than one shelf row.

    This intentionally uses classical CV + Tesseract so there are no API keys
    or hosted-model dependencies. The interface can later be backed by
    YOLO/DETR/etc.
    """

    def __init__(self, config: VisionConfig | None = None):
        self.config = config or DEFAULT_VISION_CONFIG

    def detect_and_read(self, image_path: str | Path, debug_dir: Path | None = None) -> VisionResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Unable to read uploaded image")
        logger.info(
            "Processing image %s: width=%d height=%d channels=%s",
            image_path, image.shape[1], image.shape[0],
            image.shape[2] if len(image.shape) == 3 else 1,
        )

        debug = _FileDebugWriter(debug_dir) if debug_dir else _NullDebugWriter()

        boxes, summary = self._detect_spines(image, debug)
        results: list[SpineCandidate] = []
        annotated_entries: list[dict] = []
        logger.info("detected %d spine candidates in photo %s", len(boxes), image_path)

        for idx, box in enumerate(boxes):
            x, y, w, h = box.x, box.y, box.width, box.height
            crop = image[y:y + h, x:x + w]
            logger.info(
                "OCR candidate #%d (shelf %d, local %d): x=%d y=%d w=%d h=%d",
                idx, box.shelf_index, box.local_index, x, y, w, h,
            )
            text, conf = self._ocr_best_orientation(crop)
            text = text.strip()
            logger.info("OCR candidate #%d result: text=%r confidence=%.2f", idx, text, conf)

            if debug.enabled:
                # Exactly what got handed to _ocr_best_orientation() — this is
                # the ground truth for "what does Tesseract actually receive".
                debug.save(f"crops/shelf_{box.shelf_index}_spine_{box.local_index:03d}", crop)

            annotated_entries.append({
                "index": idx, "x": x, "y": y, "w": w, "h": h,
                "aspect": h / max(w, 1), "text": text, "confidence": conf,
            })

            if text:
                logger.info("Accepted OCR candidate #%d: %r", idx, text)
                results.append(SpineCandidate(index=idx, x=x, y=y, width=w, height=h, text=text, confidence=conf))
            else:
                logger.debug("Rejected OCR candidate #%d: no usable text", idx)

        # Fallback: if segmentation found nothing at all, OCR the whole shelf image once.
        if not results:
            logger.warning("No spine OCR results found; falling back to OCR on entire shelf image")
            text, conf = self._ocr_best_orientation(image)
            text = text.strip()
            logger.info("Whole-image OCR result: text=%r confidence=%.2f", text, conf)
            h, w = image.shape[:2]
            if text:
                results.append(SpineCandidate(0, 0, 0, w, h, text, conf))
                annotated_entries.append({
                    "index": 0, "x": 0, "y": 0, "w": w, "h": h,
                    "aspect": h / max(w, 1), "text": text, "confidence": conf,
                })

        summary.final_spine_candidates = len(results)

        if debug.enabled:
            debug.save("07_annotated", _draw_annotated_overlay(image, annotated_entries))
            summary.saved_files = debug.saved_files

        logger.info(
            "vision summary: total_raw_contours=%d final_candidate_count=%d "
            "final_spine_candidates=%d shelf_row_count=%d candidates_per_shelf=%s",
            summary.total_raw_contours, summary.final_candidate_count,
            summary.final_spine_candidates, summary.shelf_row_count, summary.candidates_per_shelf,
        )

        return VisionResult(candidates=results, debug=summary)

    # ----------------------------------------------------------------
    # Top-level: shelf detection, then per-shelf spine detection
    # ----------------------------------------------------------------

    def _detect_spines(self, image: np.ndarray, debug) -> tuple[list[SpineBoxMeta], DebugSummary]:
        config = self.config
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        shelf_result = self._detect_shelf_rows(gray)
        regions = shelf_result.regions

        logger.info(
            "Shelf row detection: image_height=%d raw_candidates=%d filtered_boundaries=%d shelf_count=%d",
            h, len(shelf_result.raw_line_ys), len(shelf_result.boundaries), len(regions),
        )
        for note in shelf_result.dropped_edge_regions:
            logger.info("Shelf row detection: %s", note)

        contours_overlay = image.copy()
        accepted_boxes: list[SpineBoxMeta] = []
        all_rejected: list[dict] = []
        per_shelf_metrics = []
        total_rejection_counts = {reason: 0 for reason in RejectionReason}

        for region in regions:
            accepted, rejected, metrics, local_contours = self._detect_spines_in_shelf(gray, region, w, debug)

            if local_contours:
                sub = contours_overlay[region.y0:region.y1, :]
                cv2.drawContours(sub, local_contours, -1, (0, 255, 255), 1)

            for i, (x, y, bw, bh) in enumerate(accepted):
                accepted_boxes.append(
                    SpineBoxMeta(x=x, y=y, width=bw, height=bh, shelf_index=region.index, local_index=i)
                )
            all_rejected.extend(rejected)
            per_shelf_metrics.append(metrics)
            for reason, count in metrics["rejection_counts"].items():
                total_rejection_counts[reason] += count

        accepted_boxes.sort(key=lambda b: (b.shelf_index, b.x))
        total_raw_contours = sum(m["raw_contours"] for m in per_shelf_metrics)

        # --- diagnostic sanity checks (Task 6) ---
        warnings = list(shelf_result.warnings)
        if len(regions) < config.shelf_count_warn_low:
            warnings.append(f"only {len(regions)} shelf(s) detected (< {config.shelf_count_warn_low}) -- shelf detection may have failed")
        if len(regions) > config.shelf_count_warn_high:
            warnings.append(f"{len(regions)} shelves detected (> {config.shelf_count_warn_high}) -- shelf detection may be too fragmented")

        region_by_index = {r.index: r for r in regions}
        for box in accepted_boxes:
            if box.y == 0 and box.height > 0.5 * h:
                warnings.append(
                    f"shelf {box.shelf_index} spine {box.local_index}: candidate starts at y=0 and "
                    f"spans >50% of image height ({box.height}px) -- possible cross-shelf artifact"
                )
            region = region_by_index[box.shelf_index]
            if box.width > config.spine_max_width_to_shelf_height_ratio * region.height:
                warnings.append(
                    f"shelf {box.shelf_index} spine {box.local_index}: width {box.width}px exceeds "
                    f"{config.spine_max_width_to_shelf_height_ratio:.0%} of shelf height ({region.height}px) "
                    "-- may contain multiple adjacent books"
                )

        for metrics in per_shelf_metrics:
            if metrics["accepted"] == 0:
                warnings.append(
                    f"shelf {metrics['shelf_index']}: 0 accepted candidates out of "
                    f"{metrics['raw_contours']} raw contours"
                )

        # --- structured summary log (Task 6) ---
        lines = ["Shelf detection summary:", f"  shelf_count={len(regions)}", f"  boundaries={shelf_result.boundaries}"]
        for metrics in per_shelf_metrics:
            rc = metrics["rejection_counts"]
            lines.append(f"Shelf {metrics['shelf_index']}:")
            lines.append(f"  height={metrics['height']}")
            lines.append(f"  raw_contours={metrics['raw_contours']}")
            lines.append(f"  accepted_spines={metrics['accepted']}")
            lines.append(f"  rejected_too_short={rc[RejectionReason.TOO_SHORT_FOR_SHELF]}")
            lines.append(f"  rejected_too_tall={rc[RejectionReason.TOO_TALL_FOR_SHELF]}")
            lines.append(f"  rejected_too_wide={rc[RejectionReason.TOO_WIDE]}")
            lines.append(f"  rejected_too_narrow={rc[RejectionReason.TOO_NARROW]}")
            lines.append(f"  rejected_bad_aspect={rc[RejectionReason.BAD_ASPECT_RATIO]}")
            lines.append(f"  rejected_crosses_boundary={rc[RejectionReason.CROSSES_SHELF_BOUNDARY]}")
        lines.append(f"Total accepted spine candidates={len(accepted_boxes)}")
        logger.info("\n".join(lines))

        for warning in warnings:
            logger.warning("Shelf/spine detection: %s", warning)

        if debug.enabled:
            debug.save("04_contours_all", _add_label(contours_overlay, f"raw contours: {total_raw_contours}"))
            debug.save("05_shelf_spine_candidates", _draw_shelf_spine_overlay(image, regions, accepted_boxes, all_rejected))
            debug.save("06_shelf_rows", _draw_shelf_row_overlay(image, shelf_result.raw_line_ys, shelf_result.boundaries, regions))
            debug.save("01_gray", gray)

        summary = DebugSummary(
            debug_dir=str(debug.debug_dir) if debug.enabled else None,
            total_raw_contours=total_raw_contours,
            final_candidate_count=len(accepted_boxes),
            shelf_row_count=len(regions),
            shelf_boundaries=list(shelf_result.boundaries),
            candidates_per_shelf={m["shelf_index"]: m["accepted"] for m in per_shelf_metrics},
            rejection_counts={reason.value: count for reason, count in total_rejection_counts.items()},
            warnings=warnings,
        )
        return accepted_boxes, summary

    # ----------------------------------------------------------------
    # Task 1: shelf-row detection
    # ----------------------------------------------------------------

    def _detect_shelf_rows(self, gray: np.ndarray) -> ShelfDetectionResult:
        """Best-effort horizontal shelf-boundary detection.

        Two passes over the same horizontal-edge map:
          Stage A (raw): a loose per-row edge-density threshold -- catches
            plenty of false positives (book covers, text baselines). Kept
            only for the debug overlay/count, never used for segmentation.
          Stage B (filtered): keeps only edges that are part of a horizontal
            run spanning a large fraction of the image width. Individual
            book/text edges are rarely that long; wooden shelf boards are.
        """
        config = self.config
        h, w = gray.shape[:2]

        sobel_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
        _, horiz_edges = cv2.threshold(sobel_y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Stage A: raw candidates (debug/comparison only).
        raw_strength = horiz_edges.sum(axis=1).astype(float)
        raw_line_ys: list[int] = []
        if raw_strength.max() > 0:
            norm = raw_strength / raw_strength.max()
            raw_rows = np.where(norm >= config.shelf_raw_row_strength_threshold)[0]
            raw_line_ys = self._cluster_rows(raw_rows, config.shelf_line_cluster_gap_px)

        # Stage B: sustained long horizontal runs only.
        min_run_px = max(1, int(w * config.shelf_min_line_width_ratio))
        long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_run_px, 1))
        sustained = cv2.morphologyEx(horiz_edges, cv2.MORPH_OPEN, long_kernel)
        sustained_rows = np.where(sustained.sum(axis=1) > 0)[0]
        filtered_line_ys = self._cluster_rows(sustained_rows, config.shelf_line_cluster_gap_px)

        min_spacing_px = max(1, int(h * config.shelf_min_spacing_ratio))
        filtered_line_ys = self._merge_close_lines(filtered_line_ys, min_spacing_px)

        warnings: list[str] = []
        dropped: list[str] = []

        if not filtered_line_ys:
            warnings.append("no confident shelf boundaries found; falling back to whole image as one shelf")
            return ShelfDetectionResult([ShelfRegion(index=0, y0=0, y1=h)], [], raw_line_ys, dropped, warnings)

        interior = [(filtered_line_ys[i], filtered_line_ys[i + 1]) for i in range(len(filtered_line_ys) - 1)]
        median_h = float(np.median([b - a for a, b in interior])) if interior else float(h)

        regions_bounds: list[tuple[int, int]] = []
        top_h = filtered_line_ys[0]
        if top_h >= median_h * config.shelf_min_edge_region_ratio:
            regions_bounds.append((0, filtered_line_ys[0]))
        else:
            dropped.append(
                f"top edge region dropped as a sliver (height={top_h}px < "
                f"{config.shelf_min_edge_region_ratio:.0%} of median shelf height {median_h:.0f}px)"
            )

        regions_bounds.extend(interior)

        bottom_h = h - filtered_line_ys[-1]
        if bottom_h >= median_h * config.shelf_min_edge_region_ratio:
            regions_bounds.append((filtered_line_ys[-1], h))
        else:
            dropped.append(
                f"bottom edge region dropped as a sliver (height={bottom_h}px < "
                f"{config.shelf_min_edge_region_ratio:.0%} of median shelf height {median_h:.0f}px)"
            )

        if not regions_bounds:
            warnings.append("all candidate shelf regions were dropped as slivers; falling back to whole image")
            regions_bounds = [(0, h)]

        regions = [ShelfRegion(index=i, y0=y0, y1=y1) for i, (y0, y1) in enumerate(regions_bounds)]
        return ShelfDetectionResult(regions, filtered_line_ys, raw_line_ys, dropped, warnings)

    @staticmethod
    def _cluster_rows(rows: np.ndarray, gap_px: int) -> list[int]:
        if len(rows) == 0:
            return []
        clusters = []
        start = prev = int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r - prev > gap_px:
                clusters.append((start + prev) // 2)
                start = r
            prev = r
        clusters.append((start + prev) // 2)
        return clusters

    @staticmethod
    def _merge_close_lines(lines: list[int], min_gap_px: int) -> list[int]:
        if not lines:
            return []
        merged = [lines[0]]
        for y in lines[1:]:
            if y - merged[-1] < min_gap_px:
                merged[-1] = (merged[-1] + y) // 2
            else:
                merged.append(y)
        return merged

    # ----------------------------------------------------------------
    # Task 2 + 3: per-shelf spine detection with explicit rejection reasons
    # ----------------------------------------------------------------

    def _detect_spines_in_shelf(
        self, gray_full: np.ndarray, region: ShelfRegion, image_w: int, debug
    ) -> tuple[list[tuple[int, int, int, int]], list[dict], dict, list]:
        """Runs vertical-edge spine detection strictly within [region.y0, region.y1).

        Returns (accepted_global_boxes, rejected_entries[global coords, for
        overlay], metrics, local_contours[for the aggregate contours-all view]).
        Because the crop is hard-bounded to this shelf, no accepted box can
        span more than one shelf row.
        """
        config = self.config
        shelf_h = region.height
        gray_crop = gray_full[region.y0:region.y1, :]

        sobel_x = cv2.Sobel(gray_crop, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel_h = int(np.clip(shelf_h * config.spine_morph_close_height_ratio, config.spine_morph_close_min_px, max(shelf_h, 1)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kernel_h))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_h = shelf_h * config.spine_min_height_ratio
        max_h = shelf_h * config.spine_max_height_ratio
        min_w = max(12, int(image_w * config.spine_min_width_ratio))
        max_w = int(image_w * config.spine_max_width_ratio)

        accepted_local: list[tuple[int, int, int, int]] = []
        rejected: list[dict] = []
        rejection_counts = {reason: 0 for reason in RejectionReason}

        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = ch / max(cw, 1)

            reason = None
            if cw < min_w:
                reason = RejectionReason.TOO_NARROW
            elif cw > max_w:
                reason = RejectionReason.TOO_WIDE
            elif ch < min_h:
                reason = RejectionReason.TOO_SHORT_FOR_SHELF
            elif ch > max_h:
                reason = RejectionReason.TOO_TALL_FOR_SHELF
            elif aspect < config.spine_min_aspect_ratio:
                reason = RejectionReason.BAD_ASPECT_RATIO

            if reason is not None:
                rejection_counts[reason] += 1
                rejected.append({"x": x, "y": region.y0 + y, "w": cw, "h": ch, "aspect": aspect, "reason": reason.value})
                continue

            pad_x = max(2, int(cw * 0.08))
            pad_y = max(2, int(ch * 0.02))
            x0 = max(0, x - pad_x)
            y0_local = max(0, y - pad_y)
            x1 = min(image_w, x + cw + pad_x)
            y1_local = min(shelf_h, y + ch + pad_y)  # hard-clipped to this shelf's own crop

            global_y0 = region.y0 + y0_local
            global_y1 = region.y0 + y1_local

            if global_y0 < region.y0 or global_y1 > region.y1:
                # Should be structurally impossible given the clip above --
                # kept as an explicit defensive check per spec.
                rejection_counts[RejectionReason.CROSSES_SHELF_BOUNDARY] += 1
                rejected.append({
                    "x": x0, "y": global_y0, "w": x1 - x0, "h": y1_local - y0_local,
                    "aspect": aspect, "reason": RejectionReason.CROSSES_SHELF_BOUNDARY.value,
                })
                logger.warning(
                    "shelf %d: candidate escaped its own shelf crop after clipping -- this should not happen",
                    region.index,
                )
                continue

            accepted_local.append((x0, y0_local, x1 - x0, y1_local - y0_local))

        accepted_local = self._dedupe(accepted_local)
        accepted_local.sort(key=lambda b: b[0])
        accepted_global = [(x, region.y0 + y, bw, bh) for (x, y, bw, bh) in accepted_local]

        metrics = {
            "shelf_index": region.index,
            "height": shelf_h,
            "raw_contours": len(contours),
            "accepted": len(accepted_global),
            "rejection_counts": rejection_counts,
        }

        if debug.enabled:
            debug.save(f"shelves/shelf_{region.index}_gray", gray_crop)
            debug.save(f"shelves/shelf_{region.index}_thresh", thresh)
            debug.save(f"shelves/shelf_{region.index}_closed", closed)

        return accepted_global, rejected, metrics, list(contours)

    @staticmethod
    def _iou(a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    def _dedupe(self, boxes):
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept = []
        for box in boxes:
            if all(self._iou(box, existing) < 0.45 for existing in kept):
                kept.append(box)
        return kept

    # ----------------------------------------------------------------
    # OCR (unchanged from before the shelf-aware rewrite)
    # ----------------------------------------------------------------

    def _ocr_best_orientation(self, crop: np.ndarray) -> tuple[str, float]:
        attempts = []
        for angle in (0, 90, 270):
            rotated = self._rotate(crop, angle)
            prepared = self._prepare_for_ocr(rotated)
            data = pytesseract.image_to_data(
                prepared,
                output_type=Output.DICT,
                config="--oem 3 --psm 6",
            )
            words, confs = [], []
            for text, conf in zip(data["text"], data["conf"]):
                text = text.strip()
                try:
                    conf_val = float(conf)
                except (TypeError, ValueError):
                    conf_val = -1
                if text and conf_val >= 0:
                    words.append(text)
                    confs.append(conf_val)
            joined = " ".join(words)
            avg = float(sum(confs) / len(confs)) if confs else 0.0
            attempts.append((joined, avg))

        # Prefer a result with both text volume and confidence.
        return max(attempts, key=lambda t: (t[1] * max(len(t[0]), 1), t[1]))

    @staticmethod
    def _prepare_for_ocr(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Upscaling materially helps text on narrow spines.
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        return gray

    @staticmethod
    def _rotate(image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

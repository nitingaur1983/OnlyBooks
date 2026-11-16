"""Orchestrates one /analyze/vlm request: load image → regions (full or tiles)
→ concurrent provider calls → parse/validate → pixel boxes → cross-tile
dedupe → summary, diagnostics and debug artifacts.

Independent of the OpenCV pipeline: nothing here imports app.vision,
pytesseract, cv2, the matcher, or the database.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError

from .config import VLMConfig, load_dedupe_config
from .debug import DebugWriter, draw_detections, write_dedupe_artifacts
from .dedupe import dedupe_physical_books
from .geometry import (
    InvalidBox,
    Tile,
    compute_tiles,
    full_image_region,
    model_box_to_pixels,
    validate_in_image,
)
from .prompts import FULL_IMAGE_PROMPT, SYSTEM_PROMPT, tile_prompt
from .provider import VLMProvider, VLMProviderError
from .schemas import IdentificationStatus, ModelBook, Orientation, Readability, VLMDetection

logger = logging.getLogger(__name__)

LARGE_BOX_FRACTION = 0.5


class AnalysisMode(str, Enum):
    full = "full"
    tiled = "tiled"


class VLMRequestError(Exception):
    """Client-side problem with the request (bad image, too many tiles…)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class VLMAnalysisError(Exception):
    """Every region failed. `payload` carries diagnostics + debug paths."""

    def __init__(self, status_code: int, message: str, payload: dict):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


@dataclass
class TileResult:
    region: Tile
    is_tile: bool
    status: str = "pending"  # ok | failed | skipped
    sent_size: tuple[int, int] | None = None
    error: dict | None = None
    finish_reason: str | None = None
    model_version: str | None = None
    usage: dict = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    raw_text: str | None = None
    raw_items: int = 0
    invalid_boxes: list[dict] = field(default_factory=list)
    item_errors: list[dict] = field(default_factory=list)
    detections: list[VLMDetection] = field(default_factory=list)

    @property
    def tile_id(self) -> int | None:
        return self.region.tile_id if self.is_tile else None

    def diagnostics(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "region": self.region.as_dict(),
            "sent_size": self.sent_size,
            "status": self.status,
            "error": self.error,
            "finish_reason": self.finish_reason,
            "model_version": self.model_version,
            "usage": self.usage,
            "attempts": self.attempts,
            "timings_ms": self.timings,
            "raw_items_returned": self.raw_items,
            "valid_detections": len(self.detections),
            "invalid_boxes": self.invalid_boxes,
            "item_errors": self.item_errors,
        }


# --- image preparation --------------------------------------------------------


def load_image(path: Path, config: VLMConfig) -> Image.Image:
    try:
        with Image.open(path) as probe:
            w, h = probe.size
            if w * h > config.max_image_pixels:
                raise VLMRequestError(
                    413, f"Image is {w}x{h} ({w * h:,} px), above VLM_MAX_IMAGE_PIXELS={config.max_image_pixels:,}."
                )
            image = ImageOps.exif_transpose(probe)  # same orientation cv2.imread uses
            image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise VLMRequestError(400, f"Could not decode image: {exc}") from exc
    return image


def encode_for_model(image: Image.Image, region: Tile, max_dim: int) -> tuple[bytes, Image.Image]:
    crop = image.crop((region.x, region.y, region.x + region.width, region.y + region.height))
    if max(crop.size) > max_dim:
        crop.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=92)
    return buf.getvalue(), crop


# --- parsing --------------------------------------------------------------------


def parse_model_output(result: TileResult, image_w: int, image_h: int) -> None:
    """Fill result.detections from result.raw_text. Malformed items and boxes
    are rejected individually and recorded; nothing is invented."""
    t0 = time.perf_counter()
    text = (result.raw_text or "").strip()
    if not text:
        kind = "blocked" if result.finish_reason in {"SAFETY", "PROHIBITED_CONTENT", "IMAGE_SAFETY"} else "empty_response"
        result.status, result.error = "failed", {"kind": kind, "message": f"model returned no text (finish_reason={result.finish_reason})"}
        result.timings["json_parsing_ms"] = _ms(t0)
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        kind = "output_truncated" if result.finish_reason == "MAX_TOKENS" else "invalid_json"
        result.status, result.error = "failed", {"kind": kind, "message": str(exc)}
        result.timings["json_parsing_ms"] = _ms(t0)
        return
    items = payload.get("books") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        result.status, result.error = "failed", {"kind": "unexpected_structure", "message": "expected an object with a 'books' list"}
        result.timings["json_parsing_ms"] = _ms(t0)
        return

    result.raw_items = len(items)
    books: list[tuple[int, ModelBook]] = []
    for i, item in enumerate(items):
        try:
            books.append((i, ModelBook.model_validate(item)))
        except ValidationError as exc:
            result.item_errors.append({"item": i, "reason": "schema_validation", "detail": exc.errors(include_url=False), "raw": item})
    result.timings["json_parsing_ms"] = _ms(t0)

    t1 = time.perf_counter()
    region = result.region
    for i, book in books:
        try:
            bbox, warnings = model_box_to_pixels(book.box_2d, region)
            validate_in_image(bbox, image_w, image_h)
        except InvalidBox as exc:
            result.invalid_boxes.append({"item": i, "reason": exc.reason, "detail": str(exc), "model_box_2d": book.box_2d,
                                         "visible_title": book.visible_title})
            continue

        title_readability = book.title_readability
        if book.visible_title is None and title_readability in (Readability.CLEAR, Readability.PARTIAL):
            # Never report a title as readable when no title text was given.
            warnings.append(f"title_readability_downgraded_from_{title_readability.value}")
            title_readability = Readability.UNREADABLE
        if bbox.area > LARGE_BOX_FRACTION * region.width * region.height and book.orientation != Orientation.COVER_OUT:
            warnings.append("box_covers_most_of_region_may_span_multiple_books")

        result.detections.append(
            VLMDetection(
                index=len(result.detections),
                bbox=bbox,
                visible_title=book.visible_title,
                visible_author=book.visible_author,
                raw_visible_text=book.raw_visible_text,
                title_readability=title_readability,
                author_readability=book.author_readability,
                identification_status=book.identification_status,
                orientation=book.orientation,
                inferred_title=book.inferred_title,
                uncertainty_notes=book.uncertainty_notes,
                shelf_index=None if result.is_tile else book.shelf_index,
                model_shelf_index_in_tile=book.shelf_index if result.is_tile else None,
                tile_id=result.tile_id,
                model_box_2d=book.box_2d,
                warnings=warnings,
            )
        )
    result.timings["bbox_processing_ms"] = _ms(t1)
    result.status = "ok"


# --- orchestration --------------------------------------------------------------


def _summary(detections: list[VLMDetection]) -> dict:
    def count(pred):
        return sum(1 for d in detections if pred(d))

    return {
        "total_detections": len(detections),
        "readable_titles": count(lambda d: d.title_readability == Readability.CLEAR),
        "partial_titles": count(lambda d: d.title_readability == Readability.PARTIAL),
        "unreadable_titles": count(lambda d: d.title_readability in (Readability.UNREADABLE, Readability.NOT_VISIBLE)),
        "with_visible_author": count(lambda d: d.visible_author is not None),
        "no_readable_text": count(lambda d: d.identification_status == IdentificationStatus.NO_READABLE_TEXT),
        "spine_out": count(lambda d: d.orientation == Orientation.SPINE_OUT),
        "cover_out": count(lambda d: d.orientation == Orientation.COVER_OUT),
    }


def _token_totals(results: list[TileResult], config: VLMConfig) -> dict:
    def total(key):
        values = [r.usage.get(key) for r in results if r.usage.get(key) is not None]
        return sum(values) if values else None

    totals = {k: total(k) for k in ("prompt_tokens", "output_tokens", "thinking_tokens", "total_tokens")}
    cost = None
    if config.price_input_per_mtok is not None and config.price_output_per_mtok is not None and totals["prompt_tokens"] is not None:
        billed_output = (totals["output_tokens"] or 0) + (totals["thinking_tokens"] or 0)  # thinking billed as output
        cost = round(totals["prompt_tokens"] / 1e6 * config.price_input_per_mtok + billed_output / 1e6 * config.price_output_per_mtok, 6)
    totals["estimated_cost_usd"] = cost
    totals["cost_note"] = (
        "estimate from VLM_PRICE_INPUT_PER_MTOK / VLM_PRICE_OUTPUT_PER_MTOK (thinking tokens billed as output)"
        if cost is not None else "set VLM_PRICE_INPUT_PER_MTOK and VLM_PRICE_OUTPUT_PER_MTOK (USD per 1M tokens) for an estimate"
    )
    return totals


def _http_status_for(errors: list[dict]) -> int:
    kinds = {e.get("kind") for e in errors}
    if kinds and kinds <= {"rate_limited"}:
        return 429
    if kinds and kinds <= {"timeout"}:
        return 504
    return 502


async def analyze_image(
    image_path: Path,
    *,
    request_id: str,
    mode: AnalysisMode,
    config: VLMConfig,
    provider: VLMProvider,
    debug_dir: Path | None,
    tile_size: int | None = None,
    tile_overlap: int | None = None,
) -> dict:
    t_total = time.perf_counter()
    log = logger
    prefix = f"[vlm {request_id}]"

    t0 = time.perf_counter()
    image = await asyncio.to_thread(load_image, image_path, config)
    image_load_ms = _ms(t0)
    img_w, img_h = image.size
    log.info("%s image %dx%d provider=%s model=%s mode=%s", prefix, img_w, img_h, provider.name, provider.model, mode.value)

    t0 = time.perf_counter()
    tiling = None
    if mode == AnalysisMode.full:
        regions = [full_image_region(img_w, img_h)]
    else:
        size = tile_size or config.tile_size
        overlap = config.tile_overlap if tile_overlap is None else tile_overlap
        try:
            regions = compute_tiles(img_w, img_h, size, overlap)
        except ValueError as exc:
            raise VLMRequestError(400, str(exc)) from exc
        if len(regions) > config.max_tiles:
            raise VLMRequestError(
                400,
                f"{len(regions)} tiles needed at tile_size={size}, tile_overlap={overlap} for a {img_w}x{img_h} image, "
                f"above VLM_MAX_TILES={config.max_tiles}. Use a larger tile_size or raise VLM_MAX_TILES.",
            )
        tiling = {"tile_size": size, "tile_overlap": overlap, "rows": regions[-1].row + 1, "cols": regions[-1].col + 1}
    tiling_ms = _ms(t0)
    is_tiled = mode == AnalysisMode.tiled
    log.info("%s %d region(s) %s", prefix, len(regions), tiling or "")

    debug = DebugWriter(debug_dir)
    results = [TileResult(region=r, is_tile=is_tiled) for r in regions]
    sent_images: dict[int, Image.Image] = {}
    semaphore = asyncio.Semaphore(config.max_concurrency)
    abort = asyncio.Event()
    abort_error: dict = {}
    t_inference = time.perf_counter()

    async def run(result: TileResult) -> None:
        region = result.region
        tp = time.perf_counter()
        data, sent = await asyncio.to_thread(encode_for_model, image, region, config.max_send_dimension)
        result.timings["image_preparation_ms"] = _ms(tp)
        result.sent_size = sent.size
        sent_images[region.tile_id] = sent
        prompt = tile_prompt(region.tile_id, region.row, region.col, tiling["rows"], tiling["cols"]) if is_tiled else FULL_IMAGE_PROMPT

        tq = time.perf_counter()
        async with semaphore:
            result.timings["queue_wait_ms"] = _ms(tq)
            if abort.is_set():
                result.status, result.error = "skipped", {"kind": "aborted", "message": f"skipped after permanent error: {abort_error.get('kind')}"}
                return
            log.info("%s region %s %dx%d (sent %dx%d, %d KB) -> %s", prefix, result.tile_id, region.width, region.height,
                     sent.size[0], sent.size[1], len(data) // 1024, provider.model)
            try:
                resp = await provider.analyze(data, "image/jpeg", prompt, SYSTEM_PROMPT)
            except VLMProviderError as exc:
                result.status, result.error, result.attempts = "failed", exc.as_dict(), exc.attempts
                result.timings["api_ms"] = round(sum(a["duration_ms"] for a in exc.attempts), 1)
                log.error("%s region %s failed: %s (%s)", prefix, result.tile_id, exc.kind, str(exc)[:200])
                if exc.aborts_request:
                    abort_error.update(exc.as_dict())
                    abort.set()
                return

        result.raw_text, result.finish_reason, result.model_version = resp.text, resp.finish_reason, resp.model_version
        result.usage, result.attempts = resp.usage, resp.attempts
        result.timings["api_ms"] = resp.api_ms
        await asyncio.to_thread(parse_model_output, result, img_w, img_h)
        log.info("%s region %s api=%.0fms finish=%s items=%d valid=%d invalid_boxes=%d tokens=%s status=%s",
                 prefix, result.tile_id, resp.api_ms, resp.finish_reason, result.raw_items, len(result.detections),
                 len(result.invalid_boxes), resp.usage, result.status)
        for bad in result.invalid_boxes:
            log.warning("%s region %s invalid box: %s %s", prefix, result.tile_id, bad["reason"], bad["model_box_2d"])

    await asyncio.gather(*(run(r) for r in results))
    inference_wall_ms = _ms(t_inference)

    # Snapshot per-tile detections (tile-local numbering) before dedupe mutates them.
    per_tile = {r.region.tile_id: [d.model_copy(deep=True) for d in r.detections] for r in results}
    all_detections = [d for r in results for d in r.detections]

    t0 = time.perf_counter()
    dedupe = None
    if is_tiled:
        dedupe = dedupe_physical_books(all_detections, regions, load_dedupe_config())
        detections = dedupe.books
    else:
        detections = sorted(all_detections, key=lambda d: (d.bbox.y, d.bbox.x))
        for i, d in enumerate(detections):
            d.index = i
    dedupe_ms = _ms(t0)
    duplicates_removed = len(all_detections) - len(detections)

    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]
    errors = [{"tile_id": r.tile_id, **(r.error or {})} for r in failed]
    summary = {
        **_summary(detections),
        "detections_before_dedupe": len(all_detections),
        "duplicates_removed": duplicates_removed,
        "books_needing_review": sum(1 for d in detections if d.needs_review),
        "ambiguous_pairs": len(dedupe.ambiguous_pairs) if dedupe else 0,
        "invalid_boxes_rejected": sum(len(r.invalid_boxes) for r in results),
        "items_failed_validation": sum(len(r.item_errors) for r in results),
        "regions_total": len(results),
        "regions_succeeded": len(ok),
        "regions_failed": len(failed),
    }
    warnings = []
    if ok and not detections:
        warnings.append("model_returned_no_valid_detections")
    if failed and ok:
        warnings.append("partial_results_some_regions_failed")
    if config.persist_results:
        warnings.append("VLM_PERSIST_RESULTS=true ignored: VLM persistence is not implemented in this phase")

    tokens = _token_totals(results, config)

    # --- debug artifacts ---
    t0 = time.perf_counter()
    if debug.enabled:
        def write_debug():
            debug.save_image("original_image_path", "original.jpg", image)
            debug.save_image("annotated_image_path", "annotated.jpg",
                             draw_detections(image, detections, tiles=regions if is_tiled else None))
            if is_tiled:
                for r in results:
                    tid = r.region.tile_id
                    if tid in sent_images:
                        debug.save_image("tile_image_paths[]", f"tiles/tile_{tid:03d}.jpg", sent_images[tid])
                    reg = r.region
                    crop = image.crop((reg.x, reg.y, reg.x + reg.width, reg.y + reg.height))
                    debug.save_image("annotated_tile_paths[]", f"annotated_tiles/tile_{tid:03d}.jpg",
                                     draw_detections(crop, per_tile[tid], offset=(reg.x, reg.y), label_tile=False))
            debug.save_json("raw_response_path", "response.json", [
                {"tile_id": r.tile_id, "region": r.region.as_dict(), "finish_reason": r.finish_reason,
                 "raw_text": r.raw_text, "parsed_valid_detections": [d.model_dump(mode="json") for d in per_tile[r.region.tile_id]]}
                for r in results
            ])
            if dedupe:
                write_dedupe_artifacts(debug, image, dedupe.books, dedupe.report, dedupe.pair_decisions,
                                       dedupe.ambiguous_pairs, regions)
        await asyncio.to_thread(write_debug)
    debug_write_ms = _ms(t0)

    performance = {
        "total_ms": _ms(t_total),
        "image_load_ms": image_load_ms,
        "tiling_ms": tiling_ms,
        "image_preparation_ms_sum": round(sum(r.timings.get("image_preparation_ms", 0) for r in results), 1),
        "inference_wall_ms": inference_wall_ms,
        "api_ms_sum": round(sum(r.timings.get("api_ms", 0) for r in results), 1),
        "api_ms_max": max((r.timings.get("api_ms", 0) for r in results), default=0),
        "json_parsing_ms_sum": round(sum(r.timings.get("json_parsing_ms", 0) for r in results), 1),
        "bbox_processing_ms_sum": round(sum(r.timings.get("bbox_processing_ms", 0) for r in results), 1),
        "dedupe_ms": dedupe_ms,
        "debug_write_ms": debug_write_ms,
        "api_calls": sum(len(r.attempts) for r in results),
        "retries": sum(max(len(r.attempts) - 1, 0) for r in results),
        "max_concurrency": config.max_concurrency,
        "request_timeout_seconds": config.request_timeout_seconds,
        "per_region": [{"tile_id": r.tile_id, "status": r.status, **r.timings} for r in results],
    }

    diagnostics = {
        "request_id": request_id,
        "provider": provider.name,
        "model": provider.model,
        "mode": mode.value,
        "image": {"path": str(image_path), "width": img_w, "height": img_h},
        "coordinate_convention": "bbox: original-image pixels after EXIF orientation, origin top-left. "
                                 "model_box_2d: Gemini [ymin, xmin, ymax, xmax] normalized 0-1000 relative to the region sent.",
        "tiling": tiling,
        "config": config.public(),
        "summary": summary,
        "tokens": tokens,
        "performance": performance,
        "warnings": warnings,
        "errors": errors,
        "regions": [r.diagnostics() for r in results],
        "dedupe": dedupe.report["summary"] | {"algorithm": dedupe.report["algorithm"], "config": dedupe.report["config"]}
        if dedupe else None,
        "dedupe_decisions": dedupe.pair_decisions if dedupe else [],
    }
    debug.save_json("diagnostics_path", "diagnostics.json", diagnostics)
    performance["total_ms"] = _ms(t_total)

    log.info("%s done: status=%s detections=%d (readable=%d partial=%d unreadable=%d) duplicates_removed=%d "
             "invalid_boxes=%d regions_failed=%d tokens=%s total=%.0fms api_sum=%.0fms",
             prefix, "ok" if not failed else ("partial" if ok else "failed"), len(detections), summary["readable_titles"],
             summary["partial_titles"], summary["unreadable_titles"], duplicates_removed, summary["invalid_boxes_rejected"],
             len(failed), tokens.get("total_tokens"), performance["total_ms"], performance["api_ms_sum"])

    response = {
        "request_id": request_id,
        "status": "ok" if not failed else "partial",
        "provider": provider.name,
        "model": provider.model,
        "mode": mode.value,
        "image_width": img_w,
        "image_height": img_h,
        "tiling": tiling,
        "persistence": "disabled",
        "detections": [d.model_dump(mode="json") for d in detections],
        "summary": summary,
        "tokens": tokens,
        "performance": performance,
        "warnings": warnings,
        "errors": errors,
        "debug": {"enabled": debug.enabled, "debug_dir": str(debug_dir) if debug_dir else None, **debug.files},
    }
    if not ok:
        raise VLMAnalysisError(_http_status_for(errors), "All VLM requests failed; see errors and diagnostics.", response)
    return response

"""Unit tests for the experimental VLM pipeline. No network: the provider is
a stub, so no paid API calls are made."""

import asyncio
import json
from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.vlm import router as vlm_router_module
from app.vlm.config import load_vlm_config
from app.vlm.geometry import (
    InvalidBox,
    Tile,
    compute_tiles,
    dedupe_across_tiles,
    full_image_region,
    model_box_to_pixels,
)
from app.vlm.provider import ProviderResponse, VLMConfigError, VLMProviderError, get_provider
from app.vlm.schemas import BBox, IdentificationStatus, Orientation, Readability, VLMDetection
from app.vlm.service import AnalysisMode, TileResult, VLMAnalysisError, analyze_image, parse_model_output


# --- helpers --------------------------------------------------------------------


def book(box, title="These Truths", author="Jill Lepore", readability="CLEAR", **overrides):
    item = {
        "box_2d": box,
        "visible_title": title,
        "visible_author": author,
        "raw_visible_text": " ".join(t for t in (title, author) if t) or None,
        "title_readability": readability,
        "author_readability": "CLEAR" if author else "NOT_VISIBLE",
        "identification_status": "VISIBLE_TEXT" if title else "NO_READABLE_TEXT",
        "orientation": "SPINE_OUT",
        "inferred_title": None,
        "uncertainty_notes": None,
        "shelf_index": 0,
    }
    item.update(overrides)
    return item


def detection(x, y, w, h, tile_id, title="These Truths", readability=Readability.CLEAR):
    return VLMDetection(
        index=0, bbox=BBox(x=x, y=y, width=w, height=h), visible_title=title, visible_author=None,
        raw_visible_text=title, title_readability=readability, author_readability=Readability.NOT_VISIBLE,
        identification_status=IdentificationStatus.VISIBLE_TEXT if title else IdentificationStatus.NO_READABLE_TEXT,
        orientation=Orientation.SPINE_OUT, inferred_title=None, uncertainty_notes=None, shelf_index=None,
        tile_id=tile_id, model_box_2d=[0, 0, 1, 1],
    )


class StubProvider:
    """Returns canned JSON per call. `responses` maps tile index (call order
    is not guaranteed under concurrency, so we key by prompt content) or is a
    callable(prompt) -> ProviderResponse | Exception."""

    name = "stub"
    model = "stub-model"

    def __init__(self, respond):
        self.respond = respond
        self.calls = 0

    async def analyze(self, image_bytes, mime_type, prompt, system_prompt):
        self.calls += 1
        result = self.respond(prompt)
        if isinstance(result, Exception):
            raise result
        return result

    async def aclose(self):
        pass


def ok(payload, finish="STOP"):
    text = payload if isinstance(payload, str) or payload is None else json.dumps(payload)
    return ProviderResponse(text=text, finish_reason=finish, block_reason=None, model_version="stub-1",
                            usage={"prompt_tokens": 100, "output_tokens": 50, "thinking_tokens": 10, "total_tokens": 160},
                            api_ms=12.0, attempts=[{"attempt": 1, "duration_ms": 12.0, "outcome": "ok"}])


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return replace(load_vlm_config(), max_concurrency=2, price_input_per_mtok=None, price_output_per_mtok=None)


@pytest.fixture
def shelf_image(tmp_path):
    path = tmp_path / "shelf.jpg"
    Image.new("RGB", (1000, 2000), (200, 180, 150)).save(path)
    return path


# --- bounding boxes ---------------------------------------------------------------


def test_box_converts_with_tile_offset():
    tile = Tile(tile_id=3, row=1, col=1, x=1000, y=2000, width=2048, height=2048)
    bbox, warnings = model_box_to_pixels([0, 500, 1000, 750], tile)  # ymin, xmin, ymax, xmax
    assert bbox == BBox(x=1000 + 1024, y=2000, width=512, height=2048)
    assert warnings == []


def test_full_image_box_scales_to_original_pixels():
    bbox, _ = model_box_to_pixels([100, 250, 400, 300], full_image_region(4590, 8160))
    assert (bbox.x, bbox.y, bbox.width, bbox.height) == (1148, 816, 229, 2448)


@pytest.mark.parametrize("box,reason", [
    (None, "missing_box"),
    ([1, 2, 3], "wrong_length"),
    ([500, 100, 400, 200], "inverted_or_empty"),
    ([100, 200, 300, 200], "inverted_or_empty"),
    ([0, 0, 1500, 100], "out_of_range"),
    ([-50, 0, 100, 100], "out_of_range"),
    ([100, 100, 101, 101], "too_small"),
])
def test_invalid_boxes_rejected(box, reason):
    with pytest.raises(InvalidBox) as exc:
        model_box_to_pixels(box, full_image_region(1000, 1000))
    assert exc.value.reason == reason


def test_slight_overshoot_is_clamped_not_rejected():
    bbox, warnings = model_box_to_pixels([0, 990, 1004, 1003], full_image_region(1000, 1000))
    assert bbox.x2 == 1000 and bbox.y2 == 1000
    assert warnings == ["box_clamped_to_region"]


# --- tiling -------------------------------------------------------------------------


def test_tiles_cover_image_with_required_overlap():
    w, h, size, overlap = 4590, 8160, 2048, 256
    tiles = compute_tiles(w, h, size, overlap)
    assert len(tiles) == 3 * 5
    assert all(t.width == size and t.height == size for t in tiles)
    assert all(t.x >= 0 and t.y >= 0 and t.x + t.width <= w and t.y + t.height <= h for t in tiles)
    xs = sorted({t.x for t in tiles})
    ys = sorted({t.y for t in tiles})
    assert xs[0] == 0 and xs[-1] + size == w and ys[0] == 0 and ys[-1] + size == h
    assert all(size - (b - a) >= overlap for a, b in zip(xs, xs[1:]))
    assert all(size - (b - a) >= overlap for a, b in zip(ys, ys[1:]))
    assert [t.tile_id for t in tiles] == list(range(len(tiles)))
    assert compute_tiles(w, h, size, overlap) == tiles  # deterministic


def test_small_image_is_single_tile():
    tiles = compute_tiles(1200, 900, 2048, 256)
    assert tiles == [Tile(0, 0, 0, 0, 0, 1200, 900)]


@pytest.mark.parametrize("size,overlap", [(100, 10), (2048, 2048), (2048, -1)])
def test_invalid_tile_params(size, overlap):
    with pytest.raises(ValueError):
        compute_tiles(4000, 4000, size, overlap)


# --- dedupe ---------------------------------------------------------------------------


def test_same_book_seen_by_two_tiles_is_merged():
    a = detection(1900, 100, 80, 900, tile_id=0)
    b = detection(1905, 102, 78, 896, tile_id=1)
    kept, decisions = dedupe_across_tiles([a, b])
    assert len(kept) == 1
    assert kept[0].merged_duplicates[0].tile_id in (0, 1)
    assert decisions[0]["decision"] == "merged"


def test_book_cut_by_tile_edge_merges_into_full_copy():
    full = detection(1000, 1800, 80, 900, tile_id=0, title="These Truths")
    cut = detection(1000, 1800, 80, 250, tile_id=2, title="These Tru", readability=Readability.PARTIAL)
    kept, _ = dedupe_across_tiles([cut, full])
    assert len(kept) == 1
    assert kept[0].visible_title == "These Truths"  # better-read copy wins


def test_same_tile_detections_never_merged():
    a = detection(100, 100, 80, 900, tile_id=0)
    b = detection(102, 100, 80, 900, tile_id=0)
    kept, _ = dedupe_across_tiles([a, b])
    assert len(kept) == 2


def test_overlapping_boxes_with_different_text_kept():
    a = detection(1900, 100, 80, 900, tile_id=0, title="These Truths")
    b = detection(1905, 100, 80, 900, tile_id=1, title="Crossing to Safety")
    kept, decisions = dedupe_across_tiles([a, b])
    assert len(kept) == 2
    assert decisions[0]["decision"] == "kept_both_text_differs"


def test_same_title_different_location_not_merged():
    # Two physical copies of one book on the shelf.
    a = detection(100, 100, 80, 900, tile_id=0)
    b = detection(300, 100, 80, 900, tile_id=1)
    kept, decisions = dedupe_across_tiles([a, b])
    assert len(kept) == 2
    assert decisions == []


# --- structured response validation ------------------------------------------------------


def _parse(text, finish="STOP"):
    r = TileResult(region=full_image_region(1000, 2000), is_tile=False, raw_text=text, finish_reason=finish)
    parse_model_output(r, 1000, 2000)
    return r


def test_valid_response_parsed_into_pixel_detections():
    r = _parse(json.dumps({"books": [book([100, 100, 500, 150]), book([100, 200, 500, 260], title=None, author=None, readability="UNREADABLE")]}))
    assert r.status == "ok"
    assert [d.bbox.model_dump() for d in r.detections] == [
        {"x": 100, "y": 200, "width": 50, "height": 800},
        {"x": 200, "y": 200, "width": 60, "height": 800},
    ]
    assert r.detections[1].identification_status == IdentificationStatus.NO_READABLE_TEXT
    assert all(d.model_confidence is None for d in r.detections)


def test_invalid_items_rejected_individually():
    items = [book([100, 100, 500, 150]), book([500, 100, 100, 150]), {"box_2d": [1, 2, 3, 4], "visible_title": 7}]
    r = _parse(json.dumps({"books": items}))
    assert r.status == "ok"
    assert len(r.detections) == 1
    assert r.invalid_boxes[0]["reason"] == "inverted_or_empty"
    assert r.item_errors[0]["reason"] == "schema_validation"


def test_readable_claim_without_title_is_downgraded():
    r = _parse(json.dumps({"books": [book([100, 100, 500, 150], title=None, readability="CLEAR")]}))
    assert r.detections[0].title_readability == Readability.UNREADABLE
    assert "title_readability_downgraded_from_CLEAR" in r.detections[0].warnings


@pytest.mark.parametrize("text,finish,kind", [
    ("not json", "STOP", "invalid_json"),
    ('{"books": [{"box_2d": [1', "MAX_TOKENS", "output_truncated"),
    ("", "STOP", "empty_response"),
    (None, "SAFETY", "blocked"),
    ('{"items": []}', "STOP", "unexpected_structure"),
])
def test_bad_model_output_fails_tile_without_inventing(text, finish, kind):
    r = _parse(text, finish)
    assert r.status == "failed"
    assert r.error["kind"] == kind
    assert r.detections == []


# --- service ------------------------------------------------------------------------------


def test_full_mode_end_to_end(config, shelf_image, tmp_path):
    provider = StubProvider(lambda prompt: ok({"books": [book([100, 100, 500, 150])]}))
    result = asyncio.run(analyze_image(shelf_image, request_id="r1", mode=AnalysisMode.full, config=config,
                                       provider=provider, debug_dir=tmp_path / "dbg"))
    assert provider.calls == 1
    assert result["status"] == "ok" and result["mode"] == "full" and result["model"] == "stub-model"
    assert result["detections"][0]["bbox"] == {"x": 100, "y": 200, "width": 50, "height": 800}
    assert result["detections"][0]["shelf_index"] == 0
    assert result["summary"]["readable_titles"] == 1
    assert result["persistence"] == "disabled"
    for key in ("original_image_path", "annotated_image_path", "raw_response_path", "diagnostics_path"):
        assert (tmp_path / "dbg").joinpath(result["debug"][key].split("/dbg/")[1]).exists()
    diag = json.loads((tmp_path / "dbg" / "diagnostics.json").read_text())
    assert "api_key" not in diag["config"] and diag["config"]["api_key_configured"] is True


def test_tiled_mode_partial_failure_keeps_other_tiles(config, shelf_image, tmp_path):
    def respond(prompt):
        if "tile 1 " in prompt:
            return VLMProviderError("server_error", "boom", status_code=500, retryable=True,
                                    attempts=[{"attempt": 1, "duration_ms": 5, "outcome": "server_error"}])
        return ok({"books": [book([10, 10, 900, 100])]})

    provider = StubProvider(respond)
    result = asyncio.run(analyze_image(shelf_image, request_id="r2", mode=AnalysisMode.tiled, config=config,
                                       provider=provider, debug_dir=tmp_path / "dbg", tile_size=1000, tile_overlap=100))
    assert result["tiling"]["rows"] == 3 and result["tiling"]["cols"] == 1
    assert result["status"] == "partial"
    assert result["summary"]["regions_failed"] == 1
    assert result["errors"][0]["tile_id"] == 1
    assert len(result["detections"]) == 2
    assert all(d["tile_id"] in (0, 2) and d["shelf_index"] is None for d in result["detections"])
    assert len(result["debug"]["annotated_tile_paths"]) == 3


def test_permanent_error_skips_remaining_tiles(config, shelf_image):
    config = replace(config, max_concurrency=1)
    provider = StubProvider(lambda prompt: VLMProviderError("model_not_found", "no such model", status_code=404))
    with pytest.raises(VLMAnalysisError) as exc:
        asyncio.run(analyze_image(shelf_image, request_id="r3", mode=AnalysisMode.tiled, config=config,
                                  provider=provider, debug_dir=None, tile_size=1000, tile_overlap=100))
    assert provider.calls == 1  # no repeated calls after a permanent error
    assert exc.value.status_code == 502
    assert sorted(e["kind"] for e in exc.value.payload["errors"]) == ["aborted", "aborted", "model_not_found"]


def test_empty_model_response_is_reported_not_filled(config, shelf_image):
    provider = StubProvider(lambda prompt: ok({"books": []}))
    result = asyncio.run(analyze_image(shelf_image, request_id="r4", mode=AnalysisMode.full, config=config,
                                       provider=provider, debug_dir=None))
    assert result["detections"] == []
    assert "model_returned_no_valid_detections" in result["warnings"]
    assert result["debug"]["enabled"] is False


def test_too_many_tiles_rejected_before_any_call(config, shelf_image):
    from app.vlm.service import VLMRequestError

    provider = StubProvider(lambda prompt: ok({"books": []}))
    with pytest.raises(VLMRequestError):
        asyncio.run(analyze_image(shelf_image, request_id="r5", mode=AnalysisMode.tiled, config=replace(config, max_tiles=2),
                                  provider=provider, debug_dir=None, tile_size=512, tile_overlap=0))
    assert provider.calls == 0


# --- configuration / endpoint ----------------------------------------------------------------


def test_missing_api_key_is_config_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(VLMConfigError, match="GEMINI_API_KEY"):
        get_provider(load_vlm_config())


def test_endpoint_returns_503_without_api_key(monkeypatch, tmp_path, shelf_image):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(vlm_router_module, "UPLOAD_DIR", tmp_path / "uploads")
    app = FastAPI()
    app.include_router(vlm_router_module.router)
    with TestClient(app) as client:
        resp = client.post("/analyze/vlm", files={"image": ("shelf.jpg", shelf_image.read_bytes(), "image/jpeg")})
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"] == "vlm_not_configured"
    assert not (tmp_path / "uploads").exists()  # nothing stored when misconfigured

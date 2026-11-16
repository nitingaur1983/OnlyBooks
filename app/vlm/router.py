"""POST /analyze/vlm — experimental Gemini VLM pipeline.

Read-only with respect to the inventory database: no Session dependency, no
Repository, no matcher. Results go to the response and data/debug/vlm/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import BASE_DIR, load_vlm_config
from .provider import VLMConfigError, get_provider
from .service import AnalysisMode, VLMAnalysisError, VLMRequestError, analyze_image

logger = logging.getLogger(__name__)

# Same upload dir and naming convention as /analyze in app/main.py.
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VLM_DEBUG_DIR = BASE_DIR / "data" / "debug" / "vlm"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

router = APIRouter(tags=["VLM Analysis"])


@router.post("/analyze/vlm")
async def analyze_vlm(
    image: UploadFile = File(..., description="Bookshelf photo (JPG, PNG or WEBP)"),
    mode: AnalysisMode = Form(
        default=AnalysisMode.tiled,
        description="`tiled` (recommended for dense shelves): overlapping tiles, one VLM call each. "
        "`full`: the whole photo in one call (for benchmarking).",
    ),
    tile_size: int | None = Form(default=None, description="Tile edge in original-image px (tiled mode). Default VLM_TILE_SIZE."),
    tile_overlap: int | None = Form(default=None, description="Overlap between tiles in px (tiled mode). Default VLM_TILE_OVERLAP."),
    debug: bool | None = Form(default=None, description="Write debug images/JSON. Default VLM_DEBUG."),
):
    """Detect individual books with a vision-language model. Does not run the
    OpenCV/Tesseract pipeline and does not write to the inventory database."""
    config = load_vlm_config()
    try:
        provider = get_provider(config)  # fails fast on a missing key, before storing anything
    except VLMConfigError as exc:
        raise HTTPException(status_code=503, detail={"error": "vlm_not_configured", "message": str(exc)}) from exc

    try:
        suffix = Path(image.filename or "photo.jpg").suffix.lower() or ".jpg"
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail="Use JPG, PNG, or WEBP")
        max_bytes = config.max_upload_mb * 1024 * 1024
        data = await image.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload exceeds VLM_MAX_UPLOAD_MB={config.max_upload_mb}")
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload")

        request_id = uuid4().hex
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / f"{request_id}{suffix}"
        target.write_bytes(data)

        debug_enabled = config.debug if debug is None else debug
        debug_dir = VLM_DEBUG_DIR / request_id if debug_enabled else None
        logger.info("[vlm %s] upload %s (%d KB) mode=%s", request_id, image.filename, len(data) // 1024, mode.value)

        try:
            return await analyze_image(
                target,
                request_id=request_id,
                mode=mode,
                config=config,
                provider=provider,
                debug_dir=debug_dir,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
        except VLMRequestError as exc:
            if exc.status_code in (400, 413):
                target.unlink(missing_ok=True)
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except VLMAnalysisError as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": str(exc), **exc.payload})
    finally:
        await provider.aclose()

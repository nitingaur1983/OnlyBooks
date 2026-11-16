"""Env-driven settings for the experimental VLM pipeline.

Unlike app/config.py (frozen at import), settings here are read from the
process environment by load_vlm_config() on every request, so tests can
monkeypatch env. Note .env itself is loaded once at startup: uvicorn --reload
only watches .py files, so restart the app after editing .env.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# The OpenCV pipeline never needed a .env; the VLM pipeline needs an API key.
# override=False: real environment variables always win over .env.
load_dotenv(BASE_DIR / ".env", override=False)

# Verified against client.models.list() for this project's key (2026-10-31).
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def _str_env(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class VLMConfig:
    provider: str
    model: str
    api_key: str | None  # never logged or returned — see VLMConfig.public()
    debug: bool
    persist_results: bool  # accepted but not implemented yet; see router

    # --- Cost / size guards ---
    max_upload_mb: int
    max_image_pixels: int  # reject decoded images larger than this
    # Longest side of any image actually sent to the model (full image or
    # tile); larger crops are downscaled. Coordinates are unaffected because
    # the model answers in 0..1000 coordinates relative to what it saw.
    max_send_dimension: int
    max_tiles: int
    tile_size: int
    tile_overlap: int
    max_concurrency: int
    max_output_tokens: int
    media_resolution: str | None  # "low" | "medium" | "high" | None (model default)
    thinking_level: str | None  # "minimal" | "low" | "medium" | "high" | None

    # --- Resilience ---
    request_timeout_seconds: float
    max_retries: int  # retries for 429 / 5xx / connection errors only
    retry_on_timeout: bool  # timeouts may still be billed, so off by default
    retry_base_delay_seconds: float

    # --- Optional cost estimate (USD per 1M tokens). Unset -> cost is null. ---
    price_input_per_mtok: float | None
    price_output_per_mtok: float | None

    def public(self) -> dict:
        """Config snapshot safe to log and return to clients (no API key)."""
        data = {k: v for k, v in self.__dict__.items() if k != "api_key"}
        data["api_key_configured"] = bool(self.api_key)
        return data


def load_vlm_config() -> VLMConfig:
    return VLMConfig(
        provider=(_str_env("VLM_PROVIDER", "gemini") or "gemini").lower(),
        model=_str_env("VLM_MODEL", DEFAULT_GEMINI_MODEL),
        api_key=_str_env("GEMINI_API_KEY", None),
        debug=_bool_env("VLM_DEBUG", True),
        persist_results=_bool_env("VLM_PERSIST_RESULTS", False),
        max_upload_mb=_int_env("VLM_MAX_UPLOAD_MB", 40),
        max_image_pixels=_int_env("VLM_MAX_IMAGE_PIXELS", 60_000_000),
        max_send_dimension=_int_env("VLM_MAX_SEND_DIMENSION", 3072),
        max_tiles=_int_env("VLM_MAX_TILES", 24),
        tile_size=_int_env("VLM_TILE_SIZE", 2048),
        tile_overlap=_int_env("VLM_TILE_OVERLAP", 256),
        max_concurrency=max(1, _int_env("VLM_MAX_CONCURRENCY", 3)),
        max_output_tokens=_int_env("VLM_MAX_OUTPUT_TOKENS", 32768),
        media_resolution=(_str_env("VLM_MEDIA_RESOLUTION", "high") or "").lower() or None,
        thinking_level=(_str_env("VLM_THINKING_LEVEL", None) or "").lower() or None,
        request_timeout_seconds=_float_env("VLM_REQUEST_TIMEOUT_SECONDS", 120.0),
        max_retries=max(0, _int_env("VLM_MAX_RETRIES", 2)),
        retry_on_timeout=_bool_env("VLM_RETRY_ON_TIMEOUT", False),
        retry_base_delay_seconds=_float_env("VLM_RETRY_BASE_DELAY_SECONDS", 2.0),
        price_input_per_mtok=_float_env("VLM_PRICE_INPUT_PER_MTOK", None),
        price_output_per_mtok=_float_env("VLM_PRICE_OUTPUT_PER_MTOK", None),
    )


def load_dedupe_config():
    """Physical-book dedupe thresholds (see app/vlm/dedupe.py), each
    overridable as VLM_DEDUPE_<FIELD> (e.g. VLM_DEDUPE_STRONG_IOU=0.75)."""
    from dataclasses import fields

    from .dedupe import DedupeConfig

    defaults = DedupeConfig()
    values = {}
    for f in fields(DedupeConfig):
        name = f"VLM_DEDUPE_{f.name.upper()}"
        default = getattr(defaults, f.name)
        values[f.name] = _int_env(name, default) if isinstance(default, int) else _float_env(name, default)
    return DedupeConfig(**values)

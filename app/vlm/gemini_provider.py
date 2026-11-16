"""Gemini implementation of VLMProvider, using the official `google-genai` SDK.

Retries are done here (not by the SDK, whose retries are disabled) so every
attempt is timed and recorded. Only transient failures are retried: 429,
5xx, connection errors, and — only if VLM_RETRY_ON_TIMEOUT — timeouts.
Permanent errors (bad key, unknown model, invalid request) are never retried,
and a slow-but-successful call is simply accepted.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

import httpx
from google import genai
from google.genai import errors, types

from .config import VLMConfig
from .provider import ProviderResponse, VLMProviderError
from .schemas import ModelOutput

logger = logging.getLogger(__name__)

_MEDIA_RESOLUTION = {
    "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}
_THINKING_LEVEL = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}


def _classify(exc: Exception, retry_on_timeout: bool) -> VLMProviderError:
    if isinstance(exc, errors.ClientError):
        code = exc.code
        msg = exc.message or str(exc)
        if code == 429:
            return VLMProviderError("rate_limited", msg, status_code=code, retryable=True)
        if code == 408:
            return VLMProviderError("timeout", msg, status_code=code, retryable=retry_on_timeout)
        if code in (401, 403):
            return VLMProviderError("auth", msg, status_code=code)
        if code == 404 or ("model" in msg.lower() and ("not found" in msg.lower() or "not supported" in msg.lower())):
            return VLMProviderError("model_not_found", msg, status_code=code)
        return VLMProviderError("invalid_request", msg, status_code=code)
    if isinstance(exc, errors.ServerError):
        if exc.code == 504 or (exc.status or "").upper() == "DEADLINE_EXCEEDED":
            # The SDK forwards our timeout as a server deadline, so a slow
            # model surfaces as 504 rather than a client-side timeout.
            return VLMProviderError("timeout", exc.message or str(exc), status_code=exc.code, retryable=retry_on_timeout)
        return VLMProviderError("server_error", exc.message or str(exc), status_code=exc.code, retryable=True)
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return VLMProviderError("timeout", f"no response within timeout ({type(exc).__name__})", retryable=retry_on_timeout)
    if isinstance(exc, httpx.TransportError):
        return VLMProviderError("network", f"{type(exc).__name__}: {exc}", retryable=True)
    return VLMProviderError("unexpected", f"{type(exc).__name__}: {exc}")


def _usage(resp: types.GenerateContentResponse) -> dict:
    u = resp.usage_metadata
    return {
        "prompt_tokens": getattr(u, "prompt_token_count", None),
        "output_tokens": getattr(u, "candidates_token_count", None),
        "thinking_tokens": getattr(u, "thoughts_token_count", None),
        "total_tokens": getattr(u, "total_token_count", None),
    }


class GeminiProvider:
    name = "gemini"

    def __init__(self, config: VLMConfig):
        self.config = config
        self.model = config.model
        self._client = genai.Client(
            api_key=config.api_key,
            http_options=types.HttpOptions(
                timeout=int(config.request_timeout_seconds * 1000),  # SDK takes milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),  # we retry ourselves, see module doc
            ),
        )

    def _generate_config(self, system_prompt: str) -> types.GenerateContentConfig:
        kwargs = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": ModelOutput,
            "max_output_tokens": self.config.max_output_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),  # no tools used
        }
        if self.config.media_resolution in _MEDIA_RESOLUTION:
            kwargs["media_resolution"] = _MEDIA_RESOLUTION[self.config.media_resolution]
        if self.config.thinking_level in _THINKING_LEVEL:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=_THINKING_LEVEL[self.config.thinking_level])
        return types.GenerateContentConfig(**kwargs)

    async def analyze(self, image_bytes: bytes, mime_type: str, prompt: str, system_prompt: str) -> ProviderResponse:
        contents = [types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt]
        gen_config = self._generate_config(system_prompt)
        attempts: list[dict] = []
        max_attempts = 1 + self.config.max_retries

        for attempt in range(1, max_attempts + 1):
            start = time.perf_counter()
            try:
                # wait_for is a backstop in case the transport timeout doesn't fire.
                resp = await asyncio.wait_for(
                    self._client.aio.models.generate_content(model=self.model, contents=contents, config=gen_config),
                    timeout=self.config.request_timeout_seconds + 5,
                )
            except Exception as exc:  # classified below; nothing is swallowed
                elapsed = (time.perf_counter() - start) * 1000
                err = _classify(exc, self.config.retry_on_timeout)
                attempts.append({"attempt": attempt, "duration_ms": round(elapsed, 1), "outcome": err.kind,
                                 "status_code": err.status_code, "error": str(err)[:500]})
                if err.retryable and attempt < max_attempts:
                    delay = self.config.retry_base_delay_seconds * (2 ** (attempt - 1))
                    delay += random.uniform(0, delay / 2)
                    logger.warning("gemini attempt %d/%d failed (%s, %.0f ms); retrying in %.1fs",
                                   attempt, max_attempts, err.kind, elapsed, delay)
                    await asyncio.sleep(delay)
                    continue
                logger.error("gemini call failed permanently after %d attempt(s): %s", attempt, err.kind)
                err.attempts = attempts
                raise err from exc

            elapsed = (time.perf_counter() - start) * 1000
            attempts.append({"attempt": attempt, "duration_ms": round(elapsed, 1), "outcome": "ok"})
            candidate = resp.candidates[0] if resp.candidates else None
            finish = candidate.finish_reason.name if candidate and candidate.finish_reason else None
            block = None
            if resp.prompt_feedback and resp.prompt_feedback.block_reason:
                block = resp.prompt_feedback.block_reason.name
            try:
                text = resp.text
            except Exception:  # SDK raises on some blocked/empty responses
                text = None
            return ProviderResponse(
                text=text,
                finish_reason=finish,
                block_reason=block,
                model_version=resp.model_version,
                usage=_usage(resp),
                api_ms=round(elapsed, 1),
                attempts=attempts,
            )

        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self._client.aio.aclose()

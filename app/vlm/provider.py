"""Provider contract. The service only depends on this, so adding OpenAI or a
locally hosted open-weight VLM means writing one class with `analyze()` and
registering it in get_provider() — the /analyze/vlm contract doesn't change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .config import VLMConfig


class VLMConfigError(RuntimeError):
    """The pipeline isn't usable as configured (missing key, unknown provider).
    Raised before any API call is made."""


@dataclass
class ProviderResponse:
    text: str | None  # raw model output (expected to be JSON)
    finish_reason: str | None
    block_reason: str | None
    model_version: str | None
    usage: dict  # prompt_tokens / output_tokens / thinking_tokens / total_tokens, None when not reported
    api_ms: float  # duration of the successful call only
    attempts: list[dict] = field(default_factory=list)


class VLMProviderError(RuntimeError):
    """A call that failed after any permitted retries.

    kind: auth | model_not_found | invalid_request | rate_limited | timeout |
          server_error | network | unexpected
    """

    PERMANENT_CONFIG_KINDS = {"auth", "model_not_found"}

    def __init__(self, kind: str, message: str, *, status_code: int | None = None,
                 retryable: bool = False, attempts: list[dict] | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retryable = retryable
        self.attempts = attempts or []

    @property
    def aborts_request(self) -> bool:
        """Errors that will fail identically for every tile — stop calling."""
        return self.kind in self.PERMANENT_CONFIG_KINDS

    def as_dict(self) -> dict:
        return {"kind": self.kind, "message": str(self), "status_code": self.status_code,
                "retryable": self.retryable, "attempts": self.attempts}


class VLMProvider(Protocol):
    name: str
    model: str

    async def analyze(self, image_bytes: bytes, mime_type: str, prompt: str, system_prompt: str) -> ProviderResponse: ...

    async def aclose(self) -> None: ...


def get_provider(config: VLMConfig) -> VLMProvider:
    if config.provider == "gemini":
        if not config.api_key:
            raise VLMConfigError("GEMINI_API_KEY is not set. Add it to .env (see .env.example) or the environment.")
        from .gemini_provider import GeminiProvider

        return GeminiProvider(config)
    raise VLMConfigError(f"Unsupported VLM_PROVIDER={config.provider!r}. Supported: 'gemini'.")

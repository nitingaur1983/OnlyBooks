"""OCR cleanup: turn raw Tesseract output into evidence a matcher can reason about.

Tesseract regularly returns runs of dashes/pipes/symbols with no real words in
them (e.g. spine dividers, shadows, decorative fonts it can't read). This module
does *not* try to fix OCR mistakes in real words — it just strips symbol noise
and measures how much of the text is actually plausible language, so the
matcher can cheaply refuse to treat symbol soup as book evidence.
"""

import re
from dataclasses import dataclass, field

# Anything that isn't a letter, digit, apostrophe, hyphen, or whitespace is
# noise for our purposes (Tesseract loves emitting stray punctuation).
_NOISE_CHARS_RE = re.compile(r"[^\w\s'-]", re.UNICODE)
# Long runs of dashes/underscores are almost always spine dividers or shadow
# edges, not em-dashes in real text.
_MULTI_DASH_RE = re.compile(r"[-_]{2,}")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass
class CleanedText:
    cleaned: str
    alpha_tokens: list[str] = field(default_factory=list)
    alpha_ratio: float = 0.0  # fraction of non-space raw characters that are letters


def clean(raw_text: str) -> CleanedText:
    compact = " ".join(raw_text.split())
    no_multi_dash = _MULTI_DASH_RE.sub(" ", compact)
    stripped = _NOISE_CHARS_RE.sub(" ", no_multi_dash)
    stripped = " ".join(stripped.split())

    alpha_tokens = [t for t in _WORD_RE.findall(stripped) if len(t) > 1]

    non_space = [c for c in compact if not c.isspace()]
    alpha_count = sum(1 for c in non_space if c.isalpha())
    alpha_ratio = (alpha_count / len(non_space)) if non_space else 0.0

    return CleanedText(cleaned=stripped, alpha_tokens=alpha_tokens, alpha_ratio=alpha_ratio)

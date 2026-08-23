"""Decides whether OCR evidence is good enough to call it a real book.

This is the single gate in the whole pipeline: nothing upstream (vision, OCR
cleanup) and nothing downstream (repository) is allowed to decide "this is a
book" on its own. If BookMatcher says NO_MATCH, no Book row is ever created —
only a Detection row preserving the evidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

from .config import DEFAULT_CONFIG, MatchConfig
from .metadata_source import MetadataCandidate, MetadataSource
from .ocr_cleanup import CleanedText, clean

logger = logging.getLogger(__name__)

# A loose ISBN-10/13 shape. We only trust it once digit-stripping yields
# exactly 10 or 13 characters — see _extract_isbn.
ISBN_RE = re.compile(r"(?:97[89][\-\s]?)?(?:\d[\-\s]?){9}[\dXx]")

# Words common enough that overlapping on them alone proves nothing.
COMMON_WORDS = {
    "the", "and", "of", "a", "an", "in", "on", "to", "for", "with", "by",
    "is", "it", "or", "at", "from", "this", "that", "book", "novel",
    "story", "volume", "edition", "new", "press", "books",
}


class MatchStatus(str, Enum):
    NO_MATCH = "NO_MATCH"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    POSSIBLE_MATCH = "POSSIBLE_MATCH"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"


_STATUS_ORDER = [
    MatchStatus.NO_MATCH,
    MatchStatus.LOW_CONFIDENCE,
    MatchStatus.POSSIBLE_MATCH,
    MatchStatus.HIGH_CONFIDENCE,
]


def status_at_least(status: MatchStatus, floor: MatchStatus) -> bool:
    return _STATUS_ORDER.index(status) >= _STATUS_ORDER.index(floor)


@dataclass
class MatchResult:
    status: MatchStatus
    score: float
    reason: str
    query: str
    matched: MetadataCandidate | None


class BookMatcher:
    def __init__(self, metadata_source: MetadataSource, config: MatchConfig = DEFAULT_CONFIG):
        self.metadata_source = metadata_source
        self.config = config

    def match(self, raw_text: str, ocr_confidence: float) -> MatchResult:
        cleaned = clean(raw_text)

        isbn = self._extract_isbn(raw_text)
        if isbn:
            hit = self._try_isbn(isbn)
            if hit:
                return hit
            # ISBN-shaped text that didn't resolve still falls through to a
            # normal title/author search below instead of being discarded.

        if not self._has_sufficient_signal(cleaned):
            return MatchResult(
                status=MatchStatus.NO_MATCH,
                score=0.0,
                reason="insufficient_ocr_signal",
                query=cleaned.cleaned,
                matched=None,
            )

        query = " ".join(cleaned.alpha_tokens[:8])
        try:
            candidates = self.metadata_source.search(query, limit=self.config.metadata_search_limit)
        except Exception as exc:  # belt-and-braces; sources should already catch their own errors
            logger.warning("metadata search raised for query %r: %s", query, exc)
            return MatchResult(MatchStatus.NO_MATCH, 0.0, f"metadata_source_error:{exc}", query, None)

        if not candidates:
            return MatchResult(MatchStatus.NO_MATCH, 0.0, "no_search_results", query, None)

        best_candidate: MetadataCandidate | None = None
        best_score = -1.0
        best_reason = ""
        best_distinctive_hit = False
        for candidate in candidates:
            score, distinctive_hit, detail = self._score(cleaned, ocr_confidence, candidate)
            if score > best_score:
                best_score, best_candidate, best_reason, best_distinctive_hit = (
                    score, candidate, detail, distinctive_hit,
                )

        status = self._classify(best_score, best_distinctive_hit)
        return MatchResult(
            status=status,
            score=round(best_score, 3),
            reason=best_reason,
            query=query,
            matched=best_candidate if status != MatchStatus.NO_MATCH else None,
        )

    def _has_sufficient_signal(self, cleaned: CleanedText) -> bool:
        usable = [t for t in cleaned.alpha_tokens if len(t) >= 3]
        return len(usable) >= self.config.min_alpha_tokens and cleaned.alpha_ratio >= self.config.min_alpha_ratio

    @staticmethod
    def _extract_isbn(raw_text: str) -> str | None:
        match = ISBN_RE.search(raw_text)
        if not match:
            return None
        digits = re.sub(r"[^0-9Xx]", "", match.group(0)).upper()
        return digits if len(digits) in (10, 13) else None

    def _try_isbn(self, isbn: str) -> MatchResult | None:
        try:
            hit = self.metadata_source.lookup_isbn(isbn)
        except Exception as exc:
            logger.warning("ISBN lookup raised for %r: %s", isbn, exc)
            return None
        if not hit:
            return None
        return MatchResult(
            status=MatchStatus.HIGH_CONFIDENCE,
            score=1.0,
            reason=f"isbn_exact_match:{isbn}",
            query=isbn,
            matched=hit,
        )

    def _score(
        self, cleaned: CleanedText, ocr_confidence: float, candidate: MetadataCandidate
    ) -> tuple[float, bool, str]:
        cleaned_lower = cleaned.cleaned.lower()
        title_lower = (candidate.title or "").lower()
        author_lower = (candidate.author or "").lower()

        title_sim = SequenceMatcher(None, cleaned_lower, title_lower).ratio()

        candidate_tokens = set(re.findall(r"[a-z]+", f"{title_lower} {author_lower}"))
        query_tokens = {t.lower() for t in cleaned.alpha_tokens}
        overlap = query_tokens & candidate_tokens
        token_overlap_ratio = (len(overlap) / len(query_tokens)) if query_tokens else 0.0

        distinctive_hit = any(
            len(t) >= self.config.min_token_length_for_distinctive and t not in COMMON_WORDS
            for t in overlap
        )

        ocr_weight = max(0.0, min(ocr_confidence / 100.0, 1.0))
        composite = 0.5 * title_sim + 0.3 * token_overlap_ratio + 0.2 * ocr_weight

        if not distinctive_hit:
            # Never let a match built only out of "the"/"new"/"press" etc.
            # cross into POSSIBLE_MATCH/HIGH_CONFIDENCE territory.
            composite = min(composite, self.config.low_confidence_threshold - 0.01)

        detail = (
            f"title_sim={title_sim:.2f} token_overlap={token_overlap_ratio:.2f} "
            f"ocr_conf={ocr_confidence:.0f} distinctive_hit={distinctive_hit} "
            f"candidate='{candidate.title}' by '{candidate.author}'"
        )
        return composite, distinctive_hit, detail

    def _classify(self, score: float, distinctive_hit: bool) -> MatchStatus:
        if distinctive_hit and score >= self.config.high_confidence_threshold:
            return MatchStatus.HIGH_CONFIDENCE
        if distinctive_hit and score >= self.config.possible_match_threshold:
            return MatchStatus.POSSIBLE_MATCH
        if score >= self.config.low_confidence_threshold:
            return MatchStatus.LOW_CONFIDENCE
        return MatchStatus.NO_MATCH

"""Authoritative book metadata lookups, kept behind a small interface.

BookMatcher only ever talks to a `MetadataSource` — never to a specific API —
so OCR/matching code stays decoupled from whichever catalog we happen to use.
Open Library is free and keyless, which suits a POC; swapping in Google Books
or a local catalog later means writing one more class here, nothing else.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class MetadataCandidate:
    title: str
    author: str | None
    isbn: str | None
    publisher: str | None
    source: str
    external_id: str | None
    edition_count: int = 0


class MetadataSource(Protocol):
    def search(self, query: str, limit: int = 5) -> list[MetadataCandidate]: ...

    def lookup_isbn(self, isbn: str) -> MetadataCandidate | None: ...


class OpenLibraryMetadataSource:
    """Talks to https://openlibrary.org — no API key required.

    Network failures, timeouts, and malformed responses are caught and logged
    here so a flaky API degrades to "no evidence found" rather than crashing
    the analyze pipeline or corrupting matching decisions.
    """

    SEARCH_URL = "https://openlibrary.org/search.json"
    ISBN_URL = "https://openlibrary.org/api/books"

    def __init__(self, timeout_seconds: float = 4.0):
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int = 5) -> list[MetadataCandidate]:
        if not query.strip():
            return []
        params = urllib.parse.urlencode({"q": query, "limit": limit})
        url = f"{self.SEARCH_URL}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # network error, timeout, bad JSON, etc.
            logger.warning("Open Library search failed for %r: %s", query, exc)
            return []

        candidates = []
        for doc in data.get("docs", [])[:limit]:
            title = doc.get("title")
            if not title:
                continue
            authors = doc.get("author_name") or []
            isbns = doc.get("isbn") or []
            publishers = doc.get("publisher") or []
            candidates.append(
                MetadataCandidate(
                    title=title,
                    author=authors[0] if authors else None,
                    isbn=isbns[0] if isbns else None,
                    publisher=publishers[0] if publishers else None,
                    source="open_library",
                    external_id=doc.get("key"),
                    edition_count=doc.get("edition_count", 0),
                )
            )
        return candidates

    def lookup_isbn(self, isbn: str) -> MetadataCandidate | None:
        params = urllib.parse.urlencode(
            {"bibkeys": f"ISBN:{isbn}", "format": "json", "jscmd": "data"}
        )
        url = f"{self.ISBN_URL}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Open Library ISBN lookup failed for %r: %s", isbn, exc)
            return None

        entry = data.get(f"ISBN:{isbn}")
        if not entry or not entry.get("title"):
            return None
        authors = entry.get("authors") or []
        publishers = entry.get("publishers") or []
        return MetadataCandidate(
            title=entry["title"],
            author=authors[0]["name"] if authors else None,
            isbn=isbn,
            publisher=publishers[0]["name"] if publishers else None,
            source="open_library",
            external_id=entry.get("key"),
            edition_count=1,
        )

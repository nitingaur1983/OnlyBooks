import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from .matcher import MatchResult
from .metadata_source import MetadataCandidate
from .models import Book, BookSighting, Detection, Photo
from .vision import SpineCandidate


def _normalize_key(candidate: MetadataCandidate) -> str:
    """ISBN wins when we have it; otherwise fall back to title+author.

    This is what get_or_create_book dedupes on, so two detections of the same
    physical book (even via slightly different OCR crops) collapse to one row.
    """
    if candidate.isbn:
        return f"isbn:{candidate.isbn}"
    key_source = f"{candidate.title}|{candidate.author or ''}".lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", key_source).strip()
    return normalized or f"title:{candidate.title.lower()}"


class Repository:
    def __init__(self, db: Session):
        self.db = db

    def create_photo(self, file_name: str, file_path: str, store_name: str | None) -> Photo:
        photo = Photo(file_name=file_name, file_path=file_path, store_name=store_name)
        self.db.add(photo)
        self.db.flush()
        return photo

    def create_detection(
        self, photo: Photo, spine: SpineCandidate, cleaned_text: str, result: MatchResult
    ) -> Detection:
        """Always writes a row, whatever the match decision was.

        This is the evidence/debug trail: raw OCR, cleaned OCR, confidence, the
        query that was searched, the matcher's score, and its reasoning are all
        preserved here regardless of whether a Book was ever created.
        """
        detection = Detection(
            photo_id=photo.id,
            spine_index=spine.index,
            x=spine.x,
            y=spine.y,
            width=spine.width,
            height=spine.height,
            raw_ocr_text=spine.text,
            cleaned_ocr_text=cleaned_text,
            ocr_confidence=spine.confidence,
            match_status=result.status.value,
            match_confidence=result.score,
            match_query=result.query,
            match_reason=result.reason,
        )
        self.db.add(detection)
        self.db.flush()
        return detection

    def get_or_create_book(self, candidate: MetadataCandidate) -> Book:
        normalized_key = _normalize_key(candidate)
        existing = self.db.scalar(select(Book).where(Book.normalized_key == normalized_key))
        if existing:
            return existing
        book = Book(
            title=candidate.title,
            author=candidate.author,
            isbn=candidate.isbn,
            publisher=candidate.publisher,
            normalized_key=normalized_key,
            source=candidate.source,
            external_id=candidate.external_id,
        )
        self.db.add(book)
        self.db.flush()
        return book

    def link_book_to_detection(self, detection: Detection, book: Book) -> None:
        detection.matched_book_id = book.id

    def add_sighting(
        self, detection: Detection, book: Book, photo: Photo, store_name: str | None, confidence: float
    ) -> BookSighting:
        sighting = BookSighting(
            book_id=book.id,
            photo_id=photo.id,
            detection_id=detection.id,
            store_name=store_name,
            confidence=confidence,
        )
        self.db.add(sighting)
        self.db.flush()
        return sighting

from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    store_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    detections: Mapped[list["Detection"]] = relationship(back_populates="photo", cascade="all, delete-orphan")


class Detection(Base):
    """One segmented spine candidate from one photo, with its full OCR + matching trail.

    A Detection row is created for EVERY spine candidate the vision pipeline
    finds, regardless of whether it was ever resolved to a real book. This is
    the evidence/debug record. `books` only gets a row once BookMatcher decides
    a Detection clears the matching bar (see matched_book_id).
    """

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id"), nullable=False)
    spine_index: Mapped[int] = mapped_column(Integer, nullable=False)

    x: Mapped[int] = mapped_column(Integer, nullable=False)
    y: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    raw_ocr_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    cleaned_ocr_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    ocr_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    match_status: Mapped[str] = mapped_column(String(32), default="NO_MATCH", nullable=False, index=True)
    match_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    match_query: Mapped[str] = mapped_column(Text, default="", nullable=False)
    match_reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    matched_book_id: Mapped[int | None] = mapped_column(ForeignKey("books.id"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    photo: Mapped[Photo] = relationship(back_populates="detections")
    matched_book: Mapped["Book | None"] = relationship(foreign_keys=[matched_book_id])
    sighting: Mapped["BookSighting | None"] = relationship(back_populates="detection", uselist=False)


class Book(Base):
    """A verified book. A row here means BookMatcher had enough evidence.

    Never insert a row purely because OCR produced some text — see
    matcher.BookMatcher and the persistence gate in main.analyze().
    """

    __tablename__ = "books"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    author: Mapped[str | None] = mapped_column(String(500), nullable=True)
    isbn: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    publisher: Mapped[str | None] = mapped_column(String(255), nullable=True)
    normalized_key: Mapped[str] = mapped_column(String(1000), nullable=False, unique=True, index=True)
    source: Mapped[str] = mapped_column(String(64), default="open_library", nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    sightings: Mapped[list["BookSighting"]] = relationship(back_populates="book")


class BookSighting(Base):
    """A confirmed sighting of an identified Book at a particular photo/store/time."""

    __tablename__ = "book_sightings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id"), nullable=False)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id"), nullable=False)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), nullable=False, unique=True)
    store_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    book: Mapped[Book] = relationship(back_populates="sightings")
    detection: Mapped[Detection] = relationship(back_populates="sighting")

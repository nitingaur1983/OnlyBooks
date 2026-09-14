import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, SCHEMA_NAME
from app.matcher import MatchResult, MatchStatus
from app.metadata_source import MetadataCandidate
from app.models import Book
from app.repository import Repository
from app.vision import SpineCandidate


@pytest.fixture
def db_session():
    # Base.metadata is schema-qualified (only_books_schema, in real Postgres).
    # SQLite has no real schema concept, so we ATTACH an in-memory database
    # under that name and use a single shared connection (StaticPool) so the
    # attachment persists across the session's queries.
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _attach_schema(dbapi_connection, connection_record):
        dbapi_connection.execute(f'ATTACH DATABASE ":memory:" AS {SCHEMA_NAME}')

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def make_spine(text="", confidence=0.0):
    return SpineCandidate(index=0, x=0, y=0, width=10, height=100, text=text, confidence=confidence)


def test_no_match_detection_creates_no_book(db_session):
    repo = Repository(db_session)
    photo = repo.create_photo("shelf.jpg", "/tmp/shelf.jpg", None)
    spine = make_spine("— _—_—_", 19.0)
    result = MatchResult(MatchStatus.NO_MATCH, 0.0, "insufficient_ocr_signal", "", None)

    detection = repo.create_detection(photo, spine, "", result)

    assert detection.matched_book_id is None
    assert db_session.query(Book).count() == 0


def test_high_confidence_detection_creates_book_and_sighting(db_session):
    repo = Repository(db_session)
    photo = repo.create_photo("shelf.jpg", "/tmp/shelf.jpg", "Test Store")
    spine = make_spine("Crossing to Safety Wallace Stegner", 80.0)
    candidate = MetadataCandidate(
        "Crossing to Safety", "Wallace Stegner", "9780679732619", None, "open_library", "/works/OL1W"
    )
    result = MatchResult(MatchStatus.HIGH_CONFIDENCE, 0.9, "title_sim=0.9...", "crossing safety", candidate)

    detection = repo.create_detection(photo, spine, "Crossing to Safety Wallace Stegner", result)
    book = repo.get_or_create_book(candidate)
    repo.link_book_to_detection(detection, book)
    sighting = repo.add_sighting(detection, book, photo, "Test Store", result.score)
    db_session.commit()

    assert db_session.query(Book).count() == 1
    assert detection.matched_book_id == book.id
    assert sighting.book_id == book.id
    assert sighting.detection_id == detection.id


def test_duplicate_isbn_does_not_create_second_book_row(db_session):
    repo = Repository(db_session)
    candidate = MetadataCandidate("Some Title", "Some Author", "9780143127550", None, "open_library", None)

    book1 = repo.get_or_create_book(candidate)
    book2 = repo.get_or_create_book(candidate)

    assert book1.id == book2.id
    assert db_session.query(Book).count() == 1

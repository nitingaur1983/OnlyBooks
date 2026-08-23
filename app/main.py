import logging
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import DEFAULT_CONFIG
from .db import Base, SessionLocal, engine
from .matcher import BookMatcher, MatchStatus, status_at_least
from .metadata_source import OpenLibraryMetadataSource
from .models import Book, BookSighting, Detection
from .ocr_cleanup import clean
from .repository import Repository
from .vision import ShelfVision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

Base.metadata.create_all(engine)

app = FastAPI(title="Local Books Vision POC", version="0.2.0")
vision = ShelfVision()
metadata_source = OpenLibraryMetadataSource(timeout_seconds=DEFAULT_CONFIG.metadata_timeout_seconds)
matcher = BookMatcher(metadata_source, DEFAULT_CONFIG)

# A Detection must reach at least this status before we persist a Book/Sighting.
# Configurable via MATCH_MIN_STATUS_TO_PERSIST (see config.py).
MIN_STATUS_TO_PERSIST = MatchStatus(DEFAULT_CONFIG.min_status_to_persist)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {
        "name": "Local Books Vision POC",
        "usage": "POST a shelf photo to /analyze; inspect /docs for the interactive UI",
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    store_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    suffix = Path(image.filename or "photo.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Use JPG, PNG, or WEBP")

    target = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    target.write_bytes(await image.read())

    try:
        spines = vision.detect_and_read(target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    repo = Repository(db)
    photo = repo.create_photo(image.filename or target.name, str(target), store_name)

    detections_out = []
    identified_out = []

    for spine in spines:
        cleaned = clean(spine.text)
        result = matcher.match(spine.text, spine.confidence)
        detection = repo.create_detection(photo, spine, cleaned.cleaned, result)

        book_payload = None
        if result.matched and status_at_least(result.status, MIN_STATUS_TO_PERSIST):
            book = repo.get_or_create_book(result.matched)
            repo.link_book_to_detection(detection, book)
            sighting = repo.add_sighting(detection, book, photo, store_name, result.score)
            book_payload = {"book_id": book.id, "title": book.title, "author": book.author, "isbn": book.isbn}
            identified_out.append(
                {
                    **book_payload,
                    "detection_id": detection.id,
                    "sighting_id": sighting.id,
                    "match_confidence": result.score,
                }
            )
        else:
            logger.info(
                "detection %s not persisted as a book (status=%s, reason=%s)",
                detection.id, result.status.value, result.reason,
            )

        detections_out.append(
            {
                "detection_id": detection.id,
                "bounding_box": {"x": spine.x, "y": spine.y, "width": spine.width, "height": spine.height},
                "raw_text": spine.text,
                "cleaned_text": cleaned.cleaned,
                "ocr_confidence": round(spine.confidence, 2),
                "match_status": result.status.value,
                "match_confidence": result.score,
                "match_reason": result.reason,
                "matched_book": book_payload,
            }
        )

    db.commit()
    return {
        "photo_id": photo.id,
        "store_name": store_name,
        "detected_count": len(detections_out),
        "identified_count": len(identified_out),
        "detections": detections_out,
        "identified_books": identified_out,
    }


@app.get("/books")
def books(db: Session = Depends(get_db)):
    rows = db.scalars(select(Book).order_by(Book.id.desc())).all()
    return [
        {"id": b.id, "title": b.title, "author": b.author, "isbn": b.isbn, "source": b.source}
        for b in rows
    ]


@app.get("/detections")
def detections(db: Session = Depends(get_db), status: str | None = None):
    """Debug endpoint: every spine candidate ever seen, matched or not.

    Pass ?status=NO_MATCH (or LOW_CONFIDENCE/POSSIBLE_MATCH/HIGH_CONFIDENCE) to
    filter, e.g. to review what's being rejected.
    """
    query = select(Detection).order_by(Detection.id.desc())
    if status:
        query = query.where(Detection.match_status == status.upper())
    rows = db.scalars(query).all()
    return [
        {
            "id": d.id,
            "photo_id": d.photo_id,
            "raw_text": d.raw_ocr_text,
            "cleaned_text": d.cleaned_ocr_text,
            "ocr_confidence": d.ocr_confidence,
            "match_status": d.match_status,
            "match_confidence": d.match_confidence,
            "match_query": d.match_query,
            "match_reason": d.match_reason,
            "matched_book_id": d.matched_book_id,
        }
        for d in rows
    ]


@app.get("/sightings")
def sightings(db: Session = Depends(get_db)):
    rows = db.scalars(select(BookSighting).order_by(BookSighting.id.desc())).all()
    return [
        {
            "id": s.id,
            "book_id": s.book_id,
            "photo_id": s.photo_id,
            "detection_id": s.detection_id,
            "store_name": s.store_name,
            "confidence": s.confidence,
            "seen_at": s.seen_at,
        }
        for s in rows
    ]

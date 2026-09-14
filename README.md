# Local Books Vision POC

A deliberately small proof of concept for turning a bookstore shelf photo into searchable, **verified** book sightings.

## What it currently does

1. Accepts a JPG/PNG/WEBP shelf photo.
2. Uses classical OpenCV heuristics to find tall, spine-like regions.
3. OCRs each candidate in normal, clockwise, and counter-clockwise orientations with Tesseract.
4. Cleans the OCR text and scores it against Open Library metadata (title/author fuzzy match, ISBN exact match).
5. **Only** creates a `Book`/`BookSighting` row if the evidence clears a confidence bar. Every candidate — matched or not — is preserved as a `Detection` for debugging.

This is **not yet a production-quality spine detector or book resolver**. The point of v0.1 is to measure OCR/detection quality on real bookstore photos before choosing heavier models. v0.2 adds the identification/verification layer described below.

## Architecture

```text
Shelf photo
   ↓
FastAPI /analyze
   ↓
ShelfVision (OpenCV spine candidates + Tesseract OCR)      -- evidence
   ↓
ocr_cleanup.clean()  (strip symbol noise, measure signal)  -- evidence
   ↓
BookMatcher (matcher.py)                                   -- the ONLY "is this a book?" gate
   ├─ ISBN regex → MetadataSource.lookup_isbn()  (exact match)
   └─ title/author tokens → MetadataSource.search() + fuzzy scoring
   ↓                                              ↑
Repository ── always writes Detection    OpenLibraryMetadataSource (metadata_source.py)
   └─ writes Book + BookSighting only if match_status ≥ MATCH_MIN_STATUS_TO_PERSIST
   ↓
Postgres (only_books_db): photos → detections → books ← book_sightings
```

Each concern is its own module so pieces can be swapped independently:

- `vision.py` — spine detection + OCR. No knowledge of matching or metadata.
- `ocr_cleanup.py` — symbol-noise stripping and signal quality (`alpha_ratio`, token count).
- `matcher.py` (`BookMatcher`) — decides `NO_MATCH` / `LOW_CONFIDENCE` / `POSSIBLE_MATCH` / `HIGH_CONFIDENCE`. Never talks to a specific API — only to the `MetadataSource` interface.
- `metadata_source.py` — `OpenLibraryMetadataSource` (free, keyless). Swap in Google Books/ISBNdb/a local catalog by writing another class here.
- `repository.py` — persistence. Always writes a `Detection`; only writes `Book`/`BookSighting` when the matcher's status clears the configured bar.
- `config.py` — every threshold as an env-overridable default (see below).

### Database schema

| Table | Purpose |
|---|---|
| `photos` | One row per uploaded shelf photo. |
| `detections` | One row per **segmented spine candidate**, always written. Holds `raw_ocr_text`, `cleaned_ocr_text`, `ocr_confidence`, `match_status`, `match_confidence`, `match_query`, `match_reason`, and `matched_book_id` (nullable — null means "no book, but here's why"). This is the debug/evidence trail. |
| `books` | Only rows that cleared the matching bar. Deduped by ISBN when known, else normalized title+author. |
| `book_sightings` | Links a `Detection` (1:1) to a `Book`, `Photo`, and optional `store_name`/`seen_at`/`confidence`. Only exists for identified books. |

**A `books` row is never created from OCR text alone** — see `BookMatcher.match()` in `matcher.py`.

### Confidence thresholds (env-configurable, see `app/config.py`)

| Env var | Default | Meaning |
|---|---|---|
| `MATCH_HIGH_THRESHOLD` | `0.75` | Composite score to reach `HIGH_CONFIDENCE` |
| `MATCH_POSSIBLE_THRESHOLD` | `0.55` | Composite score to reach `POSSIBLE_MATCH` |
| `MATCH_LOW_THRESHOLD` | `0.35` | Composite score to reach `LOW_CONFIDENCE` |
| `MATCH_MIN_STATUS_TO_PERSIST` | `POSSIBLE_MATCH` | Minimum status that creates a `Book`/`BookSighting` |
| `OCR_MIN_ALPHA_TOKENS` | `2` | Min real-word tokens (len ≥ 3) before we even query metadata |
| `OCR_MIN_ALPHA_RATIO` | `0.35` | Min fraction of raw characters that must be letters |
| `MATCH_DISTINCTIVE_MIN_LEN` | `5` | Min token length to count as "distinctive" evidence (blocks matches on words like "the"/"new"/"press") |
| `METADATA_SEARCH_LIMIT` | `5` | Max Open Library search results considered |
| `METADATA_TIMEOUT_SECONDS` | `4.0` | Timeout per Open Library call |

## Database setup (Postgres)

The app persists to Postgres, not SQLite. Create the local database, role, and
privileges once:

```sql
CREATE DATABASE only_books_db;
CREATE USER app_user;
ALTER USER app_user WITH PASSWORD 'jaiganesh219@';
GRANT ALL PRIVILEGES ON DATABASE only_books_db TO app_user;
```

All app tables live in a dedicated schema, `only_books_schema` — **not**
Postgres's default `public` schema. Create it and make it `app_user`'s
default so unqualified SQL (`SELECT * FROM books`) just works in `psql`:

```sql
CREATE SCHEMA IF NOT EXISTS only_books_schema AUTHORIZATION app_user;
GRANT ALL ON SCHEMA only_books_schema TO app_user;
ALTER ROLE app_user IN DATABASE only_books_db SET search_path TO only_books_schema, public;
```

Verify you can connect (reconnect after the `ALTER ROLE` above for the new
`search_path` to take effect):

```bash
psql -d only_books_db -U app_user
```
```sql
set search_path=only_books_schema;
```

`app/db.py` defaults to exactly this local setup (`localhost:5432`, `app_user` /
`only_books_db` / `only_books_schema`). Override any part with env vars if your
setup differs:

| Env var | Default |
|---|---|
| `PGHOST` | `localhost` |
| `PGPORT` | `5432` |
| `PGDATABASE` | `only_books_db` |
| `PGUSER` | `app_user` |
| `PGPASSWORD` | `jaiganesh219@` |
| `PGSCHEMA` | `only_books_schema` |
| `DATABASE_URL` | *(unset — takes priority over all of the above if set, e.g. for a hosted DB)* |

`app.main` calls `Base.metadata.create_all(engine)` on startup, so tables
(`photos`, `detections`, `books`, `book_sightings`) are created automatically
under `only_books_schema` the first time the app runs against a fresh
database — no manual migration step needed for this POC.

## Run

Python 3.11+ is recommended. Tesseract must also be installed on the machine (`brew install tesseract` on macOS).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

#To kill the existing process
lsof -tiTCP:8000 -sTCP:LISTEN | xargs kill
#To restart the App
mkdir -p /Users/nitin/Assembly/Projects/OnlyBooks/logs
nohup .venv/bin/uvicorn app.main:app --reload >  /Users/nitin/Assembly/Projects/OnlyBooks/logs/onlybooks.log 2>&1 &
```

For running tests, install dev deps instead: `pip install -r requirements-dev.txt`, then `pytest`.

Then open:

```text
http://127.0.0.1:8000/docs
```

Use `POST /analyze`, choose a shelf image, optionally enter a store name, and execute it. The response separates raw `detections` (everything found, matched or not) from `identified_books` (only what cleared the confidence bar):

```json
{
  "photo_id": 1,
  "detected_count": 24,
  "identified_count": 9,
  "detections": [
    {
      "detection_id": 5,
      "raw_text": "Crossing to Safety Wallace Stegner",
      "cleaned_text": "Crossing to Safety Wallace Stegner",
      "ocr_confidence": 82.0,
      "match_status": "HIGH_CONFIDENCE",
      "match_confidence": 0.83,
      "match_reason": "title_sim=0.72 token_overlap=1.00 ocr_conf=82 distinctive_hit=True candidate='Crossing to Safety...' ",
      "matched_book": {"book_id": 3, "title": "Crossing to Safety", "author": "Wallace Stegner", "isbn": null}
    },
    {
      "detection_id": 6,
      "raw_text": "— _—_—_—_—_ lL —",
      "cleaned_text": "",
      "ocr_confidence": 19.0,
      "match_status": "NO_MATCH",
      "match_reason": "insufficient_ocr_signal",
      "matched_book": null
    }
  ],
  "identified_books": [
    {"book_id": 3, "title": "Crossing to Safety", "author": "Wallace Stegner", "isbn": null, "detection_id": 5, "sighting_id": 2, "match_confidence": 0.83}
  ]
}
```

You can also inspect:

- `GET /books` — only verified books.
- `GET /detections` (optionally `?status=NO_MATCH|LOW_CONFIDENCE|POSSIBLE_MATCH|HIGH_CONFIDENCE`) — full evidence trail, including rejected candidates and *why* they were rejected.
- `GET /sightings`
- Postgres DB: `only_books_db` (see [Database setup](#database-setup-postgres))
- Uploaded images: `data/uploads/`

### Inspecting the database directly

```sql
-- All detections (evidence), most recent first
SELECT id, photo_id, raw_ocr_text, cleaned_ocr_text, ocr_confidence,
       match_status, match_confidence, match_reason, matched_book_id
FROM detections
ORDER BY id DESC;

-- Successfully identified books
SELECT id, title, author, isbn, publisher, source, external_id
FROM books
ORDER BY id DESC;

-- Low-confidence / unmatched detections (candidates that were NOT turned into books)
SELECT id, photo_id, raw_ocr_text, cleaned_ocr_text, ocr_confidence,
       match_status, match_confidence, match_reason
FROM detections
WHERE match_status IN ('NO_MATCH', 'LOW_CONFIDENCE')
ORDER BY id DESC;

-- Book sightings with store/time context
SELECT s.id, b.title, b.author, s.store_name, s.confidence, s.seen_at
FROM book_sightings s
JOIN books b ON b.id = s.book_id
ORDER BY s.id DESC;
```

## What to test first

Take one reasonably straight shelf photo containing roughly 10–25 books. Avoid glare for the first test. We want to measure:

- How many real spines were segmented?
- How many produced readable text?
- How many titles could be correctly resolved (`identified_count` vs `detected_count`)?
- Which failure modes dominate: vertical text, decorative fonts, tiny text, glare, or segmentation — check `GET /detections?status=NO_MATCH` for the reasons.

Once we have 3–5 real photos and those numbers, the next engineering choice — improving spine segmentation vs. OCR preprocessing vs. matching thresholds — will be evidence-driven rather than speculative.

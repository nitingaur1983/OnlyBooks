# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI proof of concept that turns a bookstore shelf photo into **verified** book sightings: OpenCV finds spine-shaped regions, Tesseract OCRs them, and Open Library metadata decides whether the text is actually a book. README.md covers DB setup, endpoints, the response shape and handy SQL queries.

## Commands

Tesseract has to be installed on the machine (`brew install tesseract`). The venv lives at `.venv/`.

```bash
pip install -r requirements-dev.txt      # runtime deps + pytest
pytest                                   # all tests
pytest tests/test_matcher.py::test_name  # a single test
uvicorn app.main:app --reload            # dev server; interactive UI at http://127.0.0.1:8000/docs

# Restart the background server (as the README does it); logs go to ../logs/onlybooks.log
lsof -tiTCP:8000 -sTCP:LISTEN | xargs kill
nohup .venv/bin/uvicorn app.main:app --reload > /Users/nitin/Assembly/Projects/OnlyBooks/logs/onlybooks.log 2>&1 &
```

The project has no linter or formatter set up.

## Architecture

Request flow for `POST /analyze` (`app/main.py`, the place where everything is wired together):

`ShelfVision.detect_and_read()` → for each spine: `ocr_cleanup.clean()` → `BookMatcher.match()` → `Repository` writes rows.

- **`vision.py`**: a two-stage pipeline. It first finds shelf rows (looking for long horizontal edge runs), then runs spine detection **separately inside each shelf crop**. That is why all the spine size ratios in `VisionConfig` are measured against the shelf's height, not the photo's. OCR tries 3 rotations per crop. Debug images (`01_gray` … `07_annotated`, `crops/`, `shelves/`) go to `data/debug/<upload-id>/` only when you pass `debug=true` to `/analyze` or set `VISION_DEBUG=1`. `VisionResult.debug` (`DebugSummary`) is always filled in and returned as `vision_debug`. This module knows nothing about matching.
- **`matcher.py`** (`BookMatcher`): the **only** code that decides whether a candidate is a book. It returns `NO_MATCH` / `LOW_CONFIDENCE` / `POSSIBLE_MATCH` / `HIGH_CONFIDENCE`, using an ISBN exact lookup or fuzzy title/author scoring with a "distinctive token" guard. It only talks to the `MetadataSource` protocol, never to a specific API.
- **`metadata_source.py`**: `OpenLibraryMetadataSource` (no API key needed). To add another catalog, write another class that implements `search()` and `lookup_isbn()`.
- **`repository.py`**: always writes a `Detection` (the evidence trail, including rejected candidates and the reason). It writes `Book` + `BookSighting` only when the status reaches at least `MATCH_MIN_STATUS_TO_PERSIST`. Books are deduplicated by ISBN, or else by normalized title+author.
- **`config.py`**: every threshold is a frozen-dataclass default that an env var can override (`MatchConfig`, `VisionConfig`). Add new knobs here and don't hardcode them.

Invariant to keep: a `books` row must never be created from OCR text alone.

### Experimental VLM pipeline (`POST /analyze/vlm`, `app/vlm/`)

A separate Gemini-based detector registered in `main.py` via `app.include_router(vlm_router)`. It must stay independent: it doesn't import `vision.py`, cv2, pytesseract, the matcher or the DB, and it is read-only toward the database (output goes to the response and `data/debug/vlm/<request_id>/`). `service.py` runs the pipeline; `geometry.py` holds the pure logic (tiling, Gemini `box_2d` [ymin, xmin, ymax, xmax] 0–1000 → pixel conversion, cross-tile dedupe); providers implement the `provider.VLMProvider` protocol (`gemini_provider.py` is the only one). Settings come from `.env` (loaded once at import by `app/vlm/config.py`, so restart after editing it); see `.env.example`. Tests in `tests/test_vlm.py` use a stub provider and make no API calls.

## Database

- The app uses Postgres (`only_books_db`). All tables live in the **`only_books_schema`** schema, not `public`, because `Base.metadata` in `app/db.py` is schema-qualified. Connection settings come from `DATABASE_URL` or `PG*` env vars; the defaults are in `app/db.py`.
- On startup, `Base.metadata.create_all(engine)` creates the tables. There are no migrations, so changing a column on an existing table means altering or dropping it by hand.
- `data/books.db` / `data/bookclub.db` are old SQLite files. The app doesn't use them.
- Tests don't need Postgres. `tests/test_repository.py` uses in-memory SQLite and `ATTACH`es a database named after `SCHEMA_NAME` so that schema-qualified tables still work. Matcher tests use a stub `MetadataSource`, so nothing touches the network.

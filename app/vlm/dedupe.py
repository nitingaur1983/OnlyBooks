"""Physical-book deduplication across overlapping tiles.

Pure logic: no I/O, no provider calls, no database. Used by service.py for
live requests and by replay.py for offline re-runs of saved responses.

Core idea: compare observations only inside the region both tiles saw
---------------------------------------------------------------------
A book cut by a tile edge is reported by that tile with a truncated box, while
the neighbouring tile reports the full box. Comparing the two boxes directly
makes a perfect duplicate look like a weak overlap (IoU ~0.4). But inside the
*shared zone* (tile A ∩ tile B) both tiles saw the same pixels, so the same
physical book must occupy (almost) the same box there. Every geometric test
below runs on boxes clipped to that shared zone.

Decisions per pair of observations from different tiles:
  geometry  strong    : shared-zone IoU >= strong_iou
            supported : shared-zone IoU >= supported_iou (enough only with text support)
            medium    : shared-zone IoU >= medium_iou
            weak      : below that
  text      same_title    : titles agree (partial reads count)
            complementary : one side's text is found on the other (e.g. an
                            author-only read of a cut spine vs the full spine)
            consistent    : neither has a title, texts agree
            unconfirmed   : texts neither agree nor contradict (publisher logo)
            no_text       : at least one side has no legible text
            conflict      : different titles, or different authors
  merge     : strong geometry and no text conflict, or supported geometry
              with supportive text (same_title / complementary / consistent).
  ambiguous : strong geometry with a text conflict, supported/medium geometry
              without decisive text, or a weak overlap where one box sits
              inside the other. Never merged; flagged for review.

Thresholds come from the 2026-11-15 scan (1793a94d…): adjacent *different*
books reach shared-zone IoU 0.55-0.65 because spine boxes are narrow and
noisy, so geometry alone must clear 0.7.
  separate  : everything else.

Clusters are built greedily from the best merge pairs with two guards that
prevent chain merging: a cluster holds at most one observation per tile (a
tile reports each book once), and every pair inside a cluster must itself be a
"merge" pair (complete linkage) — A~B and B~C never pulls A and C together
unless A~C holds too.

Same title alone never merges anything, and containment alone never merges
anything: both need the shared-zone geometry to agree.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from .geometry import Tile, iou
from .schemas import BBox, DuplicateRef, Readability, VLMDetection

ALGORITHM_VERSION = "shared-zone-v1"


@dataclass(frozen=True)
class DedupeConfig:
    strong_iou: float = 0.7  # shared-zone IoU that alone (absent text conflict) proves one book
    supported_iou: float = 0.5  # enough when the text agrees too
    medium_iou: float = 0.35  # gray zone lower bound: flagged, never merged
    ambiguous_containment: float = 0.8  # weak overlap but one box inside the other -> review
    title_match: float = 0.75  # title similarity counted as "same title"
    author_match: float = 0.6  # below this, two legible authors contradict
    text_coverage: float = 0.75  # share of one side's words found on the other -> complementary
    min_shared_px: int = 12  # clipped boxes thinner than this carry no evidence
    edge_tolerance_px: int = 6  # box side this close to an interior tile edge counts as cut off
    competing_margin: float = 0.1  # second-best match this close to the best -> review

    def as_dict(self) -> dict:
        return asdict(self)


# --- observations -----------------------------------------------------------------


@dataclass
class Observation:
    """One detection as reported by one tile, with a stable id `t<tile>:<index>`."""

    obs_id: str
    det: VLMDetection
    tile: Tile
    cut_edges: tuple[str, ...]  # sides of the box lying on an interior tile edge

    @property
    def bbox(self) -> BBox:
        return self.det.bbox


def observation_id(tile_id: int | None, index: int) -> str:
    return f"t{tile_id}:{index}"


def _cut_edges(b: BBox, tile: Tile, image_w: int, image_h: int, tol: int) -> tuple[str, ...]:
    edges = []
    if tile.x > 0 and b.x - tile.x <= tol:
        edges.append("left")
    if tile.y > 0 and b.y - tile.y <= tol:
        edges.append("top")
    if tile.x + tile.width < image_w and tile.x + tile.width - b.x2 <= tol:
        edges.append("right")
    if tile.y + tile.height < image_h and tile.y + tile.height - b.y2 <= tol:
        edges.append("bottom")
    return tuple(edges)


def build_observations(detections: list[VLMDetection], tiles: list[Tile], config: DedupeConfig) -> list[Observation]:
    """`detections` must carry tile_id and their tile-local `index` (as parsed,
    before any dedupe renumbering)."""
    by_id = {t.tile_id: t for t in tiles}
    image_w = max(t.x + t.width for t in tiles)
    image_h = max(t.y + t.height for t in tiles)
    obs = []
    for d in detections:
        tile = by_id[d.tile_id]
        obs.append(Observation(observation_id(d.tile_id, d.index), d, tile,
                               _cut_edges(d.bbox, tile, image_w, image_h, config.edge_tolerance_px)))
    obs.sort(key=lambda o: (o.det.tile_id, o.det.index))
    return obs


# --- geometry -----------------------------------------------------------------------


def _intersect(a: BBox, b: BBox) -> BBox | None:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def _tile_box(t: Tile) -> BBox:
    return BBox(x=t.x, y=t.y, width=t.width, height=t.height)


def _axis_iou(a1: int, a2: int, b1: int, b2: int) -> float:
    inter = max(min(a2, b2) - max(a1, b1), 0)
    union = max(a2, b2) - min(a1, b1)
    return inter / union if union else 0.0


def _geometry(a: Observation, b: Observation, config: DedupeConfig) -> dict:
    inter = _intersect(a.bbox, b.bbox)
    inter_area = inter.area if inter else 0
    smaller = min(a.bbox.area, b.bbox.area)
    acx, acy = a.bbox.x + a.bbox.width / 2, a.bbox.y + a.bbox.height / 2
    bcx, bcy = b.bbox.x + b.bbox.width / 2, b.bbox.y + b.bbox.height / 2
    geo = {
        "iou": round(iou(a.bbox, b.bbox), 3),
        "intersection_over_smaller": round(inter_area / smaller, 3) if smaller else 0.0,
        "center_distance_px": round(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5, 1),
        "shared_zone": None,
        "shared_iou": 0.0,
        "shared_x_overlap": 0.0,
        "shared_y_overlap": 0.0,
        "shared_width_ratio": 0.0,
        "shared_height_ratio": 0.0,
    }
    zone = _intersect(_tile_box(a.tile), _tile_box(b.tile))
    if zone is None:
        return geo
    geo["shared_zone"] = zone.model_dump()
    ca, cb = _intersect(a.bbox, zone), _intersect(b.bbox, zone)
    if ca is None or cb is None:
        return geo
    if min(ca.width, ca.height, cb.width, cb.height) < config.min_shared_px:
        geo["shared_too_small"] = True
    geo.update({
        "shared_iou": round(iou(ca, cb), 3),
        "shared_x_overlap": round(_axis_iou(ca.x, ca.x2, cb.x, cb.x2), 3),
        "shared_y_overlap": round(_axis_iou(ca.y, ca.y2, cb.y, cb.y2), 3),
        "shared_width_ratio": round(min(ca.width, cb.width) / max(ca.width, cb.width), 3),
        "shared_height_ratio": round(min(ca.height, cb.height) / max(ca.height, cb.height), 3),
    })
    return geo


# --- text ---------------------------------------------------------------------------

_STOPWORDS = {"the", "a", "an", "of", "and", "in", "on", "to", "for", "by", "with", "at", "from", "is"}


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _words(text: str) -> list[str]:
    words = text.split()
    informative = [w for w in words if len(w) >= 3 and w not in _STOPWORDS]
    return informative or words


def _word_found(word: str, pool: list[str]) -> bool:
    for p in pool:
        if word == p:
            return True
        # Partial reads at a cut edge: "urt" ~ "kurt", "schlesse" ~ "schlesser".
        if len(word) >= 3 and len(p) >= 3 and (word in p or p in word):
            return True
        if len(word) >= 4 and SequenceMatcher(None, word, p).ratio() >= 0.8:
            return True
    return False


def _coverage(text: str, pool_text: str) -> float:
    words, pool = _words(text), pool_text.split()
    if not words or not pool:
        return 0.0
    return sum(_word_found(w, pool) for w in words) / len(words)


def _similarity(a: str, b: str) -> float:
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _texts(d: VLMDetection) -> dict:
    title, author, raw = _norm(d.visible_title), _norm(d.visible_author), _norm(d.raw_visible_text)
    return {"title": title, "author": author, "all": " ".join(t for t in (title, author, raw) if t)}


def text_relation(a: VLMDetection, b: VLMDetection, config: DedupeConfig) -> dict:
    ta, tb = _texts(a), _texts(b)
    out: dict = {"title_similarity": None, "author_similarity": None}
    if ta["author"] and tb["author"]:
        out["author_similarity"] = round(_similarity(ta["author"], tb["author"]), 3)
    # "Lydia" read as an author is not a contradiction if the other spine's
    # title is "The Scandalous Confessions of Lydia Bennet".
    authors_conflict = (out["author_similarity"] is not None and out["author_similarity"] < config.author_match
                        and _coverage(ta["author"], tb["all"]) < config.text_coverage
                        and _coverage(tb["author"], ta["all"]) < config.text_coverage)

    if not ta["all"] or not tb["all"]:
        return {**out, "relation": "no_text", "detail": "at least one observation has no legible text"}

    if ta["title"] and tb["title"]:
        sim = _similarity(ta["title"], tb["title"])
        out["title_similarity"] = round(sim, 3)
        if sim >= config.title_match:
            detail = "titles agree" + ("; authors differ" if authors_conflict else "")
            return {**out, "relation": "same_title", "detail": detail}
        # A "title" that is really the other spine's author (or vice versa).
        if _coverage(ta["title"], tb["all"]) >= config.text_coverage:
            return {**out, "relation": "complementary", "detail": f"title '{ta['title']}' found in other's text"}
        if _coverage(tb["title"], ta["all"]) >= config.text_coverage:
            return {**out, "relation": "complementary", "detail": f"title '{tb['title']}' found in other's text"}
        return {**out, "relation": "conflict", "detail": f"titles differ: '{ta['title']}' vs '{tb['title']}'"}

    if authors_conflict:
        return {**out, "relation": "conflict", "detail": f"authors differ: '{ta['author']}' vs '{tb['author']}'"}

    if ta["title"] or tb["title"]:
        titled, other = (ta, tb) if ta["title"] else (tb, ta)
        if _coverage(other["all"], titled["all"]) >= config.text_coverage:
            return {**out, "relation": "complementary", "detail": f"'{other['all']}' found alongside title '{titled['title']}'"}
        return {**out, "relation": "unconfirmed", "detail": f"'{other['all']}' not found on '{titled['all']}'"}

    if _coverage(ta["all"], tb["all"]) >= config.text_coverage or _coverage(tb["all"], ta["all"]) >= config.text_coverage:
        return {**out, "relation": "consistent", "detail": "untitled texts agree"}
    return {**out, "relation": "unconfirmed", "detail": f"'{ta['all']}' vs '{tb['all']}'"}


# --- pair decisions -------------------------------------------------------------------

_SUPPORTIVE = {"same_title", "complementary", "consistent"}


def evaluate_pair(a: Observation, b: Observation, config: DedupeConfig) -> dict:
    geo = _geometry(a, b, config)
    text = text_relation(a.det, b.det, config)
    rel = text["relation"]
    s = geo["shared_iou"]
    if geo.get("shared_too_small") or geo["shared_zone"] is None:
        strength = "none"
    elif s >= config.strong_iou:
        strength = "strong"
    elif s >= config.supported_iou:
        strength = "supported"
    elif s >= config.medium_iou:
        strength = "medium"
    else:
        strength = "weak"

    if strength == "strong" and rel != "conflict":
        decision, reason = "merge", f"same box in shared zone (IoU {s}); text {rel}"
    elif strength == "strong":
        decision, reason = "ambiguous", f"same box in shared zone (IoU {s}) but {text['detail']}"
    elif strength == "supported" and rel in _SUPPORTIVE:
        decision, reason = "merge", f"good box agreement (IoU {s}) and text {rel}: {text['detail']}"
    elif strength in ("supported", "medium") and rel != "conflict":
        decision, reason = "ambiguous", f"partial box agreement (IoU {s}); text {rel} does not settle it"
    elif strength in ("supported", "medium"):
        decision, reason = "separate", f"partial box agreement (IoU {s}) and {text['detail']}"
    elif geo["intersection_over_smaller"] >= config.ambiguous_containment and rel != "conflict":
        decision, reason = "ambiguous", (f"one box inside the other ({geo['intersection_over_smaller']}) but shapes "
                                         f"disagree in shared zone (IoU {s}); possible box spanning several books")
    else:
        decision, reason = "separate", f"boxes disagree in shared zone (IoU {s})"

    return {
        "a": a.obs_id,
        "b": b.obs_id,
        "decision": decision,
        "reason": reason,
        "geometry_strength": strength,
        "text_relation": rel,
        "text_detail": text["detail"],
        "title_similarity": text["title_similarity"],
        "author_similarity": text["author_similarity"],
        **geo,
        "a_text": _texts(a.det)["all"] or None,
        "b_text": _texts(b.det)["all"] or None,
        "a_bbox": a.bbox.model_dump(),
        "b_bbox": b.bbox.model_dump(),
        "score": round(s + (0.05 if rel in _SUPPORTIVE else 0.0), 4),
    }


def _candidate_pairs(obs: list[Observation]):
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            if a.det.tile_id == b.det.tile_id:
                continue
            if _intersect(a.bbox, b.bbox) is not None:
                yield a, b


# --- clustering ---------------------------------------------------------------------


_READABILITY_RANK = {Readability.CLEAR: 3, Readability.PARTIAL: 2, Readability.UNREADABLE: 1, Readability.NOT_VISIBLE: 0}


def _representative(members: list[Observation]) -> Observation:
    # Best-read title, then an uncut box, then the larger box; id breaks ties.
    return sorted(members, key=lambda o: (-_READABILITY_RANK[o.det.title_readability], len(o.cut_edges),
                                          -o.bbox.area, o.det.tile_id, o.det.index))[0]


def canonical_bbox(members: list[Observation]) -> BBox:
    """Each side from observations that were NOT cut off on that side (median),
    falling back to the most extreme cut value when every observation is cut."""
    def side(name: str, value, extreme):
        reliable = sorted(value(o) for o in members if name not in o.cut_edges)
        return statistics.median_low(reliable) if reliable else extreme(value(o) for o in members)

    x1 = side("left", lambda o: o.bbox.x, min)
    y1 = side("top", lambda o: o.bbox.y, min)
    x2 = side("right", lambda o: o.bbox.x2, max)
    y2 = side("bottom", lambda o: o.bbox.y2, max)
    if x2 <= x1 or y2 <= y1:  # disagreeing medians; fall back to the representative
        return _representative(members).bbox
    return BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


@dataclass
class DedupeResult:
    books: list[VLMDetection]
    report: dict
    pair_decisions: list[dict] = field(default_factory=list)  # notable pairs, for diagnostics
    ambiguous_pairs: list[dict] = field(default_factory=list)
    all_pairs: list[dict] = field(default_factory=list)  # every overlapping cross-tile pair


def _obs_record(o: Observation) -> dict:
    d = o.det
    return {
        "observation_id": o.obs_id,
        "tile_id": d.tile_id,
        "tile_local_index": d.index,
        "bbox": d.bbox.model_dump(),
        "model_box_2d": d.model_box_2d,
        "cut_edges": list(o.cut_edges),
        "visible_title": d.visible_title,
        "visible_author": d.visible_author,
        "raw_visible_text": d.raw_visible_text,
        "title_readability": d.title_readability.value,
        "author_readability": d.author_readability.value,
        "uncertainty_notes": d.uncertainty_notes,
    }


def dedupe_physical_books(detections: list[VLMDetection], tiles: list[Tile],
                          config: DedupeConfig = DedupeConfig()) -> DedupeResult:
    """Group per-tile observations into physical books. Input detections keep
    their tile-local `index`; they are not mutated. Output order is by
    position (top-to-bottom, left-to-right) and fully deterministic."""
    obs = build_observations(detections, tiles, config)
    by_id = {o.obs_id: o for o in obs}

    pairs = [evaluate_pair(a, b, config) for a, b in _candidate_pairs(obs)]
    pair_index = {(p["a"], p["b"]): p for p in pairs}

    def pair(x: str, y: str) -> dict | None:
        return pair_index.get((x, y)) or pair_index.get((y, x))

    # Flag observations with two near-equal merge partners in the same tile.
    competing: dict[str, list[str]] = {}
    merge_pairs = [p for p in pairs if p["decision"] == "merge"]
    for o in obs:
        options: dict[int, list[tuple[float, str]]] = {}
        for p in merge_pairs:
            if o.obs_id in (p["a"], p["b"]):
                other = p["b"] if p["a"] == o.obs_id else p["a"]
                options.setdefault(by_id[other].det.tile_id, []).append((p["score"], other))
        for tile_id, opts in options.items():
            opts.sort(reverse=True)
            if len(opts) > 1 and opts[0][0] - opts[1][0] < config.competing_margin:
                competing[o.obs_id] = [opts[0][1], opts[1][1]]

    # Greedy complete-linkage clustering over merge pairs, best evidence first.
    cluster_of = {o.obs_id: i for i, o in enumerate(obs)}
    clusters: dict[int, list[str]] = {i: [o.obs_id] for i, o in enumerate(obs)}
    blocked = []
    for p in sorted(merge_pairs, key=lambda p: (-p["score"], p["a"], p["b"])):
        ca, cb = cluster_of[p["a"]], cluster_of[p["b"]]
        if ca == cb:
            p["applied"] = True
            continue
        ma, mb = clusters[ca], clusters[cb]
        tiles_a = {by_id[x].det.tile_id for x in ma}
        tiles_b = {by_id[x].det.tile_id for x in mb}
        why = None
        if tiles_a & tiles_b:
            why = f"would put two observations from tile(s) {sorted(tiles_a & tiles_b)} into one book"
        else:
            bad = [(x, y) for x in ma for y in mb if (pair(x, y) or {}).get("decision") != "merge"]
            if bad:
                why = f"members {bad[0][0]} and {bad[0][1]} are not a merge pair (complete linkage)"
        if why:
            p["applied"] = False
            p["blocked_reason"] = why
            blocked.append(p)
            continue
        p["applied"] = True
        keep, drop = min(ca, cb), max(ca, cb)
        clusters[keep] = sorted(ma + mb, key=lambda x: (by_id[x].det.tile_id, by_id[x].det.index))
        for x in clusters.pop(drop):
            cluster_of[x] = keep

    # A blocked merge is settled when each side already joined a clearly
    # better partner (usually the neighbouring book seen at a slight offset);
    # it stays an open question only when it was nearly as good.
    best_applied: dict[str, float] = {}
    for p in merge_pairs:
        if p.get("applied"):
            for x in (p["a"], p["b"]):
                best_applied[x] = max(best_applied.get(x, 0.0), p["score"])
    contested = []
    for p in blocked:
        rival = max(best_applied.get(p["a"], 0.0), best_applied.get(p["b"], 0.0))
        p["superseded"] = p["score"] < rival - config.competing_margin
        if not p["superseded"] and cluster_of[p["a"]] != cluster_of[p["b"]]:
            contested.append(p)

    # Same for a gray-zone pair whose two observations each merged elsewhere
    # with clearly stronger evidence. A strong-geometry text conflict always
    # stays open: two different titles in the same box is the question itself.
    def settled(p: dict) -> bool:
        return (p["geometry_strength"] != "strong" and
                min(best_applied.get(p["a"], 0.0), best_applied.get(p["b"], 0.0)) >= p["score"] + config.competing_margin)

    gray = [p for p in pairs if p["decision"] == "ambiguous" and cluster_of[p["a"]] != cluster_of[p["b"]]]
    for p in gray:
        p["superseded"] = settled(p)
    ambiguous = [p for p in gray if not p["superseded"]]
    ambiguous += [{**p, "decision": "ambiguous", "reason": f"merge blocked: {p['blocked_reason']}"} for p in contested]
    ambiguous.sort(key=lambda p: (p["a"], p["b"]))

    review_reasons: dict[str, list[str]] = {}
    for p in ambiguous:
        for x, y in ((p["a"], p["b"]), (p["b"], p["a"])):
            review_reasons.setdefault(x, []).append(f"ambiguous with {y}: {p['reason']}")
    for x, (o1, o2) in competing.items():
        review_reasons.setdefault(x, []).append(f"near-equal matches {o1} and {o2}")

    books: list[VLMDetection] = []
    book_records: list[dict] = []
    for members_ids in clusters.values():
        members = [by_id[x] for x in members_ids]
        rep = _representative(members)
        bbox = canonical_bbox(members)
        used = [pair(x, y) for i, x in enumerate(members_ids) for y in members_ids[i + 1:]]
        used = [p for p in used if p]
        reasons = sorted({r for x in members_ids for r in review_reasons.get(x, [])})
        # Evidence that didn't contradict enough to block the merge, kept so a
        # reviewer sees e.g. an author read as a title. Nothing is reconciled:
        # the representative's own title/author are reported as-is.
        conflicts = sorted({f"{p['a']}~{p['b']}: text {p['text_relation']}: {p['text_detail']}" for p in used
                            if p["text_relation"] != "same_title" or "authors differ" in p["text_detail"]})
        titles = sorted({m.det.visible_title for m in members if m.det.visible_title})
        authors = sorted({m.det.visible_author for m in members if m.det.visible_author})

        book = rep.det.model_copy(deep=True)
        book.bbox = bbox
        book.observation_ids = list(members_ids)
        book.needs_review = bool(reasons)
        book.review_reasons = reasons
        if bbox != rep.bbox:
            book.warnings = sorted(set(book.warnings) | {"bbox_is_canonical_merge_of_observations"})
        book.merged_duplicates = []
        for m in members:
            if m is rep:
                continue
            p = pair(rep.obs_id, m.obs_id) or {}
            book.merged_duplicates.append(DuplicateRef(
                tile_id=m.det.tile_id, model_box_2d=m.det.model_box_2d, bbox=m.bbox, visible_title=m.det.visible_title,
                iou=p.get("iou", round(iou(rep.bbox, m.bbox), 3)), containment=p.get("intersection_over_smaller", 0.0),
                text_similarity=p.get("title_similarity"),
            ))
        books.append(book)
        book_records.append({
            "canonical_bbox": bbox.model_dump(),
            "representative_observation": rep.obs_id,
            "observation_ids": list(members_ids),
            "source_tiles": sorted({m.det.tile_id for m in members}),
            "observed_titles": titles,
            "observed_authors": authors,
            "title_readabilities": [m.det.title_readability.value for m in members],
            "merge_decisions": [{k: p[k] for k in ("a", "b", "reason", "geometry_strength", "text_relation", "shared_iou")}
                                for p in used],
            "conflicting_evidence": conflicts,
            "needs_review": book.needs_review,
            "review_reasons": reasons,
            "observations": [_obs_record(m) for m in members],
        })

    order = sorted(range(len(books)), key=lambda i: (books[i].bbox.y, books[i].bbox.x, books[i].observation_ids[0]))
    books = [books[i] for i in order]
    book_records = [book_records[i] for i in order]
    for i, (b, rec) in enumerate(zip(books, book_records)):
        b.index = i
        rec["book_index"] = i

    counts: dict[str, int] = {}
    for p in pairs:
        counts[p["decision"]] = counts.get(p["decision"], 0) + 1
    report = {
        "algorithm": ALGORITHM_VERSION,
        "config": config.as_dict(),
        "summary": {
            "observations": len(obs),
            "books": len(books),
            "observations_merged_away": len(obs) - len(books),
            "books_from_multiple_observations": sum(1 for b in books if len(b.observation_ids) > 1),
            "books_needing_review": sum(1 for b in books if b.needs_review),
            "overlapping_pairs_evaluated": len(pairs),
            "pair_decisions": dict(sorted(counts.items())),
            "merges_blocked_by_cluster_guards": len(blocked),
            "blocked_merges_superseded_by_better_match": sum(1 for p in blocked if p["superseded"]),
            "ambiguous_pairs_superseded_by_better_match": sum(1 for p in gray if p["superseded"]),
            "ambiguous_pairs": len(ambiguous),
            "observations_with_competing_matches": len(competing),
        },
        "books": book_records,
    }
    # Keep the pair log readable: skip pairs that merely touch.
    notable = [p for p in pairs if p["decision"] != "separate" or p["shared_iou"] >= 0.1
               or p["intersection_over_smaller"] >= 0.3]
    return DedupeResult(books=books, report=report, pair_decisions=notable, ambiguous_pairs=ambiguous, all_pairs=pairs)

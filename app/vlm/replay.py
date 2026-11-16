"""Offline replay of a saved /analyze/vlm run through the current dedupe.

    python -m app.vlm.replay --input data/debug/vlm/<request_id> \
                             [--output data/debug/vlm_replay/<run_id>]

Reads the saved `response.json` (per-tile parsed, validated detections in
original-image pixels), `diagnostics.json` and `original.jpg`, then reruns
both the legacy dedupe (geometry.dedupe_across_tiles) and the current one
(dedupe.dedupe_physical_books) and writes the results to a new directory.

Strictly offline: no provider, no metadata source and no database are
imported. The input directory is only read, never written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .config import BASE_DIR, load_dedupe_config
from .debug import DebugWriter, draw_pair_crop, write_dedupe_artifacts
from .dedupe import DedupeConfig, dedupe_physical_books, observation_id
from .geometry import Tile, dedupe_across_tiles
from .schemas import VLMDetection

REPLAY_ROOT = BASE_DIR / "data" / "debug" / "vlm_replay"


class ReplayError(Exception):
    pass


def load_saved_run(input_dir: Path) -> tuple[list[Tile], list[VLMDetection], dict]:
    response_path = input_dir / "response.json"
    if not response_path.is_file():
        raise ReplayError(f"{response_path} not found (was the run made with debug enabled?)")
    regions = json.loads(response_path.read_text())
    diagnostics_path = input_dir / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.is_file() else {}

    tiles, detections = [], []
    for region in regions:
        if region.get("tile_id") is None:
            raise ReplayError("full-mode run: there are no overlapping tiles to deduplicate")
        tiles.append(Tile(**region["region"]))
        for item in region.get("parsed_valid_detections") or []:
            detections.append(VLMDetection.model_validate(item))
    return tiles, detections, diagnostics


# --- legacy vs current comparison ------------------------------------------------


def _legacy_groups(detections: list[VLMDetection]) -> tuple[list[list[str]], float]:
    """Run the legacy dedupe on copies and express its output as groups of
    observation ids (legacy keeps no ids, so match on tile + model box)."""
    copies = [d.model_copy(deep=True) for d in detections]
    lookup: dict[tuple, list[str]] = {}
    for d in detections:
        lookup.setdefault((d.tile_id, tuple(d.model_box_2d), d.bbox.x, d.bbox.y), []).append(observation_id(d.tile_id, d.index))

    def obs_id(tile_id, box_2d, bbox) -> str:
        return lookup[(tile_id, tuple(box_2d), bbox.x, bbox.y)][0]

    t0 = time.perf_counter()
    kept, _ = dedupe_across_tiles(copies)
    elapsed = (time.perf_counter() - t0) * 1000
    groups = [sorted([obs_id(k.tile_id, k.model_box_2d, k.bbox)] +
                     [obs_id(m.tile_id, m.model_box_2d, m.bbox) for m in k.merged_duplicates]) for k in kept]
    return groups, elapsed


def _co_clustered(groups: list[list[str]]) -> set[tuple[str, str]]:
    return {(a, b) for g in groups for i, a in enumerate(g) for b in g[i + 1:]}


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def compare(legacy_groups: list[list[str]], result, all_pairs: dict) -> dict:
    new_groups = [list(b["observation_ids"]) for b in result.report["books"]]
    legacy_pairs = {_pair_key(*p) for p in _co_clustered(legacy_groups)}
    new_pairs = {_pair_key(*p) for p in _co_clustered(new_groups)}

    def describe(key):
        p = all_pairs.get(key)
        if not p:
            return {"a": key[0], "b": key[1], "note": "boxes do not overlap"}
        return {k: p[k] for k in ("a", "b", "decision", "reason", "shared_iou", "iou", "intersection_over_smaller",
                                  "text_relation", "a_text", "b_text", "a_bbox", "b_bbox")}

    only_new = sorted(new_pairs - legacy_pairs)
    only_legacy = sorted(legacy_pairs - new_pairs)
    weak_evidence = []
    for book in result.report["books"]:
        for m in book["merge_decisions"]:
            if m["geometry_strength"] != "strong" or m["text_relation"] in ("unconfirmed", "no_text"):
                weak_evidence.append({"book_index": book["book_index"], **m})
    return {
        "legacy_books": len(legacy_groups),
        "new_books": len(new_groups),
        "observation_pairs_merged_by_legacy": len(legacy_pairs),
        "observation_pairs_merged_by_new": len(new_pairs),
        "pairs_merged_only_by_new": len(only_new),
        "pairs_merged_only_by_legacy": len(only_legacy),
        "merges_with_weaker_evidence": len(weak_evidence),
        "merges_with_weaker_evidence_note": "merged pairs whose geometry was only 'supported' (text had to agree) "
                                            "or whose text was unconfirmed/absent: the first place to look for unsafe merges",
        "pairs_merged_only_by_new_detail": [describe(k) for k in only_new],
        "pairs_merged_only_by_legacy_detail": [describe(k) for k in only_legacy],
        "merges_with_weaker_evidence_detail": weak_evidence,
    }


# --- review set ------------------------------------------------------------------------


def _spread(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def select_review_set(result, all_pairs: dict, comparison: dict, size: int) -> list[dict]:
    """Deterministic, stratified sample of pair decisions for a human to judge."""
    applied = {_pair_key(m["a"], m["b"]) for b in result.report["books"] for m in b["merge_decisions"]}
    merges = [p for k, p in sorted(all_pairs.items()) if k in applied]
    strata = [
        ("merge: author/title complementary", [p for p in merges if p["text_relation"] == "complementary"], 0.2),
        ("merge: same title", [p for p in merges if p["text_relation"] == "same_title"], 0.12),
        ("merge: geometry only (no/unconfirmed text)",
         [p for p in merges if p["text_relation"] in ("no_text", "unconfirmed", "consistent")], 0.12),
        ("ambiguous: not merged, flagged", sorted(result.ambiguous_pairs, key=lambda p: (p["a"], p["b"])), 0.26),
        ("separate: overlapping boxes kept apart",
         [p for k, p in sorted(all_pairs.items()) if p["decision"] == "separate"
          and (p["shared_iou"] >= 0.3 or p["intersection_over_smaller"] >= 0.6)], 0.18),
        ("legacy merged, new did not",
         [all_pairs[_pair_key(d["a"], d["b"])] for d in comparison["pairs_merged_only_by_legacy_detail"]
          if _pair_key(d["a"], d["b"]) in all_pairs], 0.12),
    ]
    chosen, seen = [], set()
    for label, items, share in strata:
        items = [p for p in items if _pair_key(p["a"], p["b"]) not in seen]
        for p in _spread(items, max(1, round(size * share))):
            seen.add(_pair_key(p["a"], p["b"]))
            chosen.append({"stratum": label, **p})
    return chosen[:size]


# --- main ------------------------------------------------------------------------------


def replay(input_dir: Path, output_dir: Path, config: DedupeConfig, review_size: int = 40, images: bool = True) -> dict:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ReplayError("output must be outside the input directory (original debug files are never modified)")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReplayError(f"{output_dir} already exists and is not empty")

    tiles, detections, diagnostics = load_saved_run(input_dir)

    t0 = time.perf_counter()
    result = dedupe_physical_books(detections, tiles, config)
    new_ms = (time.perf_counter() - t0) * 1000
    legacy_groups, legacy_ms = _legacy_groups(detections)

    all_pairs = {_pair_key(p["a"], p["b"]): p for p in result.all_pairs}
    comparison = compare(legacy_groups, result, all_pairs)

    image = None
    original = input_dir / "original.jpg"
    if images and original.is_file():
        with Image.open(original) as im:
            image = im.convert("RGB")

    out = DebugWriter(output_dir)
    out.save_json("candidates_path", "candidates.json", [b.model_dump(mode="json") for b in result.books])
    write_dedupe_artifacts(out, image, result.books, result.report, result.pair_decisions, result.ambiguous_pairs, tiles)
    out.save_json("comparison_path", "comparison.json", comparison)

    review = select_review_set(result, all_pairs, comparison, review_size)
    for i, p in enumerate(review):
        p["review_id"] = i
        p["human_verdict"] = None  # fill in: "same_book" | "different_books" | "unsure"
        if image is not None:
            header = [f"#{i} [{p['stratum']}] {p['a']} (blue) vs {p['b']} (orange) -> {p['decision']}",
                      p["reason"], f"a: {p['a_text']}", f"b: {p['b_text']}"]
            rel = f"review/{i:02d}_{p['decision']}_{p['a'].replace(':', '-')}_{p['b'].replace(':', '-')}.jpg"
            out.save_image("review_images[]", rel, draw_pair_crop(image, p, header))
            p["image"] = rel
    out.save_json("review_set_path", "review/review_set.json", review)

    legacy_count = diagnostics.get("summary", {}).get("total_detections")
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "replayed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": result.report["algorithm"],
        "config": config.as_dict(),
        "tiles": len(tiles),
        "observations": len(detections),
        "original_run_final_candidates": legacy_count,
        "legacy_replay_final_candidates": len(legacy_groups),
        "legacy_replay_matches_original_run": legacy_count == len(legacy_groups) if legacy_count is not None else None,
        "new_final_candidates": len(result.books),
        **{k: v for k, v in result.report["summary"].items() if k not in ("observations", "books")},
        "pairs_merged_only_by_new": comparison["pairs_merged_only_by_new"],
        "pairs_merged_only_by_legacy": comparison["pairs_merged_only_by_legacy"],
        "merges_with_weaker_evidence": comparison["merges_with_weaker_evidence"],
        "dedupe_ms": {"legacy": round(legacy_ms, 1), "new": round(new_ms, 1)},
        "review_set_size": len(review),
        "network_calls": 0,
        "files": out.files,
    }
    out.save_json("replay_summary_path", "replay_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.vlm.replay", description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, type=Path, help="saved run, e.g. data/debug/vlm/<request_id>")
    parser.add_argument("--output", type=Path, help=f"default: {REPLAY_ROOT.relative_to(BASE_DIR)}/<request_id>-<UTC time>")
    parser.add_argument("--review-size", type=int, default=40, help="pair decisions to sample for manual review")
    parser.add_argument("--no-images", action="store_true", help="skip annotated images and review crops")
    args = parser.parse_args(argv)

    output = args.output or REPLAY_ROOT / f"{args.input.resolve().name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    try:
        summary = replay(args.input, output, load_dedupe_config(), args.review_size, images=not args.no_images)
    except ReplayError as exc:
        print(f"replay failed: {exc}", file=sys.stderr)
        return 2
    keys = ("original_run_final_candidates", "legacy_replay_final_candidates", "new_final_candidates",
            "books_from_multiple_observations", "books_needing_review", "ambiguous_pairs", "pairs_merged_only_by_new",
            "pairs_merged_only_by_legacy", "merges_with_weaker_evidence", "dedupe_ms", "output_dir")
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

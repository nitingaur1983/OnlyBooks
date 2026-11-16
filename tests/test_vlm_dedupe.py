"""Physical-book dedupe (app/vlm/dedupe.py) and offline replay (app/vlm/replay.py).
No network: replay tests block sockets outright."""

import hashlib
import json
import socket

import pytest
from PIL import Image

from app.vlm.dedupe import DedupeConfig, dedupe_physical_books
from app.vlm.geometry import Tile
from app.vlm.replay import ReplayError, replay
from app.vlm.schemas import BBox, IdentificationStatus, Orientation, Readability, VLMDetection

# Two tiles side by side sharing the strip x 800..1000.
LEFT = Tile(0, 0, 0, 0, 0, 1000, 1000)
RIGHT = Tile(1, 0, 1, 800, 0, 1000, 1000)
TILES = [LEFT, RIGHT]


def obs(tile, index, x, y, w, h, title=None, author=None, readability=None):
    if readability is None:
        readability = Readability.CLEAR if title else Readability.UNREADABLE
    raw = " ".join(t for t in (title, author) if t) or None
    return VLMDetection(
        index=index, bbox=BBox(x=x, y=y, width=w, height=h), visible_title=title, visible_author=author,
        raw_visible_text=raw, title_readability=readability,
        author_readability=Readability.CLEAR if author else Readability.NOT_VISIBLE,
        identification_status=IdentificationStatus.VISIBLE_TEXT if raw else IdentificationStatus.NO_READABLE_TEXT,
        orientation=Orientation.SPINE_OUT, inferred_title=None, uncertainty_notes=None, shelf_index=None,
        tile_id=tile.tile_id, model_box_2d=[0, 0, 1, 1],
    )


def groups(result):
    return sorted(sorted(b.observation_ids) for b in result.books)


def ambiguous_ids(result):
    return {(p["a"], p["b"]) for p in result.ambiguous_pairs}


# 1
def test_same_book_same_title_overlapping_boxes_merge():
    res = dedupe_physical_books([obs(LEFT, 0, 850, 100, 80, 800, "These Truths", "Jill Lepore"),
                                 obs(RIGHT, 0, 853, 102, 78, 797, "These Truths", "Jill Lepore")], TILES)
    assert groups(res) == [["t0:0", "t1:0"]]
    book = res.books[0]
    assert book.visible_title == "These Truths" and not book.needs_review
    assert book.merged_duplicates[0].tile_id in (0, 1)


# 2
def test_author_only_read_of_cut_spine_merges_with_titled_read():
    # Book at x 960..1040 is cut by the left tile's right edge (1000): that
    # tile only read the author.
    cut = obs(LEFT, 0, 960, 100, 40, 800, None, "Jemimah Wei")
    full = obs(RIGHT, 0, 961, 101, 79, 799, "The Original Daughter", "Jemimah Wei")
    res = dedupe_physical_books([cut, full], TILES)
    assert groups(res) == [["t0:0", "t1:0"]]
    book = res.books[0]
    assert book.visible_title == "The Original Daughter"  # nothing invented, best read wins
    assert book.bbox.x2 == 1040  # right side from the read that was not cut there, never the tile edge
    record = res.report["books"][0]
    assert record["observed_authors"] == ["Jemimah Wei"]
    assert record["merge_decisions"][0]["text_relation"] == "complementary"


def test_author_misread_as_title_still_complementary():
    cut = obs(LEFT, 0, 960, 100, 40, 800, "JEMIMAH WEI")
    full = obs(RIGHT, 0, 961, 101, 79, 799, "THE ORIGINAL DAUGHTER", "JEMIMAH WEI")
    res = dedupe_physical_books([cut, full], TILES)
    assert groups(res) == [["t0:0", "t1:0"]]


def test_author_match_without_spatial_agreement_does_not_merge():
    # Same author one book-width to the right: a neighbouring book by the same writer.
    res = dedupe_physical_books([obs(LEFT, 0, 850, 100, 70, 800, None, "Elizabeth Strout"),
                                 obs(RIGHT, 0, 905, 100, 70, 800, "Olive Kitteridge", "Elizabeth Strout")], TILES)
    assert len(res.books) == 2


# 3
@pytest.mark.parametrize("title_a,title_b", [(None, None), ("Hera", "Atalanta")])
def test_adjacent_books_touching_or_slightly_overlapping_stay_separate(title_a, title_b):
    res = dedupe_physical_books([obs(LEFT, 0, 850, 100, 80, 800, title_a),
                                 obs(RIGHT, 0, 920, 100, 80, 800, title_b)], TILES)
    assert len(res.books) == 2
    assert not res.ambiguous_pairs


# 4
def test_two_physical_copies_of_same_title_stay_separate():
    detections = [
        obs(LEFT, 0, 820, 100, 80, 800, "Vigil", "George Saunders"),
        obs(LEFT, 1, 900, 100, 80, 800, "Vigil", "George Saunders"),
        obs(RIGHT, 0, 822, 101, 79, 799, "Vigil", "George Saunders"),
        obs(RIGHT, 1, 901, 101, 79, 799, "Vigil", "George Saunders"),
    ]
    res = dedupe_physical_books(detections, TILES)
    assert groups(res) == [["t0:0", "t1:0"], ["t0:1", "t1:1"]]


def test_copy_seen_by_one_tile_only_is_not_absorbed_by_its_twin():
    res = dedupe_physical_books([obs(LEFT, 0, 820, 100, 80, 800, "Vigil"), obs(LEFT, 1, 900, 100, 80, 800, "Vigil"),
                                 obs(RIGHT, 0, 901, 101, 79, 799, "Vigil")], TILES)
    assert groups(res) == [["t0:0"], ["t0:1", "t1:0"]]


# 5
def test_oversized_box_does_not_swallow_the_books_inside_it():
    detections = [
        obs(LEFT, 0, 800, 100, 200, 800),  # one box over three spines, no text
        obs(RIGHT, 0, 800, 100, 65, 800, "Hera"),
        obs(RIGHT, 1, 866, 100, 66, 800, "Atalanta"),
        obs(RIGHT, 2, 933, 100, 67, 800, "Circe"),
    ]
    res = dedupe_physical_books(detections, TILES)
    assert len(res.books) == 4
    assert all(len(b.observation_ids) == 1 for b in res.books)
    oversized = next(b for b in res.books if b.observation_ids == ["t0:0"])
    assert oversized.needs_review  # flagged: one box contains several others


# 6
def test_contradictory_titles_in_nearly_identical_boxes_are_flagged_not_merged():
    res = dedupe_physical_books([obs(LEFT, 0, 850, 100, 80, 800, "The Anthropocene"),
                                 obs(RIGHT, 0, 851, 100, 80, 800, "The Anthropologists", "Aysegul Savas")], TILES)
    assert len(res.books) == 2
    assert ambiguous_ids(res) == {("t0:0", "t1:0")}
    assert all(b.needs_review for b in res.books)
    assert "titles differ" in res.ambiguous_pairs[0]["reason"]


# 7
def test_chain_overlap_does_not_collapse_distinct_books():
    # Three tiles all see x 800..1000. A~B and B~C agree geometrically, A and C
    # are different books (titles conflict), B has no text.
    t2 = Tile(2, 0, 2, 600, 0, 1000, 1000)
    a = obs(LEFT, 0, 850, 100, 80, 800, "These Truths")
    b = obs(RIGHT, 0, 862, 100, 80, 800)
    c = obs(t2, 0, 874, 100, 80, 800, "Crossing to Safety")
    res = dedupe_physical_books([a, b, c], [LEFT, RIGHT, t2])
    assert not any({"t0:0", "t2:0"} <= set(b.observation_ids) for b in res.books)
    assert len(res.books) == 2
    assert res.report["summary"]["merges_blocked_by_cluster_guards"] >= 1
    assert any(b.needs_review for b in res.books)  # the losing link is surfaced


def test_never_two_observations_from_one_tile_in_a_book():
    # Two reads from the left tile both overlap one right-tile read.
    res = dedupe_physical_books([obs(LEFT, 0, 850, 100, 80, 800, "Hera"), obs(LEFT, 1, 852, 100, 80, 800, "Hera"),
                                 obs(RIGHT, 0, 851, 100, 80, 800, "Hera")], TILES)
    for b in res.books:
        assert len({x.split(":")[0] for x in b.observation_ids}) == len(b.observation_ids)


# 8
def test_book_cut_by_horizontal_tile_boundary_merges_and_keeps_full_extent():
    top = Tile(0, 0, 0, 0, 0, 1000, 1000)
    bottom = Tile(1, 1, 0, 0, 800, 1000, 1000)
    upper = obs(top, 0, 100, 700, 80, 300, "These Tru", readability=Readability.PARTIAL)  # cut at y=1000
    lower = obs(bottom, 0, 101, 800, 79, 600, None, "Jill Lepore")  # cut at y=800
    res = dedupe_physical_books([upper, lower], [top, bottom])
    assert groups(res) == [["t0:0", "t1:0"]]
    assert res.books[0].bbox == BBox(x=100, y=700, width=80, height=700)  # top from upper, bottom from lower
    assert res.books[0].visible_title == "These Tru"
    record = res.report["books"][0]
    assert {o["observation_id"]: o["cut_edges"] for o in record["observations"]} == {"t0:0": ["bottom"], "t1:0": ["top"]}


def test_slivers_at_tile_edge_need_overlap_in_shared_zone():
    # A sliver of the next shelf's book peeking into the shared zone must not
    # merge with an unrelated book that also touches it.
    top = Tile(0, 0, 0, 0, 0, 1000, 1000)
    bottom = Tile(1, 1, 0, 0, 800, 1000, 1000)
    res = dedupe_physical_books([obs(top, 0, 100, 200, 80, 605), obs(bottom, 0, 100, 803, 80, 600)], [top, bottom])
    assert len(res.books) == 2


def test_dedupe_is_deterministic_and_input_order_independent():
    detections = [obs(LEFT, i, 820 + 60 * i, 100, 58, 800, f"Book {i}") for i in range(3)]
    detections += [obs(RIGHT, i, 821 + 60 * i, 101, 57, 799, f"Book {i}") for i in range(3)]
    first = dedupe_physical_books(detections, TILES)
    second = dedupe_physical_books(list(reversed(detections)), TILES)
    assert first.report == second.report
    assert [b.model_dump() for b in first.books] == [b.model_dump() for b in second.books]


def test_thresholds_are_configurable():
    pair = [obs(LEFT, 0, 850, 100, 80, 800), obs(RIGHT, 0, 862, 100, 80, 800)]  # shared IoU ~0.74, no text
    assert len(dedupe_physical_books(pair, TILES).books) == 1
    assert len(dedupe_physical_books(pair, TILES, DedupeConfig(strong_iou=0.8)).books) == 2


# --- replay (9) ------------------------------------------------------------------------------


@pytest.fixture
def saved_run(tmp_path):
    """A minimal debug directory as /analyze/vlm writes it."""
    run = tmp_path / "vlm" / "abc123"
    run.mkdir(parents=True)
    detections = {
        0: [obs(LEFT, 0, 850, 100, 80, 800, "These Truths"), obs(LEFT, 1, 960, 100, 40, 800, None, "Jemimah Wei"),
            obs(LEFT, 2, 100, 100, 80, 800, "Hera")],
        1: [obs(RIGHT, 0, 852, 101, 78, 798, "These Truths"),
            obs(RIGHT, 1, 961, 101, 79, 799, "The Original Daughter", "Jemimah Wei"),
            obs(RIGHT, 2, 1500, 100, 80, 800, "Atalanta")],
    }
    response = [{"tile_id": t.tile_id, "region": t.as_dict(), "finish_reason": "STOP", "raw_text": "{}",
                 "parsed_valid_detections": [d.model_dump(mode="json") for d in detections[t.tile_id]]} for t in TILES]
    (run / "response.json").write_text(json.dumps(response))
    (run / "diagnostics.json").write_text(json.dumps({"summary": {"total_detections": 5}}))
    Image.new("RGB", (1800, 1000), (200, 180, 150)).save(run / "original.jpg")
    return run


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("replay attempted a network connection")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _digest(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def test_replay_twice_gives_identical_results_offline_and_leaves_input_untouched(saved_run, tmp_path, no_network):
    before = _digest(saved_run)
    s1 = replay(saved_run, tmp_path / "out1", DedupeConfig(), review_size=10)
    s2 = replay(saved_run, tmp_path / "out2", DedupeConfig(), review_size=10)
    assert _digest(saved_run) == before

    for name in ("candidates.json", "dedupe_report.json", "ambiguous_pairs.json", "pair_decisions.json",
                 "comparison.json", "review/review_set.json"):
        assert (tmp_path / "out1" / name).read_bytes() == (tmp_path / "out2" / name).read_bytes(), name
    for name in ("deduplicated_annotated.jpg", "dedupe_review.jpg"):
        assert (tmp_path / "out1" / name).is_file()

    assert s1["legacy_replay_final_candidates"] == 5  # legacy keeps the author-only read separate
    assert s1["new_final_candidates"] == s2["new_final_candidates"] == 4
    assert s1["network_calls"] == 0
    candidates = json.loads((tmp_path / "out1" / "candidates.json").read_text())
    assert sorted(c["visible_title"] for c in candidates) == ["Atalanta", "Hera", "The Original Daughter", "These Truths"]


def test_replay_refuses_to_write_into_the_input(saved_run):
    with pytest.raises(ReplayError):
        replay(saved_run, saved_run / "replay", DedupeConfig())


def test_replay_refuses_full_mode_runs(tmp_path):
    run = tmp_path / "full"
    run.mkdir()
    (run / "response.json").write_text(json.dumps([{"tile_id": None, "region": {}, "parsed_valid_detections": []}]))
    with pytest.raises(ReplayError):
        replay(run, tmp_path / "out", DedupeConfig())

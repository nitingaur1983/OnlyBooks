from app.config import MatchConfig
from app.matcher import BookMatcher, MatchStatus
from app.metadata_source import MetadataCandidate


class StubMetadataSource:
    """A MetadataSource test double with no network calls."""

    def __init__(self, results=None, isbn_result=None):
        self.results = results or []
        self.isbn_result = isbn_result
        self.search_calls: list[str] = []

    def search(self, query, limit=5):
        self.search_calls.append(query)
        return self.results

    def lookup_isbn(self, isbn):
        return self.isbn_result


class FailingMetadataSource:
    def search(self, query, limit=5):
        raise RuntimeError("network down")

    def lookup_isbn(self, isbn):
        raise RuntimeError("network down")


def make_matcher(source, **overrides):
    return BookMatcher(source, MatchConfig(**overrides))


def test_garbage_ocr_never_queries_metadata_and_is_no_match():
    source = StubMetadataSource(
        results=[MetadataCandidate("Some Book", "Some Author", None, None, "open_library", None)]
    )
    matcher = make_matcher(source)
    raw = "— _—_—_—_—_—_——_———_—_—_—_—_— lL —"

    result = matcher.match(raw, ocr_confidence=19.0)

    assert result.status == MatchStatus.NO_MATCH
    assert result.matched is None
    assert source.search_calls == []  # never even attempted a lookup


def test_strong_title_author_match_is_high_confidence():
    source = StubMetadataSource(
        results=[
            MetadataCandidate(
                "Crossing to Safety", "Wallace Stegner", "9780679732619",
                "Random House", "open_library", "/works/OL123W",
            )
        ]
    )
    matcher = make_matcher(source)

    result = matcher.match("Crossing to Safety Wallace Stegner", ocr_confidence=85.0)

    assert result.status == MatchStatus.HIGH_CONFIDENCE
    assert result.matched.title == "Crossing to Safety"


def test_generic_word_only_overlap_does_not_reach_possible_match():
    source = StubMetadataSource(
        results=[MetadataCandidate("The Story of Love", "Some Author", None, None, "open_library", None)]
    )
    matcher = make_matcher(source)

    result = matcher.match("the story new edition", ocr_confidence=60.0)

    assert result.status in (MatchStatus.NO_MATCH, MatchStatus.LOW_CONFIDENCE)


def test_no_search_results_is_no_match():
    matcher = make_matcher(StubMetadataSource(results=[]))

    result = matcher.match("Reasonably Plausible Title Words", ocr_confidence=70.0)

    assert result.status == MatchStatus.NO_MATCH


def test_isbn_hit_short_circuits_to_high_confidence():
    isbn_hit = MetadataCandidate(
        "Some ISBN Book", "Author X", "9780143127550", None, "open_library", "/works/OL999W"
    )
    matcher = make_matcher(StubMetadataSource(isbn_result=isbn_hit))

    result = matcher.match("garbage text 9780143127550 more garbage", ocr_confidence=40.0)

    assert result.status == MatchStatus.HIGH_CONFIDENCE
    assert result.matched.isbn == "9780143127550"


def test_metadata_source_failure_degrades_to_no_match_not_a_crash():
    matcher = make_matcher(FailingMetadataSource())

    result = matcher.match("Reasonably Plausible Title Words", ocr_confidence=70.0)

    assert result.status == MatchStatus.NO_MATCH

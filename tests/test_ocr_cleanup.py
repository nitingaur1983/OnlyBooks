from app.ocr_cleanup import clean


def test_symbol_noise_yields_low_alpha_ratio_and_no_real_words():
    # This is the exact kind of garbage that was previously inserted into
    # `books` (see conversation/README): spine dividers and shadow edges
    # misread as dashes, not text.
    raw = "— _—_—_—_—_—_——_———_—_—_—_—_—_—_—_—_—_—__ lL —"
    result = clean(raw)
    meaningful = [t for t in result.alpha_tokens if len(t) >= 3]
    assert meaningful == []
    assert result.alpha_ratio < 0.35


def test_real_looking_title_survives_cleanup():
    raw = "Crossing to  Safety   -   Wallace Stegner"
    result = clean(raw)
    assert "Crossing" in result.alpha_tokens
    assert "Stegner" in result.alpha_tokens
    assert result.alpha_ratio > 0.7


def test_empty_input_is_handled():
    result = clean("")
    assert result.cleaned == ""
    assert result.alpha_tokens == []
    assert result.alpha_ratio == 0.0

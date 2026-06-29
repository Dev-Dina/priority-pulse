"""Unit tests for the ML feature extractor (pure, deterministic function)."""

from app.ml.feature_engineering import engineer_features


def test_keyword_flags_detected():
    f = engineer_features("I need a refund, my flight was cancelled and I am stranded!")
    assert f["has_refund"] == 1
    assert f["has_cancel"] == 1
    assert f["has_stranded"] == 1
    assert f["exclamation_count"] == 1


def test_absent_keywords_are_zero():
    f = engineer_features("hello there, lovely weather today")
    assert f["has_refund"] == 0
    assert f["has_cancel"] == 0
    assert f["has_delay"] == 0
    assert f["profanity_count"] == 0


def test_caps_ratio_is_uppercase_fraction():
    assert engineer_features("HELP")["caps_ratio"] == 1.0
    assert engineer_features("help")["caps_ratio"] == 0.0


def test_punctuation_counts():
    f = engineer_features("really?! are you sure?!")
    assert f["question_count"] == 2
    assert f["exclamation_count"] == 2


def test_time_mention_detection():
    assert engineer_features("we waited 3 hours")["has_time_mention"] == 1
    assert engineer_features("no time reference here")["has_time_mention"] == 0


def test_returns_all_expected_features():
    expected = {
        "text_length", "word_count", "exclamation_count", "question_count",
        "caps_ratio", "has_refund", "has_cancel", "has_delay", "has_help",
        "has_broken", "has_stranded", "has_medical", "profanity_count",
        "has_time_mention",
    }
    assert set(engineer_features("any text").keys()) == expected

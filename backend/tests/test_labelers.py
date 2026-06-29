"""Unit tests for the weak-supervision labeler.

Includes a regression guard for the permanently-dead `has_missing` branch in
`label_ticket`: it scores `has_missing` (+2) but `extract_features` never emits
that key, so the branch can never fire. These tests pin that behaviour.
"""

import pytest

from app.data_processing.labelers import PriorityLabeler


@pytest.fixture
def labeler():
    return PriorityLabeler()


def test_critical_keyword_forces_urgent(labeler):
    out = labeler.label_ticket("there is a medical emergency on board")
    assert out["priority"] == "URGENT"
    assert out["score"] == 10  # hard override


def test_delay_crosses_threshold(labeler):
    out = labeler.label_ticket("my flight is delayed")
    assert out["priority"] == "URGENT"
    assert out["score"] >= 3


def test_benign_text_is_normal(labeler):
    out = labeler.label_ticket("thanks for the lovely flight")
    assert out["priority"] == "NORMAL"
    assert out["score"] < 3


def test_empty_text_is_normal(labeler):
    out = labeler.label_ticket("   ")
    assert out["priority"] == "NORMAL"
    assert out["score"] == 0


def test_extract_features_never_emits_has_missing(labeler):
    # Root cause of the dead branch: the key the scorer reads is never produced.
    f = labeler.extract_features("my bag is missing")
    assert "has_missing" not in f
    assert f["has_urgency_keywords"] is True  # "missing" matches urgency, not has_missing


def test_has_missing_branch_is_dead(labeler):
    # "missing" earns urgency (+2) only; the dead has_missing (+2) never applies,
    # so the score stays at 2 (NORMAL). If a change makes the branch live, the
    # score would jump to 4 (URGENT) and this guard fails.
    out = labeler.label_ticket("missing")
    assert out["score"] == 2
    assert out["priority"] == "NORMAL"

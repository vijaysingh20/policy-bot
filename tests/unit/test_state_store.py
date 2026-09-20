from uuid import uuid4

import pytest

from backend.evaluation import (
    create_evaluation_state,
    finish_evaluation_state,
    get_evaluation_state,
    initialize_evaluation_store,
)
from backend.schemas import EvaluationScores, EvaluationState

SCORES = EvaluationScores(faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "state.sqlite3"
    initialize_evaluation_store(path)
    return path


@pytest.fixture
def pending_id(db_path):
    evaluation_id = uuid4()
    create_evaluation_state(EvaluationState(evaluation_id=evaluation_id), db_path)
    return evaluation_id


def test_new_evaluation_is_pending(db_path, pending_id):
    assert get_evaluation_state(pending_id, db_path).status == "pending"


def test_unknown_evaluation_returns_none(db_path):
    assert get_evaluation_state(uuid4(), db_path) is None


def test_finishing_stores_the_scores(db_path, pending_id):
    finish_evaluation_state(
        EvaluationState(evaluation_id=pending_id, status="completed", scores=SCORES), db_path
    )
    state = get_evaluation_state(pending_id, db_path)
    assert state.status == "completed"
    assert state.scores == SCORES


def test_a_finished_evaluation_cannot_be_finished_again(db_path, pending_id):
    done = EvaluationState(evaluation_id=pending_id, status="completed", scores=SCORES)
    finish_evaluation_state(done, db_path)
    failed = EvaluationState(evaluation_id=pending_id, status="failed", reason="late retry")
    with pytest.raises(ValueError, match="no longer pending"):
        finish_evaluation_state(failed, db_path)
    assert get_evaluation_state(pending_id, db_path).status == "completed"  # unchanged
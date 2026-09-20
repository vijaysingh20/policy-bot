from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.schemas import EvaluationScores, EvaluationState, QueryPlan

SCORES = EvaluationScores(faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7)


@pytest.mark.parametrize("route, queries", [
    ("single", ["annual leave days"]),
    ("decompose", ["annual leave days", "sick leave days"]),
    ("out_of_scope", []),
])
def test_valid_query_plans(route, queries):
    assert QueryPlan(route=route, queries=queries).queries == queries


@pytest.mark.parametrize("route, queries", [
    ("single", []),                              # single needs exactly one
    ("single", ["leave", "benefits"]),           # ...not two
    ("decompose", ["only one"]),                 # decompose needs two or more
    ("out_of_scope", ["leave"]),                 # out_of_scope must not retrieve
    ("single", ["   "]),                         # blank query
    ("multi", ["leave"]),                        # unknown route
])
def test_invalid_query_plans_are_rejected(route, queries):
    with pytest.raises(ValidationError):
        QueryPlan(route=route, queries=queries)


@pytest.mark.parametrize("fields", [
    {"status": "pending", "scores": SCORES},       # pending cannot have scores
    {"status": "pending", "reason": "why"},        # ...or a reason
    {"status": "completed"},                       # completed needs scores
    {"status": "completed", "scores": SCORES, "reason": "x"},
    {"status": "failed"},                          # failed needs a reason
    {"status": "skipped", "reason": "   "},        # ...a non-blank one
    {"status": "failed", "reason": "boom", "scores": SCORES},
])
def test_inconsistent_evaluation_states_are_rejected(fields):
    with pytest.raises(ValidationError):
        EvaluationState(evaluation_id=uuid4(), **fields)


@pytest.mark.parametrize("field, value", [
    ("faithfulness", 1.5),
    ("faithfulness", -0.1),
    ("context_precision", float("nan")),
])
def test_scores_outside_their_range_are_rejected(field, value):
    with pytest.raises(ValidationError):
        EvaluationScores(**{**SCORES.model_dump(), field: value})
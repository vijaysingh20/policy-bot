from uuid import uuid4

from backend.config import DATABASE_DIR
from backend.schemas import EvaluationState, EvaluationScores
from backend.evaluation import (
    initialize_evaluation_store,
    create_evaluation_state,
    get_evaluation_state,
    finish_evaluation_state
)


def main() -> None:
    db_path = DATABASE_DIR / "evaluation_store_check.sqlite3"
    initialize_evaluation_store(db_path)

    state = EvaluationState(evaluation_id=uuid4())

    create_evaluation_state(state=state, db_path=db_path)

    # Synthetic scores used only to check storage behavior.
    completed = EvaluationState(
        evaluation_id=state.evaluation_id,
        status="completed",
        scores=EvaluationScores(
            faithfulness=0.9,
            answer_relevancy=0.8,
            context_precision=0.7,
        ),
    )

    finish_evaluation_state(
        state=completed,
        db_path=db_path,
    )

    loaded = get_evaluation_state(
        evaluation_id=state.evaluation_id,
        db_path=db_path,
    )

    assert loaded == completed
    print("Completed state saved correctly.")

    late_failure = EvaluationState(
        evaluation_id=state.evaluation_id,
        status="failed",
        reason="Simulated late worker failure.",
    )

    try:
        finish_evaluation_state(
            state=late_failure,
            db_path=db_path,
        )
    except ValueError as exc:
        print("Late update rejected:", exc)
    else:
        raise AssertionError("A finished evaluation was overwritten")

    after = get_evaluation_state(
        evaluation_id=state.evaluation_id,
        db_path=db_path,
    )

    assert after == completed
    print("Original completed result preserved.")


if __name__ == "__main__":
    main()
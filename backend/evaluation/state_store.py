import sqlite3
from uuid import UUID
from contextlib import closing
from pathlib import Path
from backend.schemas import EvaluationState

def initialize_evaluation_store(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_states (
                    evaluation_id TEXT PRIMARY KEY NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'pending', 'completed', 'skipped', 'failed'
                        )
                    ),
                    state_json TEXT NOT NULL
                )
            """)

def create_evaluation_state(
    state: EvaluationState,
    db_path: Path
) -> None:
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute("""
                INSERT INTO evaluation_states(
                    evaluation_id,
                    status,
                    state_json
                )
                VALUES (?, ?, ?)
            """, (
                str(state.evaluation_id),
                state.status,
                state.model_dump_json()
            ))

def get_evaluation_state(
    evaluation_id: UUID,
    db_path: Path
) -> EvaluationState | None:
    with closing(sqlite3.connect(db_path)) as connection:
        cursor = connection.execute(
            """
            SELECT state_json
            FROM evaluation_states
            WHERE evaluation_id = ?
            """,
            (str(evaluation_id),)
        )

        row = cursor.fetchone()

    if row is None:
        return None

    return EvaluationState.model_validate_json(row[0])

def finish_evaluation_state(
    state: EvaluationState,
    db_path: Path,
) -> None:
    if state.status == "pending":
        raise ValueError("The new state must be completed, skipped, or failed")

    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_states
                SET status = ?, state_json = ?
                WHERE evaluation_id = ?
                  AND status = 'pending'
                """,
                (
                    state.status,
                    state.model_dump_json(),
                    str(state.evaluation_id),
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError("Evaluation was not found or is no longer pending")
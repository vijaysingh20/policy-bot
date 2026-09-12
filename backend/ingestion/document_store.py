import sqlite3
from uuid import UUID
from contextlib import closing
from pathlib import Path

from backend.schemas import DocumentUploadResponse

def initialize_document_store(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY NOT NULL,
                    document_json TEXT NOT NULL
                )
                """
            )

def create_document_record(
    document: DocumentUploadResponse,
    db_path: Path
) -> None:
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    document_json
                )
                VALUES(?, ?)
                """,
                (
                    str(document.document_id),
                    document.model_dump_json()
                )
            )

def get_document_record(
    document_id: UUID,
    db_path: Path
) -> DocumentUploadResponse | None:
    with closing(sqlite3.connect(db_path)) as connection:
        row = connection.execute(
            """
            SELECT document_json
            FROM documents
            where document_id = ?
            """,
            (str(document_id),),
        ).fetchone()

    if row is None:
        return None

    validated_row = DocumentUploadResponse.model_validate_json(row[0])
    return validated_row

def update_document_record(
    document: DocumentUploadResponse,
    db_path: Path,
) -> None:
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET document_json = ?
                WHERE document_id = ?
                """,
                (
                    document.model_dump_json(),
                    str(document.document_id),
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError("Document record was not found")
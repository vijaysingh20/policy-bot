"""Read one uploaded document from the application's existing SQLite database.

Place this file in backend/ and run from the project root:
    uv run -m backend.inspect_document_store <document-id>

This script does not initialize tables, insert records, or call any AI models.
"""

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from backend.config import DATABASE_DIR
from backend.schemas import DocumentUploadResponse


def read_document(
    document_id: UUID,
    db_path: Path,
) -> DocumentUploadResponse | None:
    # mode=ro prevents writes and prevents creating an empty database by mistake.
    database_uri = db_path.resolve().as_uri() + "?mode=ro"

    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        row = connection.execute(
            """
            SELECT document_json
            FROM documents
            WHERE document_id = ?
            """,
            (str(document_id),),
        ).fetchone()

    if row is None:
        return None

    document = DocumentUploadResponse.model_validate_json(row[0])
    if document.document_id != document_id:
        raise ValueError("The stored JSON document ID does not match its row ID.")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an uploaded document record without changing SQLite."
    )
    parser.add_argument("document_id", type=UUID, help="Document ID returned by POST /documents")
    args = parser.parse_args()

    db_path = DATABASE_DIR / "evaluations.sqlite3"
    print("Database:", db_path.resolve())

    if not db_path.is_file():
        print("Database not found. Check DATABASE_DIR and run the API startup first.")
        return 1

    try:
        document = read_document(args.document_id, db_path)
    except sqlite3.Error as exc:
        print(f"Could not read the document table: {exc}")
        return 1
    except (ValidationError, ValueError) as exc:
        print(f"Stored document validation failed: {exc}")
        return 1

    if document is None:
        print("Document not found:", args.document_id)
        print("Confirm that POST /documents registered it in this database.")
        return 1

    print("Document found; its stored metadata passed Pydantic validation.")
    print(document.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

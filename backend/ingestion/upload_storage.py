from pathlib import Path
from typing import BinaryIO
from uuid import UUID
import shutil
from backend.config import UPLOAD_DIR


def save_uploaded_pdf(
    source: BinaryIO,
    document_id: UUID,
    upload_dir: Path
) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)

    destination = upload_dir / f"{document_id}.pdf"
    source.seek(0)
    target = destination.open("xb")

    try:
        with target:
            shutil.copyfileobj(
                source,
                target,
                length=1024*1024
            )
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise

    return destination
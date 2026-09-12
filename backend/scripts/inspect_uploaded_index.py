from uuid import UUID

from backend.config import INDEX_DIR, UPLOAD_DIR
from backend.ingestion import build_index


def main() -> None:
    document_id = UUID(
        "9bc39ef3-50c3-49a2-b5c0-2154553bc93d"
    )

    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    output_dir = INDEX_DIR.parent / str(document_id)

    # Prevent this inspection run from overwriting existing assets.
    if output_dir.exists():
        raise FileExistsError(
            f"An output directory already exists: {output_dir}"
        )

    index, embedded_chunks, manifest, _model = build_index(
        pdf_path=pdf_path,
        output_dir=output_dir,
        source_name="handbook.pdf",
    )

    print("\nDocument ID:", document_id)
    print("Assets directory:", output_dir)
    print("Indexed vectors:", index.ntotal)
    print("Dimensions:", index.d)
    print("Manifest chunks:", manifest.chunk_count)
    print(
        "First source:",
        embedded_chunks[0].chunk.metadata.model_dump(),
    )


if __name__ == "__main__":
    main()
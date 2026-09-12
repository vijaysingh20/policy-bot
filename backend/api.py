from uuid import UUID, uuid4
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import HealthResponse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from backend.config import INDEX_DIR, DATABASE_DIR, UPLOAD_DIR, MAX_UPLOAD_BYTES
from backend.embedding import load_model_for_query
from backend.retrieval import load_reranker, load_retrieval_assets
from backend.schemas import QuestionResponse, QuestionRequest, EvaluationState, UploadInfo, DocumentUploadResponse
from backend.app.query import run_question
from backend.evaluation import initialize_evaluation_store, get_evaluation_state
from backend.ingestion import save_uploaded_pdf, initialize_document_store, create_document_record, get_document_record, update_document_record, build_index

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Loading retrieval resources...")
    index, chunks, manifest = load_retrieval_assets(INDEX_DIR)

    app.state.index = index
    app.state.chunks = chunks
    app.state.manifest = manifest

    app.state.embedding_model = load_model_for_query(manifest)
    app.state.reranker = load_reranker()
    app.state.document_resources = {}

    print("Loaded vectors:", app.state.index.ntotal)

    db_path = DATABASE_DIR / "evaluations.sqlite3"
    initialize_evaluation_store(db_path)
    initialize_document_store(db_path)

    app.state.evaluation_db_path = db_path

    try:
        yield
    finally:
        print("Releasing retrieval resources...")
        del app.state.index
        del app.state.chunks
        del app.state.manifest
        del app.state.embedding_model
        del app.state.reranker
        del app.state.evaluation_db_path
        del app.state.document_resources

app = FastAPI(
    title="HR Policy Bot",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="hr-policy-bot-backend",
    )

@app.post("/questions", response_model=QuestionResponse)
def ask_question(
    body: QuestionRequest,
    request: Request,
    background_tasks: BackgroundTasks
) -> QuestionResponse:
    state = request.app.state

    index = state.index
    chunks = state.chunks
    manifest = state.manifest
    embedding_model = state.embedding_model

    if body.document_id is not None:
        document = get_document_record(
            document_id=body.document_id,
            db_path=state.evaluation_db_path
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

        if document.status != "ready":
            raise HTTPException(
                status_code=409,
                detail="The document must be indexed before asking questions.",
            )

        if body.document_id not in state.document_resources:
            assets_dir = INDEX_DIR.parent / str(body.document_id)

            try:
                loaded_index, loaded_chunks, loaded_manifest = (
                    load_retrieval_assets(assets_dir)
                )
            except (OSError, ValueError, RuntimeError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail="The document's retrieval assets are unavailable.",
                ) from exc

            same_embedding_config = (
                loaded_manifest.embedding_model == state.embedding_model
                and loaded_manifest.embedding_revision == state.manifest.embedding_revision
                and loaded_manifest.dimensions == state.manifest.dimensions
            )

            loaded_model = (
                state.embedding_model if same_embedding_config else load_model_for_query(loaded_manifest)
            )

            state.document_resources[body.document_id] = (
                loaded_index,
                loaded_chunks,
                loaded_manifest,
                loaded_model,
            )
        index, chunks, manifest, embedding_model = (
            state.document_resources[body.document_id]
        )

    return run_question(
        question=body.question,
        index=state.index,
        chunks=state.chunks,
        manifest=state.manifest,
        embedding_model=state.embedding_model,
        reranker=state.reranker,
        background_tasks=background_tasks,
        db_path=state.evaluation_db_path,
    )

@app.get(
    "/evaluations/{evaluation_id}",
    response_model= EvaluationState
)
def read_evaluation(
    evaluation_id: UUID,
    request: Request
) -> EvaluationState:
    result = get_evaluation_state(
        evaluation_id=evaluation_id,
        db_path=request.app.state.evaluation_db_path
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found"
        )
    return result

@app.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=201
)
def upload_document(
    file: UploadFile,
    request: Request
) -> DocumentUploadResponse:
    try:
        file.file.seek(0,2)
        size_bytes = file.file.tell()
        file.file.seek(0)

        if size_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        if size_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The maximum supported file size is 10 MiB."
            )

        reader = PdfReader(file.file)

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Encrypted PDF's are not supported yet."
            )

        page_count = len(reader.pages)

        if page_count == 0:
            raise HTTPException(
                status_code=400,
                detail="The PDF contains no pages."
            )

        document_id = uuid4()

        document = DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            content_type=file.content_type,
            page_count=page_count,
            size_bytes=size_bytes
        )

        saved_path = save_uploaded_pdf(
            source=file.file,
            document_id=document_id,
            upload_dir=UPLOAD_DIR
        )

        try:
            create_document_record(
                document=document,
                db_path=request.app.state.evaluation_db_path
            )
        except Exception as exc:
            saved_path.unlink(missing_ok=True)
            raise
        return document
    except PdfReadError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a PDF."
        ) from exc
    finally:
        file.file.seek(0)


@app.post(
    "/documents/{document_id}/ingest",
    response_model=DocumentUploadResponse
)
def ingest_document(
    document_id: UUID,
    request: Request
)-> DocumentUploadResponse:
    db_path = request.app.state.evaluation_db_path

    document = get_document_record(
        document_id=document_id,
        db_path=db_path
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    assets_dir = INDEX_DIR.parent / str(document_id)

    if not pdf_path.is_file():
        raise HTTPException(
            status_code=409,
            detail="The document record exists, but it's pdf is missing."
        )

    if not assets_dir.exists():
        build_index(
            pdf_path=pdf_path,
            output_dir=assets_dir,
            source_name=document.filename or "document.pdf"
        )

    try:
        load_retrieval_assets(assets_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The document's retrieval assets could not be loaded. "
                "Ingestion needs repair before this document can be used."
            ),
        ) from exc

    ready_document = DocumentUploadResponse.model_validate({
        **document.model_dump(),
        "status": "ready"
    })

    update_document_record(
        document=ready_document,
        db_path=db_path
    )

    return ready_document

@app.post("/upload/inspect", response_model=UploadInfo)
def inspect_upload(file: UploadFile) -> UploadInfo:
    try:
        file.file.seek(0)
        reader = PdfReader(file.file)

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Encrypted PDFs are not supported yet."
            )

        page_count = len(reader.pages)
        if page_count == 0:
            raise HTTPException(
                status_code=400,
                detail="The PDF contains no pages."
            )
        
        return UploadInfo(
            filename= file.filename,
            content_type= file.content_type,
            page_count=page_count
        )
    except PdfReadError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a PDF."
        ) from exc
    finally:
        file.file.seek(0)
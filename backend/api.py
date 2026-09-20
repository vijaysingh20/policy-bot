import logging
from pathlib import Path

from uuid import UUID, uuid4
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import HealthResponse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from backend.config import CORS_ORIGINS, DATABASE_DIR, EMBEDDING_MODEL, INDEX_DIR, MAX_UPLOAD_BYTES, UPLOAD_DIR
from backend.embedding import load_model_for_ingestion
from backend.retrieval import DocumentResources, load_document_resources, load_reranker, load_retrieval_assets
from backend.schemas import QuestionResponse, QuestionRequest, EvaluationState, DocumentUploadResponse
from backend.app.query import run_question
from backend.evaluation import initialize_evaluation_store, get_evaluation_state
from backend.ingestion import save_uploaded_pdf, initialize_document_store, create_document_record, get_document_record, update_document_record, build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_resources_for(assets_dir: Path, state) -> DocumentResources:
    return load_document_resources(
        assets_dir,
        shared_model=state.embedding_model,
        shared_model_name=EMBEDDING_MODEL,
        shared_revision=state.embedding_revision
    )

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Loading embedding model and reranker...")
    app.state.embedding_model, app.state.embedding_revision = (
        load_model_for_ingestion(EMBEDDING_MODEL)
    )
    app.state.reranker = load_reranker()
    app.state.document_resources = {}

    try:
        app.state.default_resources = load_resources_for(INDEX_DIR, app.state)
        logger.info("Default handbook loaded: %d vectors", app.state.default_resources.index.ntotal)
    except FileNotFoundError:
        app.state.default_resources = None
        logger.info("No default handbook index at %s; upload a PDF to begin.", INDEX_DIR)

    db_path = DATABASE_DIR / "evaluations.sqlite3"
    initialize_evaluation_store(db_path)
    initialize_document_store(db_path)
    app.state.evaluation_db_path = db_path

    yield

app = FastAPI(
    title="HR Policy Bot",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="hr-policy-bot-backend",
    )

def resolve_document_resources(
    document_id: UUID | None,
    state
) -> DocumentResources:
    if document_id is None:
        if state.default_resources is None:
            raise HTTPException(
                status_code=409,
                detail="No default handbook is loaded. Upload a PDF and pass its document_id.",
            )
        return state.default_resources

    document = get_document_record(document_id=document_id, db_path=state.evaluation_db_path)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if document.status != "ready":
        raise HTTPException(
            status_code=409,
            detail="The document must be indexed before asking questions.",
        )

    cached = state.document_resources.get(document_id)
    if cached is not None:
        return cached

    try:
        resources = load_resources_for(INDEX_DIR.parent / str(document_id), state)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=409,
            detail="The document's retrieval assets are unavailable.",
        ) from exc

    state.document_resources[document_id] = resources
    return resources


@app.post("/questions", response_model=QuestionResponse)
def ask_question(
    body: QuestionRequest,
    request: Request,
    background_tasks: BackgroundTasks
) -> QuestionResponse:
    state = request.app.state

    resources = resolve_document_resources(body.document_id, state)

    return run_question(
        question=body.question,
        index=resources.index,
        chunks=resources.chunks,
        manifest=resources.manifest,
        embedding_model=resources.embedding_model,
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
            source_name=document.filename or "document.pdf",
            model=request.app.state.embedding_model,
            model_revision=request.app.state.embedding_revision,
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

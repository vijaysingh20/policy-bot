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
from backend.ingestion import save_uploaded_pdf

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Loading retrieval resources...")
    index, chunks, manifest = load_retrieval_assets(INDEX_DIR)

    app.state.index = index
    app.state.chunks = chunks
    app.state.manifest = manifest

    app.state.embedding_model = load_model_for_query(manifest)
    app.state.reranker = load_reranker()

    print("Loaded vectors:", app.state.index.ntotal)

    db_path = DATABASE_DIR / "evaluations.sqlite3"
    initialize_evaluation_store(db_path)

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
def upload_document(file: UploadFile) -> DocumentUploadResponse:
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

        save_pdf = save_uploaded_pdf(
            source=file.file,
            document_id=document_id,
            upload_dir=UPLOAD_DIR
        )

        return DocumentUploadResponse(
            filename=file.filename,
            content_type=file.content_type,
            page_count=page_count,
            size_bytes=size_bytes,
            document_id=document_id
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a PDF."
        ) from exc
    finally:
        file.file.seek(0)

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
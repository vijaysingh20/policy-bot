from uuid import UUID

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from backend.schemas import HealthResponse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from backend.config import INDEX_DIR, DATABASE_DIR
from backend.embedding import load_model_for_query
from backend.retrieval import load_reranker, load_retrieval_assets
from backend.schemas import QuestionResponse, QuestionRequest, EvaluationState
from backend.app.query import run_question
from backend.evaluation import initialize_evaluation_store, get_evaluation_state

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
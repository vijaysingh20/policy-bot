import logging
from pathlib import Path
from uuid import uuid4

import faiss
from fastapi import BackgroundTasks
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.evaluation import create_evaluation_state, save_evaluation_record
from backend.evaluation.worker import run_evaluation_job
from backend.generation import (
    build_answer_chain,
    build_context,
    build_query_planner,
    resolve_answer_sources,
    sanitize_citations
)
from backend.retrieval import retrieve_for_plan
from backend.schemas import (
    AnswerDraft,
    ChunkRecord,
    EvaluationJob,
    EvaluationState,
    IndexManifest,
    QueryPlan,
    QuestionResponse,
    RetrievalContext
)

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_ANSWER = (
    "I can help with questions about the supplied HR "
    "handbook. Please ask an HR-policy question."
)
OUT_OF_SCOPE_REASON = "Question is outside the HR-policy scope."


@traceable(name="hrpolicy_bot_question", process_inputs=lambda inputs: {"question": inputs["question"]})
def run_question(
    question: str,
    *,
    index: faiss.Index,
    chunks: list[ChunkRecord],
    manifest: IndexManifest,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    background_tasks: BackgroundTasks,
    db_path: Path
) -> QuestionResponse:
    evaluation_id = uuid4()

    run = get_current_run_tree()
    trace_id = run.id if run is not None else None
    project_name = run.session_name if run is not None else None

    plan: QueryPlan = build_query_planner().invoke({"question": question})
    logger.info("Query plan: %s", plan.model_dump_json())

    if plan.route == "out_of_scope":
        draft = AnswerDraft(answer=OUT_OF_SCOPE_ANSWER, source_ids=[])
        create_evaluation_state(
            state=EvaluationState(
                evaluation_id=evaluation_id, status="skipped", reason=OUT_OF_SCOPE_REASON
            ),
            db_path=db_path
        )

        try:
            save_evaluation_record(
                trace_id=trace_id,
                question=question,
                plan=plan,
                draft=draft,
                context=RetrievalContext(context_text="", sources=[]),
                status="skipped",
                reason=OUT_OF_SCOPE_REASON
            )
        except Exception:
            logger.exception("Could not log skipped evaluation %s", evaluation_id)

        return QuestionResponse(
            answer=draft.answer,
            sources=[],
            evaluation_id=evaluation_id,
            evaluation_status="skipped",
            scores=None,
            trace_id=trace_id
        )

    matches = retrieve_for_plan(
        plan=plan,
        index=index,
        chunks=chunks,
        embedding_model=embedding_model,
        reranker=reranker
    )
    context = build_context()

    draft: AnswerDraft = build_answer_chain().invoke(
        {"context": context.context_text, "question": question}
    )
    draft = sanitize_citations(draft, context)
    resolved_sources = resolve_answer_sources(draft, context)

    create_evaluation_state(state=EvaluationState(evaluation_id=evaluation_id), db_path=db_path)
    background_tasks.add_task(
        run_evaluation_job,
        job=EvaluationJob(
            evaluation_id=evaluation_id,
            question=question,
            query_plan=plan,
            draft=draft,
            context=context,
            manifest=manifest,
            trace_id=trace_id,
            project_name=project_name
        ),
        db_path=db_path
    )

    return QuestionResponse(
        answer=draft.answer,
        sources=resolved_sources,
        evaluation_id=evaluation_id,
        evaluation_status="pending",
        scores=None,
        trace_id=trace_id,
    )
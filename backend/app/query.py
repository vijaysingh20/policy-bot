import logging

import faiss
from pathlib import Path
from uuid import uuid4
from fastapi import BackgroundTasks
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.generation import (
    build_answer_chain,
    build_context,
    build_query_planner,
    resolve_answer_sources,
)
from backend.observability import print_tracing_status
from backend.retrieval import (
    retrieve_for_plan,
)
from backend.schemas import (
    AnswerDraft,
    ChunkRecord,
    IndexManifest,
    QueryPlan,
    RetrievalContext,
    QuestionResponse,
    EvaluationJob,
    EvaluationState
)
from backend.helper import save_evaluation_records
from backend.evaluation.worker import run_evaluation_job
from backend.evaluation import create_evaluation_state

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_ANSWER = (
    "I can help with questions about the supplied HR "
    "handbook. Please ask an HR-policy question."
)

@traceable(
    name="hrpolicy_bot_question",
    process_inputs=lambda inputs: {
        "question": inputs["question"],
    }
)
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
    question_run = get_current_run_tree()
    evaluation_id = uuid4()

    trace_id = question_run.id if question_run is not None else None
    project_name = (
        question_run.session_name
        if question_run is not None
        else None
    )
    print_tracing_status()

    planner = build_query_planner()

    plan: QueryPlan = planner.invoke({
        "question": question
    })
    print("\nQuery plan:")
    print(plan.model_dump_json(indent=2))

    if plan.route == "out_of_scope":
        draft = AnswerDraft(
            answer=OUT_OF_SCOPE_ANSWER,
            source_ids=[],
        )

        print("\nAnswer:")
        print(draft.answer)
        print("\nSources:\nNo sources cited.")

        context = RetrievalContext(
            context_text="",
            sources=[],
        )

        reason = "Question is outside the HR-policy scope."

        create_evaluation_state(
            state=EvaluationState(
                evaluation_id=evaluation_id,
                status="skipped",
                reason=reason,
            ),
            db_path=db_path,
        )

        try:
            save_evaluation_records(
                trace_id=question_run.id if question_run is not None else None,
                question=question,
                plan=plan,
                draft=draft,
                context=context,
                scores=None,
                status="skipped",
                reason="Question is outside the HR-policy scope.",
            )
        except Exception:
            logger.exception(
                "Could not log skipped evaluation %s",
                evaluation_id,
            )

        return QuestionResponse(
            answer=draft.answer,
            sources=[],
            evaluation_status="skipped",
            scores=None,
            trace_id=question_run.id if question_run is not None else None,
        )

    matches = retrieve_for_plan(
        plan=plan,
        index=index,
        chunks=chunks,
        embedding_model=embedding_model,
        reranker=reranker
    )

    print("\nUnique selected chunks:", len(matches))
    context = build_context(matches)

    chain = build_answer_chain()

    draft: AnswerDraft = chain.invoke({
        "context": context.context_text,
        "question": question
    })

    resolved_sources = resolve_answer_sources(draft, context)

    job = EvaluationJob(
        evaluation_id=evaluation_id,
        question=question,
        query_plan=plan,
        draft=draft,
        context=context,
        manifest=manifest,
        trace_id=trace_id,
        project_name=project_name,
    )

    response = QuestionResponse(
        answer=draft.answer,
        sources=resolved_sources,
        evaluation_id=evaluation_id,
        evaluation_status="pending",
        scores=None,
        trace_id=trace_id,
    )

    create_evaluation_state(
        state=EvaluationState(evaluation_id=evaluation_id),
        db_path=db_path,
    )

    background_tasks.add_task(
        run_evaluation_job,
        job=job,
        db_path=db_path,
    )

    return response

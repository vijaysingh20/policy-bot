import faiss
from pathlib import Path
from typing import Literal
from uuid import UUID
from langsmith import trace
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.config import INDEX_DIR, CHAT_MODEL
from backend.embedding import load_model_for_query
from backend.evaluation import (
    append_evaluation_record,
    CONTEXT_PRECISION_PROMPT_VERSION,
    build_evaluation_input,
    evaluate_faithfulness,
    faithfulness_scorer,
    evaluate_answer_relevancy,
    answer_relevancy_scorer,
    context_precision_scorer,
    evaluate_context_precision,
)

from backend.retrieval import (
    load_reranker,
    load_retrieval_assets,
)
from backend.schemas import (
    AnswerDraft,
    ChunkRecord,
    ContextSource,
    EvaluationRecord,
    EvaluationScores,
    IndexManifest,
    RetrievalContext,
)

def score_answer(
    question: str,
    draft: AnswerDraft,
    context: RetrievalContext,
    manifest: IndexManifest
) -> EvaluationScores:
    with trace(
        "ragas_evaluation",
        run_type="chain",
        metadata={
            "context_precision_variant": "without_reference",
            "context_precision_prompt_version": (
                CONTEXT_PRECISION_PROMPT_VERSION
            )
        }
    ) as evaluation_run:
        sample = build_evaluation_input(
            question=question,
            draft=draft,
            context=context,
        )

        faithfulness_score = evaluate_faithfulness(
            sample=sample,
            scorer=faithfulness_scorer(),
        )

        print("\nFaithfulness:", round(faithfulness_score, 4))

        relevancy_score = evaluate_answer_relevancy(
            sample=sample,
            scorer=answer_relevancy_scorer(manifest),
        )

        print("\nAnswer relevancy:", round(relevancy_score, 4))

        precision_score = evaluate_context_precision(
            sample=sample,
            scorer=context_precision_scorer(),
        )

        print("\nContext precision:", round(precision_score, 4))

        scores = EvaluationScores(
            faithfulness=faithfulness_score,
            answer_relevancy=relevancy_score,
            context_precision=precision_score
        )

        evaluation_run.end(
            outputs=scores.model_dump(mode="json")
        )

    return scores

def print_sources(sources: list[ContextSource]) -> None:
    print("\nSources:")

    if not sources:
        print("No sources cited.")

    for source in sources:
        metadata = source.chunk.metadata

        print(
            f"\n[{source.source_id}] "
            f"{metadata.source} — PDF page {metadata.page_number}"
            f" — Chunk {source.chunk.chunk_number}"
        )

def save_evaluation_records(
        trace_id: UUID | None,
        question, 
        plan,
        draft,
        context,
        scores: EvaluationScores | None = None,
        status: Literal["completed", "skipped", "failed"] = "completed",
        reason: str | None = None
    ) -> None:
    record = EvaluationRecord(
        trace_id=trace_id,
        question=question,
        query_plan=plan,
        draft=draft,
        context=context,
        scores=scores,
        status=status,
        reason=reason,
        evaluator_model=CHAT_MODEL,
        context_precision_prompt_version=CONTEXT_PRECISION_PROMPT_VERSION
    )

    log_path = (
        Path(__file__).resolve().parent
        / "logs"
        / "evaluations.jsonl"
    )

    append_evaluation_record(
        record=record,
        log_path=log_path
    )

    print("Evaluation record saved:", record.record_id)
    print("Log file:", log_path)

def load_query_resources(
    assets_dir: Path = INDEX_DIR
) -> tuple[
    faiss.Index,
    list[ChunkRecord],
    IndexManifest,
    SentenceTransformer,
    CrossEncoder,
]:
    with trace("load_retrieval_resources", run_type="chain"):
        index, chunks, manifest = load_retrieval_assets(assets_dir)

        embedding_model = load_model_for_query(manifest)
        reranker = load_reranker()

    return index, chunks, manifest, embedding_model, reranker


from pathlib import Path

import faiss
from langsmith import trace, traceable
from langsmith.run_helpers import get_current_run_tree
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.config import INDEX_DIR
from backend.embedding import load_model_for_query
from backend.evaluation import (
    CONTEXT_PRECISION_PROMPT_VERSION,
    answer_relevancy_scorer,
    build_evaluation_input,
    context_precision_scorer,
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_faithfulness,
    faithfulness_scorer,
)
from backend.generation import (
    build_answer_chain,
    build_context,
    build_query_planner,
    resolve_answer_sources,
)
from backend.observability import log_evaluation_feedback, print_tracing_status
from backend.retrieval import (
    load_reranker,
    load_retrieval_assets,
    retrieve_for_plan,
)
from backend.schemas import (
    AnswerDraft,
    ChunkRecord,
    ContextSource,
    EvaluationScores,
    IndexManifest,
    QueryPlan,
    RetrievalContext,
)

OUT_OF_SCOPE_ANSWER = (
    "I can help with questions about the supplied HR "
    "handbook. Please ask an HR-policy question."
)


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

@traceable(name="hrpolicy_bot_question")
def run_question(question: str) -> AnswerDraft:
    question_run = get_current_run_tree()
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
        return draft

    (
        index,
        chunks,
        manifest,
        embedding_model,
        reranker,
    ) = load_query_resources()

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

    print("\nAnswer:")
    print(draft.answer)

    scores = score_answer(
        question=question,
        draft=draft,
        context=context,
        manifest=manifest,
    )

    print_sources(resolved_sources)

    if question_run is not None:
        print(f"question run found {question_run.id}")
        log_evaluation_feedback(
            run_id=question_run.id,
            project_name=question_run.session_name,
            scores=scores
        )

    return draft

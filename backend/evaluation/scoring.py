"""Run the three RAGAS metrics for one answered question."""

from langsmith import trace

from backend.evaluation.metrics import (
    answer_relevancy_scorer,
    build_evaluation_input,
    context_precision_scorer,
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_faithfulness,
    faithfulness_scorer
)
from backend.evaluation.prompts import CONTEXT_PRECISION_PROMPT_VERSION
from backend.schemas import AnswerDraft, EvaluationScores, IndexManifest, RetrievalContext


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
            "context_precision_prompt_version": CONTEXT_PRECISION_PROMPT_VERSION
        },
    ) as run: 
        sample = build_evaluation_input(question=question, draft=draft, context=context)
        scores =  EvaluationScores(
            faithfulness=evaluate_faithfulness(sample, faithfulness_scorer()),
            answer_relevancy=evaluate_answer_relevancy(sample, answer_relevancy_scorer(manifest)),
            context_precision=evaluate_context_precision(sample, context_precision_scorer())
        )
        run.end(outputs=scores.model_dump(mode="json"))

    return scores
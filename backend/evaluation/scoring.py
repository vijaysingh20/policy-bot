"""Run the three RAGAS metrics for one answered question."""

from backend.evaluation.metrics import (
    answer_relevancy_scorer,
    build_evaluation_input,
    context_precision_scorer,
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_faithfulness,
    faithfulness_scorer
)

from backend.schemas import AnswerDraft, EvaluationScores, IndexManifest, RetrievalContext


def score_answer(
    question: str,
    draft: AnswerDraft,
    context: RetrievalContext,
    manifest: IndexManifest
) -> EvaluationScores:
    sample = build_evaluation_input(question=question, draft=draft, context=context)

    return EvaluationScores(
        faithfulness=evaluate_faithfulness(sample, faithfulness_scorer()),
        answer_relevancy=evaluate_answer_relevancy(sample, answer_relevancy_scorer(manifest)),
        context_precision=evaluate_context_precision(sample, context_precision_scorer())
    )
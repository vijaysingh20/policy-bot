import logging
from functools import lru_cache

from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)

from backend.config import (
    CHAT_MODEL,
    DEVICE,
    EVALUATION_TIMEOUT_SECONDS,
    MAX_RETRIES,
)
from backend.evaluation.prompts import UsefulContextPrompt
from backend.schemas import (
    AnswerDraft,
    EvaluationInput,
    IndexManifest,
    RetrievalContext,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def evaluator_embeddings(model_name: str, revision: str) -> HuggingFaceEmbeddings:
    """Load the embedding model once per process, not once per evaluation."""
    return HuggingFaceEmbeddings(
        model=model_name,
        revision=revision,
        device=DEVICE,
        normalize_embeddings=True,
    )


def build_evaluator_llm():
    client = AsyncOpenAI(
        timeout=EVALUATION_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES
    )

    return llm_factory(
        CHAT_MODEL,
        client=client,
        temperature=0,
    )

def build_evaluation_input(
    question: str,
    draft: AnswerDraft,
    context: RetrievalContext
) -> EvaluationInput:
    retrieved_context: list[str] = []
    for source in context.sources:
        retrieved_context.append(source.chunk.text)

    return EvaluationInput(
        user_input=question.strip(),
        response=draft.answer,
        retrieved_contexts=retrieved_context
    )

def faithfulness_scorer() -> Faithfulness:
    return Faithfulness(llm=build_evaluator_llm())

def evaluate_faithfulness(
    sample: EvaluationInput,
    scorer: Faithfulness
) -> float:
    if not sample.retrieved_contexts:
        raise ValueError("This evaluation requires retrieved context")

    result = scorer.score(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts,
    )

    return float(result.value)

def answer_relevancy_scorer(
    manifest: IndexManifest
) -> AnswerRelevancy:
    return AnswerRelevancy(
        llm=build_evaluator_llm(),
        embeddings=evaluator_embeddings(manifest.embedding_model, manifest.embedding_revision),
    )


def evaluate_answer_relevancy(
    sample: EvaluationInput,
    scorer: AnswerRelevancy
) -> float:
    result = scorer.score(
        user_input = sample.user_input,
        response = sample.response
    )

    return float(result.value)

def context_precision_scorer() -> ContextPrecisionWithoutReference:
    scorer = ContextPrecisionWithoutReference(llm=build_evaluator_llm())
    scorer.prompt = UsefulContextPrompt()

    return scorer

def evaluate_context_precision(
    sample: EvaluationInput,
    scorer: ContextPrecisionWithoutReference
) -> float:
    if not sample.retrieved_contexts:
        raise ValueError("This evaluation requires retrieved context")

    result = scorer.score(
        user_input = sample.user_input,
        response = sample.response,
        retrieved_contexts = sample.retrieved_contexts
    )
    logger.debug("Context precision %r: %s", result.value, result.reason)

    return float(result.value)

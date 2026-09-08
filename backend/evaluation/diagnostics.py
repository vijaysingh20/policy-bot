"""Ad-hoc probes for understanding a metric's raw judgments."""

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecisionWithoutReference
from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
)

from backend.config import CHAT_MODEL, EVALUATION_TIMEOUT_SECONDS, MAX_RETRIES
from backend.evaluation.prompts import UsefulContextPrompt
from backend.schemas import EvaluationInput


async def inspect_context_precision(
    sample: EvaluationInput,
) -> None:
    control = EvaluationInput(
        user_input="How can an employee review their personnel record?",
        response=(
            "The employee should contact the Office of Human Resources."
        ),
        retrieved_contexts=[
            "Employees who wish to review their personnel record "
            "should contact the Office of Human Resources."
        ],
    )

    async with AsyncOpenAI(
        timeout=EVALUATION_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    ) as client:
        evaluator_llm = llm_factory(
            CHAT_MODEL,
            client=client,
            temperature=0,
        )

        scorer = ContextPrecisionWithoutReference(llm=evaluator_llm)
        scorer.prompt = UsefulContextPrompt()

        for label, current in [
            ("Positive control", control),
            ("Actual sample", sample),
        ]:
            print(f"\n=== {label} ===")
            print("Question:", current.user_input)
            print("Answer:", current.response)

            for rank, text in enumerate(
                current.retrieved_contexts,
                start=1,
            ):
                metric_input = ContextPrecisionInput(
                    question=current.user_input,
                    context=text,
                    answer=current.response,
                )

                judgment = await scorer.llm.agenerate(
                    scorer.prompt.to_string(metric_input),
                    ContextPrecisionOutput,
                )

                print(f"\nContext {rank}:")
                print(text)
                print(judgment.model_dump_json(indent=2))

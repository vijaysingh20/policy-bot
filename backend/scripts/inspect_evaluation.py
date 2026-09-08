from backend.evaluation import evaluate_faithfulness, faithfulness_scorer
from backend.schemas import EvaluationInput


def main() -> None:
    sample = EvaluationInput(
        user_input=(
            "How can an employee review their personnel record?"
        ),
        response=(
            "The employee should contact the Office of Human Resources. "
            "The employee must pay a 5,000-rupee processing fee."
        ),
        retrieved_contexts=[
            "Employees who wish to review their personnel record "
            "should contact the Office of Human Resources."
        ],
    )

    scorer = faithfulness_scorer()
    faithfulness_score = evaluate_faithfulness(
        sample=sample,
        scorer=scorer
    )
    print("\nFaithfulness:", round(faithfulness_score, 4))

if __name__ == "__main__":
    main()
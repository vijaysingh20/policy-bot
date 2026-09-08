from backend.generation import build_query_planner
from backend.schemas import QueryPlan

def main() -> None:
    planner = build_query_planner()

    questions = [
        "How can an employee review their personnel record?",
        (
            "How can an employee review their personnel record, "
            "and what is the annual leave policy?"
        ),
        "How do I bake a chocolate cake?",
    ]

    for question in questions:
        plan: QueryPlan = planner.invoke({
            "question": question,
        })

        print("\nQuestion:", question)
        print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
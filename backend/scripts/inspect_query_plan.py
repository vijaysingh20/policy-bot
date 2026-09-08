"""Check that QueryPlan rejects route/query combinations that disagree.

Run with: python -m backend.scripts.inspect_query_plan
"""

from pydantic import ValidationError

from backend.schemas import QueryPlan

CASES = [
    ("single", ["Q1", "Q2"]),
    ("out_of_scope", ["Q1", "Q2"]),
    ("decompose", ["Q1", "Q2"]),
]


def main() -> None:
    for route, queries in CASES:
        try:
            QueryPlan(route=route, queries=queries)
        except ValidationError:
            print(route, "REJECTED")
        else:
            print(route, "ACCEPTED")


if __name__ == "__main__":
    main()

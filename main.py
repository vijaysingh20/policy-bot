"""Ask the HR policy bot a question.

Run with: python main.py
"""

from langchain_core.tracers.langchain import wait_for_all_tracers

from backend.app import run_question

DEFAULT_QUESTION = (
    "How can an employee review their personnel record, "
    "and what is the annual leave policy?"
)


def main() -> None:
    run_question(question=DEFAULT_QUESTION)


if __name__ == "__main__":
    try:
        main()
    finally:
        wait_for_all_tracers()

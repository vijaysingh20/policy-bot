import os


def print_tracing_status() -> None:
    """Report whether LangSmith tracing is configured for this run."""
    print("Tracing:", os.getenv("LANGSMITH_TRACING"))
    print("Key loaded:", bool(os.getenv("LANGSMITH_API_KEY")))
    print("Project:", os.getenv("LANGSMITH_PROJECT"))
    print(
        "Endpoint:",
        os.getenv("LANGSMITH_ENDPOINT", "US default"),
    )

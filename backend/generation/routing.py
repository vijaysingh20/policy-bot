from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from backend.config import CHAT_MODEL, CHAT_TIMEOUT_SECONDS, MAX_RETRIES
from backend.generation.prompts import build_planner_prompt
from backend.schemas import QueryPlan

def build_query_planner() -> Runnable:
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        timeout=CHAT_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )

    structured_planner = llm.with_structured_output(
        QueryPlan,
        method="json_schema",
        strict=True
    )

    prompt = build_planner_prompt()

    return prompt | structured_planner

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from backend.config import CHAT_MODEL, CHAT_TIMEOUT_SECONDS, MAX_RETRIES
from backend.generation.prompts import build_answer_prompt
from backend.schemas import AnswerDraft

def build_answer_chain() -> Runnable:
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        timeout=CHAT_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES
    )

    structured_model = llm.with_structured_output(
        AnswerDraft,
        method="json_schema",
        strict=True
    )

    prompt = build_answer_prompt()

    return prompt | structured_model

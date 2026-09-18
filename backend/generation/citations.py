import logging
import re

from backend.schemas import AnswerDraft, ContextSource, RetrievalContext

logger = logging.getLogger(__name__)

CITATION = re.compile(r"\[(S\d+)\]")
CITATION_WITH_LEADING_SPACE = re.compile(r"\s*\[(S\d+)\]")


def sanitize_citations(draft: AnswerDraft, context: RetrievalContext) -> AnswerDraft:
    allowed = {source.source_id for source in context.sources}
    referenced = set(CITATION.findall(draft.answer)) | set(draft.source_ids)
    unknown = referenced - allowed
    if unknown:
        logging.warning("LLM cited unknown sources %s; removing them.", sorted(unknown))

    answer = CITATION_WITH_LEADING_SPACE.sub(
        lambda match: match.group(0) if match.group(1) in allowed else "",
        draft.answer,
    )

    inline_ids = list(dict.fromkeys(CITATION.findall(answer)))
    declare_only = [sid for sid in dict.fromkeys(draft.source_ids)
                    if sid in allowed and sid not in inline_ids]
    return AnswerDraft(answer=answer, source_ids=inline_ids + declare_only)


def resolve_answer_sources(
    draft: AnswerDraft, context: RetrievalContext
) -> list[ContextSource]:
    by_id = {source.source_id: source for source in context.sources}
    return [by_id[sid] for sid in draft.source_ids if sid in by_id]
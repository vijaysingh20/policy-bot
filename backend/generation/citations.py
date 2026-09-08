import re

from backend.schemas import AnswerDraft, ContextSource, RetrievalContext

def validate_answer_citations(
    draft: AnswerDraft,
    context: RetrievalContext
) -> None:
    allowed_ids = {
        source.source_id for source in context.sources
    }

    inline_ids = set(
        re.findall(r"\[(S\d+)\]", draft.answer)
    )

    declared_ids = set(draft.source_ids)

    referenced_ids = inline_ids | declared_ids

    invalid_ids = referenced_ids - allowed_ids

    if invalid_ids:
        raise ValueError(
            f"Unknown source IDs: {sorted(invalid_ids)}"
        )

    if inline_ids != declared_ids:
        raise ValueError("Mismatch between inline and declared IDs")

def resolve_answer_sources(
    draft: AnswerDraft,
    context: RetrievalContext
) -> list[ContextSource]:
    validate_answer_citations(draft, context)

    cited_ids = set(draft.source_ids)
    resolved_sources: list[ContextSource] = []

    for source in context.sources:
        if source.source_id in cited_ids:
            resolved_sources.append(source)

    return resolved_sources
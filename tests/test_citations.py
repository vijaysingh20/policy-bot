import logging

from backend.generation.citations import resolve_answer_sources, sanitize_citations
from backend.schemas import AnswerDraft


def test_valid_citations_pass_through_unchanged(make_context):
    draft = AnswerDraft(answer="You get 20 days of leave [S1] after probation [S2].",
                        source_ids=["S1", "S2"])
    clean = sanitize_citations(draft, make_context(3))
    assert clean == draft


def test_unknown_inline_citation_is_removed_not_raised(make_context, caplog):
    draft = AnswerDraft(answer="You get 20 days of leave [S1] [S9].", source_ids=["S1", "S9"])
    with caplog.at_level(logging.WARNING):
        clean = sanitize_citations(draft, make_context(3))
    assert clean.answer == "You get 20 days of leave [S1]."
    assert clean.source_ids == ["S1"]
    assert "S9" in caplog.text


def test_citation_used_inline_but_not_declared_is_still_shown(make_context):
    draft = AnswerDraft(answer="Remote work needs approval [S2].", source_ids=[])
    clean = sanitize_citations(draft, make_context(3))
    assert clean.source_ids == ["S2"]


def test_declared_but_not_inline_citation_is_kept(make_context):
    draft = AnswerDraft(answer="Remote work needs approval [S2].", source_ids=["S2", "S3"])
    assert sanitize_citations(draft, make_context(3)).source_ids == ["S2", "S3"]


def test_resolved_sources_carry_the_cited_pages(make_context):
    context = make_context(3)
    draft = sanitize_citations(
        AnswerDraft(answer="See [S3] and [S1].", source_ids=["S3", "S1"]), context
    )
    pages = [s.chunk.metadata.page_number for s in resolve_answer_sources(draft, context)]
    assert pages == [3, 1]
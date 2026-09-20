"""Measure what the cross-encoder adds: retrieval quality before vs after re-ranking.

Run from the project root:
    uv run python -m backend.evaluation.rerank_comparison

For every golden-set question with known relevant pages, FAISS retrieves one pool of
CANDIDATE_TOP_K chunks. "Before" keeps FAISS's own top FINAL_TOP_K; "after" lets the
cross-encoder pick FINAL_TOP_K from the *same* pool. Only the ordering differs, so any
change in the metrics is caused by the re-ranker alone.
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from backend.config import CANDIDATE_TOP_K, FINAL_TOP_K, HANDBOOK_PDF, INDEX_DIR, PROJECT_ROOT
from backend.retrieval.reranking import rerank_candidate
from backend.retrieval.vector_index import search_faiss
from backend.schemas import SearchRequest

GOLDEN_SET = PROJECT_ROOT / "eval" / "golden_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    relevant_pages: list[int]
    candidate_pages: list[int]      # FAISS pool, in FAISS order
    before_pages: list[int]         # FAISS top-k
    after_pages: list[int]          # cross-encoder top-k from the same pool
    before_rr: float
    after_rr: float


def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def reciprocal_rank(pages: list[int], relevant: set[int]) -> float:
    """1/rank of the first relevant page; 0 if none is present."""
    for rank, page in enumerate(pages, start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_question(
    item: dict,
    *,
    index,
    chunks,
    embedding_model,
    reranker,
    top_k: int = FINAL_TOP_K,
    candidate_k: int = CANDIDATE_TOP_K,
) -> QuestionResult:
    question = item["question"]
    relevant = set(item["relevant_pages"])

    candidates = search_faiss(
        SearchRequest(query=question, top_k=candidate_k), index, chunks, embedding_model
    )
    reranked = rerank_candidate(SearchRequest(query=question, top_k=top_k), candidates, reranker)

    candidate_pages = [c.chunk.metadata.page_number for c in candidates]
    before_pages = candidate_pages[:top_k]
    after_pages = [r.candidate.chunk.metadata.page_number for r in reranked]

    return QuestionResult(
        id=item["id"],
        category=item["category"],
        question=question,
        relevant_pages=sorted(relevant),
        candidate_pages=candidate_pages,
        before_pages=before_pages,
        after_pages=after_pages,
        before_rr=reciprocal_rank(before_pages, relevant),
        after_rr=reciprocal_rank(after_pages, relevant),
    )


def summarize(results: list[QuestionResult]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in results:
        groups[result.category].append(result)
        groups["all"].append(result)

    return {
        name: {
            "questions": len(group),
            "before_hit": mean(r.before_rr > 0 for r in group),
            "after_hit": mean(r.after_rr > 0 for r in group),
            "before_mrr": mean(r.before_rr for r in group),
            "after_mrr": mean(r.after_rr for r in group),
            "pool_recall": mean(
                any(p in r.relevant_pages for p in r.candidate_pages) for r in group
            ),
        }
        for name, group in groups.items()
    }


def render_markdown(summary: dict, results: list[QuestionResult], top_k: int, pool: int) -> str:
    lines = [
        f"| Category | Questions | Hit@{top_k} before | Hit@{top_k} after "
        f"| MRR@{top_k} before | MRR@{top_k} after | Recall@{pool} (pool) |",
        "|---|---|---|---|---|---|---|",
    ]
    order = sorted(summary, key=lambda name: (name == "all", name))
    for name in order:
        s = summary[name]
        label = f"**{name}**" if name == "all" else name
        lines.append(
            f"| {label} | {s['questions']} | {s['before_hit']:.2f} | {s['after_hit']:.2f} "
            f"| {s['before_mrr']:.2f} | {s['after_mrr']:.2f} | {s['pool_recall']:.2f} |"
        )

    changed = [r for r in results if r.after_rr != r.before_rr]
    if changed:
        lines += ["", "Questions where re-ranking changed the rank of the first correct page:", "",
                  "| Question | Relevant pages | Before (pages) | After (pages) |",
                  "|---|---|---|---|"]
        for r in sorted(changed, key=lambda r: r.after_rr - r.before_rr, reverse=True):
            lines.append(f"| {r.question} | {r.relevant_pages} | {r.before_pages} | {r.after_pages} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    from backend.embedding import load_model_for_query
    from backend.ingestion.pipeline import build_index
    from backend.retrieval import load_reranker, load_retrieval_assets

    if not (INDEX_DIR / "manifest.json").exists():
        print(f"No index at {INDEX_DIR}; building it from {HANDBOOK_PDF} ...")
        build_index(HANDBOOK_PDF, INDEX_DIR)

    index, chunks, manifest = load_retrieval_assets(INDEX_DIR)
    embedding_model = load_model_for_query(manifest)
    reranker = load_reranker()

    answerable = [item for item in load_golden_set() if item["relevant_pages"]]
    results = [
        evaluate_question(item, index=index, chunks=chunks,
                          embedding_model=embedding_model, reranker=reranker)
        for item in answerable
    ]
    summary = summarize(results)
    report = render_markdown(summary, results, FINAL_TOP_K, CANDIDATE_TOP_K)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rerank_comparison.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "rerank_comparison.json").write_text(
        json.dumps({"summary": summary, "questions": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    print(report)
    print(f"Saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
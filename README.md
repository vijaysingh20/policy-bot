---
title: HR Policy Bot
emoji: 📘
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

## Live Demo

- **App:** [https://YOUR-VERCEL-URL.vercel.app](https://YOUR-VERCEL-URL.vercel.app)
- **API:** https://vijaysinghdev-policy-bot.hf.space (interactive docs at `/docs`)
- **Repo:** https://github.com/vijaysingh20/policy-bot

## Overview

A single-source RAG assistant for employee handbooks. A user uploads one HR
policy PDF, asks questions in plain English, and gets an answer grounded in
that document with page-cited source chunks. Multi-part questions are
automatically decomposed into sub-queries; questions outside the document's
scope are declined rather than answered from the model's general knowledge.

## Overview

A single-source RAG assistant for employee handbooks. A user uploads one HR
policy PDF, asks questions in plain English, and gets an answer grounded in
that document with page-cited source chunks. Multi-part questions are
automatically decomposed into sub-queries; questions outside the document's
scope are declined rather than answered from the model's general knowledge.

Architecturally equivalent to the class reference build — embeddings →
FAISS retrieval → LCEL RAG chain → cross-encoder re-ranking → RAGAS
evaluation — applied to a different domain and document type (HR Policy
Bot, option 6 from the assignment brief).

## Architecture

\`\`\`mermaid
flowchart LR
    subgraph Ingestion
        A[PDF Upload] --> B[Load pages<br/>skip table-of-contents]
        B --> C[Chunk<br/>200 tokens, 40 overlap]
        C --> D[Embed<br/>all-MiniLM-L6-v2]
        D --> E[(FAISS index)]
    end

    subgraph "Query time"
        F[Question] --> G[LCEL Query Planner<br/>single / decompose / out_of_scope]
        G -->|out_of_scope| H[Decline, no retrieval]
        G -->|single or decompose| I[FAISS search<br/>top 20 candidates]
        E --> I
        I --> J[Cross-encoder re-rank<br/>ms-marco-MiniLM-L6-v2<br/>top 3]
        J --> K[LCEL Answer Chain<br/>gpt-4.1-mini]
        K --> L[Citation sanitizer<br/>drop unknown S# refs]
        L --> M[Answer + cited sources]
    end

    subgraph "Background evaluation"
        M -.-> N[RAGAS: faithfulness,<br/>answer relevancy,<br/>context precision]
        N --> O[(JSONL log)]
        N --> P[(SQLite evaluation state<br/>polled by frontend)]
    end
\`\`\`

## Tech stack

| Layer | Choice |
|---|---|
| Frontend | Next.js, TypeScript, Tailwind |
| Backend | FastAPI, Python 3.12, `uv` |
| Orchestration | LangChain (LCEL) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector store | FAISS (cosine similarity) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L6-v2` (local, CPU) |
| Generation & planning | OpenAI `gpt-4.1-mini` |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |
| Evaluation state | SQLite (polled by the frontend while RAGAS runs in the background) |
| Observability | LangSmith (optional — tracing is a no-op if `LANGSMITH_TRACING` is unset) |
| Backend hosting | Hugging Face Spaces (Docker) |
| Frontend hosting | Vercel |

## Setup

\`\`\`bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY

uv run uvicorn backend.api:app --reload      # backend on :8000
cd frontend && npm install && npm run dev    # frontend on :3000
\`\`\`

Set `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` in `frontend/.env.local` for
local development. The app is upload-driven — no PDF ships with the repo.

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/documents` | Upload a PDF, validated (size, encryption, page count) |
| `POST` | `/documents/{document_id}/ingest` | Build the FAISS index for an uploaded document |
| `POST` | `/questions` | Ask a question; optional `document_id` (defaults to the loaded handbook if any) |
| `GET` | `/evaluations/{evaluation_id}` | Poll for the background RAGAS result |

Full interactive schema at `/docs` on the live API.

## Retrieval & re-ranking evaluation

Evaluated with a 29-question golden set per document (`eval/golden_set.jsonl`,
`eval/golden_set_mississippi.jsonl`), each labeled with its expected route
and the PDF page(s) that answer it. `simple` questions use the handbook's
own wording; `paraphrase` questions deliberately use different words;
`multi_part` questions require decomposition; `not_in_document` and
`out_of_scope` check that the system declines rather than invents an answer.

For every answerable question, FAISS retrieves one shared pool of 20
candidates. "Before" keeps FAISS's own top 3; "after" lets the cross-encoder
re-rank the *same* pool down to 3 — isolating the re-ranker's effect from
retrieval itself. Metrics: Hit@3 (is a correct page in the top 3?) and
MRR@3 (how high does the first correct page rank?).

Reproduce with:

\`\`\`bash
uv run python -m backend.evaluation.rerank_comparison
uv run python -m backend.evaluation.rerank_comparison --pdf data/mississippi-handbook.pdf --golden eval/golden_set_mississippi.jsonl
\`\`\`

### Bennett College handbook (43 pages)

First pass showed re-ranking *hurting* retrieval:

| | Hit@3 | MRR@3 |
|---|---|---|
| FAISS only | 1.00 | 0.98 |
| + re-ranker (before fix) | 0.96 | 0.80 |

Error analysis traced this to table-of-contents pages: a TOC line like
`3.7 BEREAVEMENT LEAVE ... 26` is, to a cross-encoder reading question and
passage together, a near-perfect topical match — despite containing no
answer. A bi-encoder actually protects against this by accident, since a
TOC chunk's blended, multi-topic embedding scores low on similarity; the
cross-encoder has no such protection.

**Fix:** detect and skip table-of-contents pages at load time
(`is_table_of_contents` in `backend/ingestion/pdf_loader.py`) — a page
where most lines end in dot-leaders + a page number.

| | Hit@3 | MRR@3 |
|---|---|---|
| FAISS only | 1.00 | 0.98 |
| + re-ranker (after fix) | 1.00 | 0.84 |

Re-ranking still trails FAISS alone. With FAISS already near-perfect on
this document, there's almost no room for the re-ranker to help and every
mistake it makes shows up as a net loss. We then compared re-ranker models
directly:

| Re-ranker | Hit@3 | MRR@3 |
|---|---|---|
| FAISS only | 1.00 | 0.98 |
| `ms-marco-MiniLM-L6-v2` (current) | 1.00 | 0.84 |
| `BAAI/bge-reranker-base` (12× larger) | 1.00 | 0.89 |

**Decision rule, set before running the comparison:** switch models only if
one beats the FAISS baseline by ≥0.05 MRR. Neither does — `bge-reranker-base`
is better than the smaller model but still loses to no re-ranking at all —
so we kept `ms-marco-MiniLM-L6-v2`, the smallest and fastest option.

### Mississippi State Employee Handbook (63 pages, second document)

Chosen specifically because it has ten near-duplicate leave-type sections
in one chapter — the condition under which re-ranking should matter most.
`is_table_of_contents` was validated against a different layout here too:
it correctly flagged all 4 real TOC pages with zero false positives.

| Category | FAISS MRR | + `ms-marco-MiniLM-L6-v2` |
|---|---|---|
| simple | 0.69 | **0.92** (+0.23) |
| paraphrase | 0.79 | 0.69 (−0.10) |
| multi_part | 1.00 | 0.90 (−0.10) |
| **all** | **0.80** | 0.81 (+0.01) |

The baseline is much weaker here than on Bennett (0.80 vs 0.98 MRR),
confirming that documents with more topically-similar sections give a
bi-encoder less to work with. The re-ranker rescued several exact-wording
questions FAISS missed entirely (e.g. resignation notice: miss → rank 1),
but lost several paraphrased questions it previously got right, by
rewarding surface word overlap over meaning. The two effects roughly
cancel (net +0.01), below the 0.05 bar, so `ms-marco-MiniLM-L6-v2` was
kept unchanged. `bge-reranker-base` was also tested here and performed
worse (0.69 MRR) — the larger model that won on Bennett is not universally
better.

**Conclusion:** re-ranking's value is document-dependent. It helps most
when the bi-encoder baseline is weak (many similar sections) and on
exact-wording questions; it can hurt when the corpus has noisy
high-word-overlap-but-wrong-topic chunks (table of contents) or when
questions are heavily paraphrased. Full results and the underlying
per-question predictions are in `eval/results/`.

## RAGAS evaluation

Every answered question is scored in the background
(`backend/evaluation/rerank_comparison.py` handles retrieval eval;
`backend/evaluation/scoring.py` handles per-answer RAGAS) on three metrics,
logged to `logs/evaluations.jsonl`, and polled by the frontend via
`GET /evaluations/{id}` until complete.

Sample entries from a local run (trimmed for readability):

| Question | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| "Can I vape on campus?" | 1.00 | 0.84 | 0.83 |
| "Notice period for staff vs. faculty?" (decomposed) | 1.00 | 0.82 | 0.83 |
| "403(b) match percentage?" *(not in document)* | 0.50 | 0.00 | 0.00 |
| "How long is the probation period?" | 1.00 | 0.89 | 1.00 |

**On the 403(b) row:** the bot correctly declined to answer rather than
inventing a percentage — the desired behavior. The near-zero scores are a
known characteristic of RAGAS, not a failure of the app: answer relevancy
works by generating hypothetical questions from the *answer* and comparing
them to the real question, so a refusal ("I couldn't find enough
information...") generates hypothetical questions resembling nothing,
scoring near 0 almost by construction. Context precision correctly reports
that nothing retrieved was useful — because for this question, nothing in
the document is. Both retrieved near-empty header-only pages (3 and 43),
the same boilerplate-only pages identified during PDF loading.

Scores are intentionally not uniform. On this run, faithfulness averaged
0.875, answer relevancy 0.64, and context precision 0.67 across 4
completed evaluations — variation that reflects genuine differences in
question difficulty rather than a rubber-stamped metric.

**Limitation:** the evaluator LLM is the same model family as the
generator (`gpt-4.1-mini`), a known source of self-evaluation bias worth
noting for anyone extending this evaluation.

## Testing

105 tests across three layers, run with \`uv run pytest\`:

- **Unit** — schema validation, citation sanitization, chunking, the
  evaluation state machine. Fast, deterministic, no models loaded.
- **Fake-model** — retrieval and re-ranking logic tested against a real
  FAISS index with deterministic fake encoders (`tests/unit/test_retrieval.py`),
  and the full `run_question` flow with a faked LLM chain
  (`tests/unit/test_query.py`) — including a regression test for a bug
  where `build_context()` was called with no arguments, caught by writing
  the first test that actually exercised the in-scope answer path.
- **API** — FastAPI's `TestClient` against the real app with models
  monkeypatched, including CORS preflight behavior.

RAGAS/LLM-scored evaluation and the re-ranking comparison are excluded from
the default run (`-m "not slow and not eval"`) since they cost money and
call external services; run explicitly with \`uv run pytest -m eval\`.

## Deployment

- Backend: Docker image on Hugging Face Spaces (CPU Basic). Both models are
  baked into the image at build time, so cold starts don't require a
  download. PyTorch is pinned to the CPU-only wheel index to keep the
  image small.
- Frontend: Next.js on Vercel, `NEXT_PUBLIC_API_URL` pointed at the Space.
- **Storage is ephemeral.** Uploaded PDFs, the SQLite evaluation state, and
  the JSONL log do not persist across a Space restart — acceptable for a
  demo, called out here so it isn't mistaken for a bug.
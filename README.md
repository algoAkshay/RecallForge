# RecallForge

**Persistent research memory that knows when to remember and when to search.**

RecallForge is a local-first web-research agent that accumulates evidence across sessions, decides whether stored research is sufficient for a new question, and performs fresh retrieval only when necessary. Its routing is deterministic and explainable: freshness, semantic relevance, evidence coverage, and answerability determine whether a response uses MEMORY or WEB.

`Python 3.10+` · `Streamlit` · `LangChain` · `FAISS` · `134 offline tests` · [MIT License](LICENSE)

<!-- SCREENSHOT: Main RecallForge research workspace -->
<!-- Suggested file: docs/screenshots/research-workspace.png -->

<!-- SCREENSHOT: Route decision + sources/debug details -->
<!-- Suggested file: docs/screenshots/routing-debug.png -->

<!-- SCREENSHOT: Sidebar history / rename / export / delete -->
<!-- Suggested file: docs/screenshots/chat-history.png -->

## Why RecallForge

Many research assistants search the web on every question, forget prior evidence, reuse context that is merely similar rather than sufficient, and obscure why retrieval happened. RecallForge treats research as durable evidence: it reuses memory only when evidence clears deterministic quality gates and otherwise returns to the web.

## Core capabilities

- **Persistent research memory** — successful web evidence is retained across restarts in local FAISS storage.
- **Deterministic MEMORY / WEB routing** — route selection is a policy decision, not an opaque model preference.
- **Freshness detection** — current, latest, and breaking-information queries go directly to WEB.
- **Coverage and answerability gates** — similarity alone cannot authorize incomplete memory reuse.
- **Bounded web retrieval** — search, extraction, overall acquisition, and concurrency have explicit limits.
- **One-shot MEMORY → WEB recovery** — insufficient MEMORY synthesis can make one evidence-isolated WEB attempt.
- **Deterministic provenance** — application-owned `[S1]`, `[S2]` IDs prevent invented source URLs.
- **Persistent chat history** — SQLite threads support rename, delete, Markdown export, and relative timestamps.
- **Research controls and observability** — Search fresh and Debug details improve control without changing AUTO mode.

## Architecture

```text
┌──────────────────────────────┐
│          User query          │
└──────────────┬───────────────┘
               │
               ▼
       Freshness detection
               │
       ┌───────┴────────┐
       │                │
     Fresh          Not fresh
       │                │
       ▼                ▼
      WEB      Persistent FAISS memory
                        │
                        ▼
               Semantic retrieval
                        │
                        ▼
         Coverage + answerability gates
                        │
                ┌───────┴────────┐
                │                │
           Sufficient        Insufficient
                │                │
                ▼                ▼
             MEMORY            WEB
                └───────┬────────┘
                        ▼
             Evidence-grounded synthesis
                        │
                        ▼
             Citation validation
                        │
                        ▼
               Persisted chat history
```

## Routing pipeline

```text
freshness → semantic retrieval → semantic sufficiency → coverage → answerability → MEMORY / WEB
```

Freshness is evaluated before memory reuse. For non-fresh questions, FAISS L2 distance is lower-is-better: a strong result is at or below `0.5`; an acceptable result is at or below `1.0` and must meet the policy’s evidence requirements. These are current production constants, not universal embedding calibration.

Similarity is necessary but not enough. RecallForge also checks whether retrieved evidence covers the query’s key concepts and whether a short broad question has descriptive evidence about its actual subject.

## WEB reliability

| Control | Current implementation |
| --- | --- |
| DuckDuckGo search timeout | 6 seconds |
| Per-page fetch/extract timeout | 10 seconds |
| Total acquisition budget | 25 seconds |
| Maximum concurrent fetches | 4 |

Blocking fetch and extraction work is moved off the event loop. Successful sibling pages are retained when another source fails or times out, and the system returns explicit partial-failure states rather than silently treating incomplete research as success.

## Persistent memory and provenance

Research memory persists locally in `.storage/research_memory/`. Evidence identity uses SHA-256 content hashes to avoid duplicate insertion, and durable deduplication state is reconstructed after restart. FAISS persistence includes pickle-backed metadata, so only application-owned local storage should be loaded.

Citation identifiers are application-owned and deterministic. The synthesizer receives only valid evidence IDs; unknown IDs are removed or warned about, and the UI renders authoritative source records. Provenance integrity improves traceability, but it does not itself prove factual correctness or entailment.

## Research workspace and history

RecallForge keeps local SQLite research threads separate from semantic memory. Threads can be renamed, deleted, exported to Markdown, and revisited with human-readable timestamps. Deleting a chat does not clear research memory.

The workspace includes copy and source expansion, Light / Dark / System appearance modes, a one-shot **Search fresh** override, and optional **Debug details**. Current-request diagnostics can show route, mode, total latency, memory retrieval, search, fetch/extract, embedding/indexing, synthesis, fallback occurrence, source counts, and safe fetch outcomes. Full timing breakdown is intentionally not added to the historical chat schema.

## Evaluation

The included evaluation is deliberately **small, authored, offline, and controlled**. It is useful for regression detection and policy discussion, not as a statistically representative industry benchmark.

| Held-out controlled routing split | Result |
| --- | --- |
| Always-WEB | 8 / 16 correct |
| Always-MEMORY* | 11 / 16 correct |
| Semantic-only | 14 / 16 correct |
| RecallForge policy | **15 / 16 correct** |

`*` Always-MEMORY still sends freshness-sensitive queries to WEB.

- False MEMORY: **0**
- False WEB: **1**
- Weighted error: **1**
- Coverage ablation: semantic-only produced 2 false MEMORY decisions; the current policy produced 0.
- Freshness suite: 12 / 12 authored cases (TP 6, TN 6, FP 0, FN 0).

The evaluator labels its synthetic distance cases as unsuitable for production threshold calibration. Run it locally with `python -m src.eval.routing_benchmark`.

## Tech stack

| Layer | Technology |
| --- | --- |
| UI | Streamlit |
| Orchestration | Python, LangChain, LangGraph |
| Synthesis model | Gemini via `langchain-google-genai` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector memory | FAISS |
| Web search | DuckDuckGo (`ddgs`) |
| Extraction | trafilatura |
| Chat persistence | SQLite |
| Testing | pytest |

## Repository structure

```text
src/
├── agents/       # synthesis and bounded fallback
├── tools/        # routing, retrieval, web acquisition, provenance
├── storage/      # SQLite chat history
├── ui/           # Streamlit presentation and safe debug formatting
├── eval/         # offline controlled evaluation
└── documents/    # experimental, disabled by default
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m streamlit run src/main.py
```

Set `GOOGLE_API_KEY` in `.env` (or use `GEMINI_API_KEY`). On POSIX shells, activate with `source .venv/bin/activate`.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini credential used by the default synthesis model. |
| `GEMINI_API_KEY` | Supported alternative Gemini credential. |
| `MODEL` | Optional model identifier; defaults to `google_genai:gemini-3.5-flash-lite`. |
| `LINKMIND_MEMORY_PATH` | Legacy backward-compatible override for the local FAISS memory path. |

Use `.env.example` as the starting point. Never commit actual credentials.

## Representative usage

These examples describe expected behavior for a suitable evidence state, not guarantees for arbitrary pages or queries.

1. `Explain Redis persistence.` → likely **WEB** while no sufficient related memory exists.
2. A later related follow-up, `How does Redis persist data?` → may use **MEMORY** if retained evidence clears all gates.
3. `What is the latest stable Redis version?` → **WEB** because it requests current information.

Select **Search fresh** to explicitly use fresh WEB research for one question, even when memory could answer it.

## Engineering decisions

- Semantic similarity is not the same as answerability.
- Freshness is checked before memory reuse.
- Retrieval latency needs bounded, truthful failure behavior.
- Citation identity should be application-owned, not model-invented.
- Persistence adds both durability and trust/corruption boundaries.

## Testing

Default checks are offline and do not require Gemini, DuckDuckGo, or an embedding-model download:

```powershell
python -m compileall -q src tests
python -m pytest -q
python -m src.eval.routing_benchmark
```

They validate syntax/importability, unit and regression behavior, and controlled routing/freshness/coverage/persistence evaluation. The current suite contains **134 tests and 6 subtests**.

## Limitations

- The architecture is local-first and single-user; FAISS and SQLite are not distributed services.
- The evaluation corpus is small and authored.
- Synthesis requires a Gemini-compatible configured model.
- WEB answer quality depends on public pages retrieved at request time.
- There is no independent factual-entailment verifier.
- Rich request timing diagnostics are not fully persisted with historical messages.
- Experimental document ingestion is disabled in the default UI.

## Experimental document ingestion

RecallForge contains experimental local PDF ingestion and document-retrieval modules. They are intentionally disabled in the default UI while the primary online research-memory workflow remains the stable demo path.

## Security and trust notes

API keys are loaded from environment variables and are not printed by the application. Persistent FAISS loading relies on local trusted storage; only deserialize index data created by or trusted by the user.

## License

See [LICENSE](LICENSE). Existing license attribution and history are retained unchanged.

# Codebase-QA-Agent

Paste a public GitHub URL, and the app clones the repo, splits it into chunks, indexes them in ChromaDB and writes a summary card. You can then ask questions in a chat and get a streamed answer with `[file:lines]` citations. After each answer, the server checks every citation against the code it actually retrieved and marks it ✓ or ⚠ in the UI. Retrieval is hybrid (embeddings + BM25, merged with Reciprocal Rank Fusion). Everything was measured on a 24-question golden dataset, and the numbers below are real, including the ones that went against my expectations.

**Stack:** Python 3.11 · FastAPI · ChromaDB · `rank_bm25` · React + Vite · Docker / docker compose · Groq (`openai/gpt-oss-20b`) or Ollama (`llama3.1:8b`) through the `openai` SDK.

## Architecture

```mermaid
flowchart LR
    U["Browser<br/>React + Vite (nginx)"] -->|"POST /index"| API
    U -->|"POST /chat (SSE stream)"| API

    subgraph API["FastAPI backend (one container)"]
        direction TB
        IDX["RepoIndexer<br/>clone, chunk, embed"]
        RET["Hybrid retrieval<br/>vector + BM25 + RRF"]
        AGT["agent.stream_answer<br/>prompt, stream, verify citations"]
        SUM["summarizer"]
        LLMC["llm_client.py<br/>the only provider-aware file"]
        IDX --> CH[("ChromaDB<br/>PersistentClient<br/>one collection per repo")]
        IDX --> BM[("BM25 index<br/>JSON next to Chroma")]
        RET --> CH
        RET --> BM
        AGT --> RET
        AGT --> LLMC
        SUM --> LLMC
    end

    IDX -.->|"git clone --depth 1"| GH[("GitHub")]
    LLMC --> LLM["Groq gpt-oss-20b (hosted)<br/>or Ollama llama3.1:8b (local)"]
```

## Request flow

**`POST /index`** (a plain `def`, so FastAPI runs it in a worker thread and the server keeps answering other requests)

1. `clone`: shallow clone into a temp dir, after a size check (limit 200 MB, 5-minute timeout).
2. `chunk`: walk supported files and split them into line-window chunks (60 lines, 10 lines of overlap). Each chunk has `content`, `file`, `lines`, `language` and `node_type`.
3. In parallel:
   - `embed`: upsert the chunks into a per-repo ChromaDB collection (all-MiniLM-L6-v2 via ONNX, batches of 8), then build the BM25 index.
   - `summary`: one LLM call using the README, file listing and entry points. It doesn't need embeddings, so it overlaps with the slowest step.
4. Save metadata, delete the clone, and return stats plus `timings_ms` (clone / chunk / embed / summary / total).

**`POST /chat`** (Server-Sent Events)

1. Retrieve the top 8 chunks. Hybrid = top 24 from the vector search + top 24 from BM25, merged with RRF (k = 60), top 8 kept.
2. Send a `sources` event, then build the prompt (system prompt + last 6 history messages + the 8 chunks + the question).
3. Stream the model's tokens as `text` events.
4. Once the answer is complete, `verify_citations` parses every `[file:lines]` and sends a `confidence` event. It also sends per-citation results.
5. Send a `timings` event (retrieval ms, time to first token, total) and `[DONE]`.

## Design decisions and trade-offs

| Decision | Why | Cost / what I'd say about it honestly |
|---|---|---|
| **Sync `/index`, parallel summary** | An `async def` that does blocking work freezes the event loop. A plain `def` runs in the thread pool. The summary overlaps with embedding, so indexing takes about max(embed, summary), not the sum. | The summary (about 0.8 s) is completely hidden behind embedding, so the gain is small. The real bottleneck is embedding. |
| **ChromaDB `PersistentClient`, one collection per repo** | Indexed repos survive restarts (`load_persisted_repos()` restores them). Per-repo collections make deleting and re-indexing trivial. | In-process, so one container only. Render's free tier has no persistent disk, so there the index lasts only until the next deploy. |
| **Embedding model baked into the image, loaded at startup** | The first `/index` doesn't pay a download, and `/chat` doesn't pay a model load. | Image is bigger. |
| **Hybrid retrieval (BM25 + vector, RRF)** | The embedder only reads the first 256 tokens of each chunk (about half of the indexed text). BM25 reads the whole chunk. RRF merges by rank, so BM25 scores and cosine distances never need to be on the same scale. | BM25 is about 3.5 ms. On my eval, BM25 *alone* ranked better than hybrid (see Evaluation). I kept hybrid because the LLM reads all 8 chunks, so Recall@8 is the number that limits the answer. |
| **No cross-encoder reranker** | Doesn't fit the Render free tier's 512 MB, and the retrieval gap is already large enough that cheaper fixes should come first. | Listed under next steps. `RERANKER_ENABLED` in `.env.example` is a leftover; there is no reranker code. |
| **Function-aware Python chunker, built but off** | I measured it before shipping it. It put the right file at rank 1 more often but did not improve Recall@8 and made indexing `psf/requests` 53% slower. | It's behind `FUNCTION_AWARE_CHUNKS = False`, with its own tests and an eval flag (`--chunker ast`). |
| **`UPSERT_BATCH_SIZE = 8`** | Indexing `psf/requests` under a hard 512 MB limit was OOM-killed at a peak of 721 MiB with batch 100. Batch 8 peaks at about 353 MiB. | Embedding is about 35–45% slower. I chose staying on the free tier over speed. |
| **Verifiable citations** | The prompt asks for `[file:lines]`, but a prompt is a request, not a guarantee. `verify_citations` is a pure function (64 unit tests across the repo pass, 38 of them for citations). A citation is valid only if the file is indexed **and** the cited lines overlap a chunk retrieved for this answer. | It proves the citation points at code the model was given. It does **not** prove the claim is supported by that code. |
| **Provider abstraction** | `llm_client.py` is the only file that imports the `openai` SDK or names a provider. Ollama and Groq both speak the OpenAI API, so switching is an env var plus one function. | That's how the Groq model retirement was handled with a one-line change. |
| **No LangChain / LlamaIndex** | The pipeline is about a dozen small functions. Owning them made it possible to time every stage, swap retrieval methods behind one `method=` argument, and write the eval harness against the same code the server runs. | I give up the ready-made integrations. |
| **Model change** | Groq retired its Llama models, so the hosted app uses `openai/gpt-oss-20b` (`reasoning_effort="low"`). Local Ollama still runs `llama3.1:8b`. | **All numbers in this README were measured with gpt-oss-20b**, not Llama. |

## Evaluation

### Method

- **Golden dataset:** 24 questions over 2 repos (12 each): `psf/requests` pinned at `3c7baaa` and this repo pinned at `918a854`. Per repo there are 3 questions of each kind: *definition* ("where is X defined"), *flow* ("how does X work"), *config* and *identifier*. Every question lists the file(s) that contain the answer, and `python -m eval.golden_dataset --verify` checks the labels against the pinned checkouts.
- **Retrieval eval** (`python -m eval.eval_retrieval --method embedding|bm25|hybrid`): index the pinned repos and ask each question. It is deterministic for BM25 and on the host. In Docker, embedding runs varied by about one question's worth (one of four runs moved a single rank).
- **Answer eval** (`python -m eval.eval_answers`): run all 24 questions through the real `/chat` path in Docker and score the saved answers with the same `verify_citations` the server uses (`--rescore` re-scores a saved run without new LLM calls).
- **Caveat:** I wrote the questions knowing the code, so they reuse its vocabulary ("max redirects"). That favours keyword search. n = 24 means one question moves a metric by about 0.04, and a per-kind row (n = 6) by about 0.17. Read the per-kind rows as direction, not precision.

### What the metrics mean

| Metric | In plain English |
|---|---|
| **Recall@k** | Of the files that contain the answer, what share shows up in the top k retrieved chunks? k counts *chunks*, since the model is handed the top 8. Recall@8 is the ceiling on what the answer can be based on. |
| **MRR** (mean reciprocal rank) | 1 / the rank of the first chunk from a correct file, averaged over questions (0 if none is in the top 8). 1.0 means the right file is always first, 0.5 means it's typically second. It measures *ordering*. |
| **Citation validity** | Of the `[file:lines]` citations in an answer, the share where the file is indexed and the lines overlap a chunk retrieved for that answer. |
| **TTFT** | Time to first token: how long the user waits before the first word appears (retrieval + prompt building + model latency, including the model's hidden reasoning). |
| **p50 / p95** | The median, and the value 95% of requests beat. p95 shows the slow tail. With 10–24 samples, p95 is basically the second-slowest request, so I quote p50 and treat p95 as rough. |

### Retrieval quality

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@8 | MRR |
|---|---|---|---|---|---|
| Embedding-only (baseline) | 0.33 | 0.68 | 0.78 | 0.90 | 0.56 |
| Keyword (BM25) only | **0.56** | **0.83** | **0.89** | 0.95 | **0.79** |
| Hybrid (RRF, k=60), the default for `/chat` | 0.35 | 0.78 | 0.87 | **0.97** | 0.65 |
| Hybrid + Python function-aware chunks (not adopted) | 0.51 | 0.72 | 0.83 | 0.94–0.97 | 0.71–0.72 |

Setup: measured 2026-11-19 on the host with ChromaDB 0.5.23 and the same all-MiniLM-L6-v2 ONNX model as the container. The last row was measured in Docker, where retrieval isn't fully deterministic, so it is shown as a range over 3 runs.

**What I learned from it:**
- **The embedder misses most of each chunk.** all-MiniLM-L6-v2 reads at most 256 tokens. A 60-line chunk has a median of about 500–550 tokens, and 85–89% of chunks are over the limit, so the embeddings cover only about 46–51% of the indexed text. In every miss I checked, the answer sat below that cutoff.
- **My hypothesis was wrong.** I expected identifier questions to be the weak spot for embeddings, the usual argument for hybrid search. They were the *strongest* kind (R@1 0.83, R@8 1.00). The weak kinds were *definition* (MRR 0.32) and *flow* (R@1 0.08), where the answer is a few lines inside a long chunk.
- **Hybrid fixed the one total miss.** "Where are the exception classes for connect and read timeouts defined?" had no relevant chunk in the baseline's top 8. With hybrid, `exceptions.py` is in the top 8 (rank 8), and no question has nothing relevant in the top 8.
- **Hybrid is not best at ranking.** BM25 alone beats it on MRR (0.79 vs 0.65). RRF rewards chunks that both retrievers rank moderately well, and test files and `HISTORY.md` match every query, so they collect two votes. I did not tune the weights, because that would be tuning on my own test set.

### Answer quality (citations)

| Run | Citation validity | Notes |
|---|---|---|
| Baseline (prompt v1, embedding-only) | 100% (3/3) | Only 1 of 24 answers used the `[file:lines]` format at all, so mean confidence was 0.04. The other 23 named files in prose ("see `indexer.py` lines 51–110"), which can't be verified. A follow-up live run got 0/10 bracket citations with the same prompt. |
| Hybrid + prompt v2 (explicit format, example, no prose references) | not measured yet | The run hit Groq's daily token cap (200k tokens/day, about 100k for a full 24-question run). |

So the honest status: the **validator works** (64 tests pass), but the baseline **model did not follow the citation format**. Prompt v2 is the fix, and it is not yet measured.

Answer run: 24/24 answered, 0 errors, in Docker with Groq `openai/gpt-oss-20b` (`reasoning_effort="low"`), questions 35 s apart.

## Latency

Baseline measured 2026-11-18 on local Docker (Docker Desktop on a Mac, 8 CPUs / 4 GB for the VM), Groq `openai/gpt-oss-20b`. Index values are `this repo (61 chunks) / psf/requests (335 chunks)`, one run each. Chat values come from 10 questions (5 per repo), 8 chunks retrieved, timed server-side.

| Stage | Baseline |
|---|---|
| Index: clone | 1.4 s / 1.2 s |
| Index: chunk | 3 ms / 7 ms |
| Index: embed | 6.7 s / **43.5 s** |
| Index: summary (runs in parallel with embed) | 0.7 s / 0.8 s |
| Index: total | 8.1 s / 44.6 s |
| Chat: retrieval | p50 110 ms / p95 269 ms |
| Chat: TTFT | p50 774 ms / p95 907 ms |
| Chat: total | p50 1.45 s / p95 2.10 s |

**Why it looks like this:**
- **Embedding is about 97% of indexing time** (about 130 ms per chunk), and it's the only stage that grows with repo size. The summary is fully hidden behind it. I have not sped it up yet (see next steps), so there's no "after" column for embedding.
- **Chat latency is mostly the model**, not retrieval. In the 24-question answer run, TTFT minus retrieval ranged from 0.40 to 1.17 s, with a median TTFT of 879 ms (p95 2.06 s, driven by two slow retrievals).
- **Hybrid adds nothing measurable.** In Docker, over 3 × 12 questions: vector top 8 p50 561 ms; BM25 top 24 p50 3.5 ms; hybrid top 8 p50 555 ms. Retrieval time is the time to embed the question. That was about 460 ms in the container that day, much slower than the 110 ms retrieval in the baseline. It was the same for all methods and I have not found the cause.
- **I traded indexing speed for memory:** `UPSERT_BATCH_SIZE` 100 → 8 took indexing `psf/requests` from an OOM kill (peak 721 MiB) to a peak of about 353 MiB, and embed time from about 44 s to about 60 s.

## Running locally

Prerequisites: Docker, and a Groq API key (free) or a local Ollama.

```bash
cp backend/.env.example backend/.env
# For Groq:  set LLM_PROVIDER=groq and GROQ_API_KEY=...
# For Ollama: leave LLM_PROVIDER=ollama and run `ollama pull llama3.1:8b`
#             (on Docker Desktop set OLLAMA_BASE_URL=http://host.docker.internal:11434/v1)

docker compose up --build
```

- Frontend: http://localhost:8080 · Backend: http://localhost:8000 (`/health`, `/index`, `/chat`, `/repos`).
- Indexed repos are stored in the `chroma_data` volume and restored on restart.
- `VITE_API_URL` is a **build arg** for the frontend because Vite bakes it into the JS bundle and the *browser* calls it, so it is `http://localhost:8000`, not the compose service name.

Tests and evals (from `backend/`):

```bash
python -m pytest tests                       # 64 tests
python -m eval.eval_retrieval --method hybrid
python -m eval.eval_answers                  # needs the LLM; ~100k Groq tokens for a full run
```

Deploy: backend on Render (Docker, free plan, `render.yaml`), frontend on Vercel (static build). A GitHub Action pings `/health` every 10 minutes so the free instance doesn't sleep.

## Limitations and next steps

- **Single-process, so not horizontally scalable.** The index lives in one process's memory plus local disk (Render's free tier has none, so repos vanish on redeploy). To scale out: run ChromaDB as its own server, move indexing onto a job queue (Celery + Redis), and keep `/chat` stateless.
- **Memory:** each loaded repo adds roughly 10–25 MiB on top of the base, so only a handful of repos fit in 512 MB. Next step: evict the least recently used repos.
- **Embedding is slow** (130 ms per chunk). Ideas, none tried yet: skip fixtures and lockfiles, drop chunk overlap, profile ONNX threads.
- **Citation checking is mechanical.** It proves a citation points at retrieved code, not that the code supports the claim. The prompt-v2 fix is unmeasured.
- **Retrieval:** test files and changelogs distract. Next steps: down-weight them at query time, and add a cross-encoder reranker (it doesn't fit in 512 MB, so it needs a bigger instance).
- **Single-shot retrieval.** Each question retrieves once and answers once ("agent" in the name means a helper, not an autonomous loop). A possible next step is a ReAct loop that can search again, read a file or follow a call when the first retrieval isn't enough.
- **Free-tier limits:** Groq allows about 8k tokens/minute and 200k/day, which caps how many questions and eval runs fit in a day. If both retries fail on a 429, the stream ends without an error event (not fixed).
- **Eval size:** 24 questions over 2 Python/JS repos. It shows direction, not statistical significance.

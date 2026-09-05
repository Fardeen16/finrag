# Agentic RAG for Financial Analysis — Project Context

> Handoff document for AI coding assistants. Read this before making changes.

## What this project is

A production-grade agentic RAG system for financial analysis, answering questions about
Alphabet Inc. using its 10-K filing and structured financial data.

The distinguishing goal: **own the entire stack around an open-weights model**, not just
the application layer. Inference runs on a self-hosted vLLM server, not a hosted API.
Every layer from the GPU up is something the author configured and measured.

Target roles: AI Forward Deployed Engineer, ML/AI Engineer, GenAI Engineer.

---

## Hardware and environment

| Item | Value |
|---|---|
| Machine | Lenovo Legion 7, RTX 4070 Laptop |
| VRAM | 8 GB total, ~6.89 GB free (Windows desktop holds ~1.1 GB) |
| OS | Windows + WSL2 (Ubuntu 22.04) |
| Python | 3.12 (via deadsnakes PPA — **3.10 is not viable**, see below) |
| Project root | `~/finrag` (Linux filesystem — never work under `/mnt/c/`) |
| venv | `~/finrag/.venv` |

---

## Current state

### Working

**vLLM inference server.** Qwen2.5-7B-Instruct-AWQ served locally with an
OpenAI-compatible endpoint on port 8000.

Launch script at `~/finrag/serve.sh`:

```bash
#!/bin/bash
cd ~/finrag
source .venv/bin/activate
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.86 \
  --dtype half \
  --enforce-eager \
  --max-num-seqs 8 \
  --port 8000
```

**Every flag above is load-bearing.** Do not change them without reading the
"Environment constraints" section — most exist to work around a specific
documented failure.

**Load testing harness** at `~/finrag/loadtest.sh` — takes `[num_requests] [max_tokens]`,
reports per-request latency, wall time, and aggregate throughput.

**Existing agent** (built earlier, currently being migrated to the local model):

- **Phase 1 — Data.** Alphabet 10-K from SEC EDGAR (required a `User-Agent` header to
  avoid 403). HTML chunked with `unstructured`'s `chunk_by_title` to respect document
  hierarchy. Chunks enriched with summaries, keywords, and hypothetical questions.
  Embeddings in Qdrant; structured quarterly financials in SQLite.
- **Phase 2 — Four tools.** Librarian (semantic search + CrossEncoder re-ranking),
  SQL Analyst (numeric queries), Trend Analyst (QoQ/YoY via pandas `pct_change()`),
  Scout (DuckDuckGo live search).
- **Phase 3 — LangGraph supervisor.** Four nodes: Gatekeeper (query specificity check),
  Planner (tool selection, informed by past failures), Auditor (scores tool output
  confidence; routes back to Planner if score < 3), Synthesizer (final grounded answer).

### Verify before assuming

These were in progress at the time of writing. **Check the actual state of the repo
rather than trusting this section.**

- Migration from `ChatGoogleGenerativeAI` to the local vLLM endpoint
- Guided-decoding JSON schemas for the Planner and Auditor nodes
- The `--max-num-seqs 16` experiment (see "Open experiment" below)

---

## Measured performance

Benchmarks on the config above (`max_num_seqs 8`, identical short prompts, 150 max tokens):

| Concurrency | Wall time | Avg latency | Aggregate throughput |
|---|---|---|---|
| 1 | 2.62s | 2.60s | 40.0 tok/s |
| 4 | 2.75s | 2.73s | 152.8 tok/s |
| 16 | 4.99s | 3.81s | 336.5 tok/s |

From the vLLM startup log:

```
GPU KV cache size: 22,384 tokens
Maximum concurrency for 4,096 tokens per request: 5.46x
Model loading took 5.29 GiB
```

**Key finding.** Scaling is near-linear to N=4 (3.8× throughput for 5% higher latency),
because LLM decode is memory-bandwidth bound — weights are read once per decode step
regardless of batch size, so batching amortizes the expensive part.

At N=16 the latency distribution is **bimodal**: 8 requests at ~2.65s, 8 at ~4.98s, with
nothing in between. Two clean populations indicates queueing, not resource contention
(contention produces a smear, not a split). Root cause is the `--max-num-seqs 8`
scheduler cap, not KV cache exhaustion — 22,384 tokens is ample for 16 short sequences.

### Open experiment

Raise `--max-num-seqs` to 16 and re-run `./loadtest.sh 16`. Expected: the bimodal split
collapses to a single cluster, p99 drops from ~4.98s toward ~2.7s, throughput holds or
improves. **This has not been confirmed yet.** Do not state the improvement as fact
anywhere (resume, README, interview prep) until measured.

If raising the cap does *not* help, the next binding constraint is KV cache. Fixes in
order: `--kv-cache-dtype fp8` (roughly doubles token capacity), then reduce
`--max-model-len`.

---

## Environment constraints

Five distinct failures were resolved to reach a working server. Each flag or version
choice below exists for a reason — reverting one will reproduce the failure.

**1. Driver / CUDA runtime mismatch.** vLLM's bundled PyTorch is built against CUDA 13.x.
An older Windows driver exposing CUDA 12.9 caused `RuntimeError: The NVIDIA driver on
your system is too old`. Fixed by updating the *Windows* driver to R580+. The driver
lives on Windows; WSL borrows it. Never install a GPU driver inside WSL.

**2. `RuntimeError: UVA is not available`.** WSL2's GPU virtualization doesn't advertise
Unified Virtual Addressing, so vLLM's V2 model runner refuses to allocate its pinned host
buffer. Fixed by `export VLLM_WSL2_ENABLE_PIN_MEMORY=1`. Known upstream issue
(vllm-project/vllm#47387).

**3. Python 3.10 incompatibility.** `flashinfer` uses subscripted generics
(`array.array[int]`) in a runtime-evaluated annotation, which raises
`TypeError: 'type' object is not subscriptable` on 3.10. **Do not downgrade Python.**

**4. KV cache budget.** `gpu_memory_utilization` must cover weights (5.29 GiB) *plus*
engine overhead (activations, compile workspace, CUDA graphs). At 0.82 the remainder was
-0.04 GiB. `--enforce-eager` skips CUDA graph capture and frees several hundred MB —
this is the largest single saving, at some decode throughput cost.

**5. Missing `nvcc`.** FlashInfer JIT-compiles its sampling kernels at runtime and needs
the CUDA toolkit, which isn't installed (only the driver is). Fixed by
`export VLLM_USE_FLASHINFER_SAMPLER=0` to use the precompiled native sampler.
The `deep_gemm` import warning has the same root cause and is harmless.

> Installing the full CUDA toolkit in WSL would resolve 5 properly and is needed for any
> custom kernel work, but is not required for serving.

---

## Roadmap

Following a 14-step "build the whole production system around an open-weights model"
progression. Position: steps 1–3 done, 6–7 done, currently entering 4–5.

- [x] 1. Understand the inference path
- [x] 2. Build an inference server (vLLM, OpenAI-compatible)
- [x] 3. Learn why inference is fast or slow (batching, KV cache, quantization, scheduling)
- [ ] 4. **Build the backend** — users, auth, chat history, rate limits, error handling
- [ ] 5. **Build a chat product** — streaming frontend, conversation management
- [x] 6. RAG (chunking, retrieval, re-ranking)
- [x] 7. Tool calling and agent loop
- [~] 8. Evaluation and monitoring — retrieval precision exists; latency, tokens, GPU, cost do not
- [ ] 9. Reliability and safety — auth, retries, timeouts, data isolation, prompt injection
- [ ] 10. Deploy (AWS)
- [ ] 11. Load test under real conditions
- [ ] 12. Scale to multiple servers / multi-GPU
- [ ] 13. Automate the lifecycle (CI, versioning, eval gates, rollback)

---

## Next up: Phase C — the backend

Build in this order.

### 1. FastAPI wrapper

Single `POST /chat` endpoint taking a query and conversation ID, running the LangGraph
agent, returning the answer. Get the agent reachable over HTTP before adding anything else.

### 2. Stream agent state, not just tokens

**This is the project's differentiator — prioritise it.** Most chat backends stream text
deltas. This agent has a supervisor graph, so stream node transitions alongside tokens
using SSE (`StreamingResponse`) driven by LangGraph's `astream_events`:

```
event: node    data: {"node": "planner", "tools": ["sql_analyst"]}
event: node    data: {"node": "auditor", "score": 2, "action": "replan"}
event: node    data: {"node": "planner", "tools": ["librarian"]}
event: token   data: {"text": "Alphabet's 2024 revenue"}
```

Watching the Auditor reject a plan and the Planner retry is far more compelling in a demo
than watching text appear.

### 3. Persistence — DynamoDB

Partition key `user_id`, sort key `timestamp`. On-demand billing. Store the full turn
including which tools ran and the audit scores — this data feeds the Phase D eval work.

### 4. Auth — JWT

`/login` endpoint, passlib for hashing, python-jose for tokens, `user_id` in the claims.
Keep it minimal.

### 5. Multi-tenant isolation

**Every Qdrant query must filter on `user_id` taken from the JWT, enforced in the
retrieval function — never in the prompt.** A prompt instruction is not an access control.
Write a test proving user A's query cannot surface user B's documents.

### 6. Reliability

Per-user rate limit, hard wall-clock timeout on the whole agent loop, circuit breaker when
vLLM is unreachable. The agent has previously hit `GraphRecursionError`, so a ceiling on
graph execution is not optional.

### Latency expectation

One user query triggers multiple LLM calls: Gatekeeper → Planner → tools → Auditor
(possibly twice) → Synthesizer. At ~40 tok/s single-stream that is plausibly 15–30 seconds
end to end. **Instrument per-node latency from the start.** Finding that (for example) the
Auditor accounts for 60% of wall time, then caching or shrinking it, is a quantified
improvement with a real before/after.

---

## AWS deployment plan (Phase E)

Deliberately minimal. GPU inference stays off AWS — a `g5.xlarge` is ~\$1/hr and there is
no GPU free tier. vLLM will run on RunPod (or Kaggle's dual T4s for the multi-GPU work).

```
Browser
  └─ ECS Fargate: FastAPI + LangGraph  ──→ RunPod: vLLM (Qwen2.5-7B-AWQ)
       ├─ DynamoDB (chat history)
       ├─ Secrets Manager (RunPod URL, Qdrant key)
       └─ S3 (uploaded filings)
            └─ ObjectCreated → Lambda (container image) → chunk, embed, upsert → Qdrant
```

Two compute models, each matched to its workload: synchronous chat on a warm Fargate
container, bursty stateless ingestion on Lambda.

Cost and configuration notes:

- **No ALB** (~\$16/mo just to exist). Fargate task gets a public IP; security group
  restricted to the developer's IP.
- **No NAT Gateway** (~\$32/mo). Fargate in a *public* subnet. The Lambda stays out of
  the VPC entirely — attaching it to a VPC would cost it internet access and force a NAT.
- **Fargate 0.25 vCPU / 0.5 GB.** Set desired count to 0 when not in use
  (`aws ecs update-service --desired-count 0`) — it bills continuously otherwise.
- **Task role vs execution role.** Execution role pulls from ECR and writes CloudWatch
  logs. Task role is what application code uses (DynamoDB, Secrets Manager). Putting
  application permissions on the execution role is the most common Fargate mistake.
- Region `us-west-2`, fixed. Budget alerts at \$25 and \$50.
- Build Lambda images with `docker build --platform linux/amd64`.

**Fargate cannot reach `localhost` on the dev machine.** Either deploy vLLM to RunPod, or
run a Cloudflare Tunnel from WSL for local testing.

---

## Conventions

- Work in `~/finrag` on the Linux filesystem. Cross-filesystem I/O under `/mnt/c/` is
  dramatically slower.
- Run the vLLM server under `tmux` (session `vllm`) so it survives terminal closure.
  Note that `wsl --shutdown` kills tmux sessions.
- Keep a `NOTES.md` logging every failure, hypothesis, and measured result. This is
  interview material as much as documentation.
- Record benchmark numbers as tables with the exact config that produced them. A number
  without its config is not reproducible.
- **Resume/README consistency:** the project uses **Qdrant**, not FAISS. Qdrant is
  required — FAISS has no native payload filtering, which multi-tenant isolation depends on.
- The "30% Top-K retrieval precision improvement" claim needs a defensible eval set,
  query count, and judging method before it is defended in a technical interview.

## Method

When something is slow or broken: read the logs, look at the *distribution* rather than
the average, form a hypothesis about which layer is responsible, test it with a
measurement, and only then reach for outside help. Do not add a fix without a measurement
that confirms it worked.
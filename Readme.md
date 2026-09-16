# Financial Analyst Agent

An agentic RAG assistant for financial filings. A LangGraph supervisor plans tool calls, audits its own evidence, and replans when a tool fails. The UI streams that walk — not just the final answer.

**Live:** [finrag-zqup.onrender.com](https://finrag-zqup.onrender.com)

## Screenshots

<!-- Drop files into docs/screenshots/ then these will render. -->

**Chat:**
<img width="1108" height="668" alt="finrag_firstpage" src="https://github.com/user-attachments/assets/a44edc03-d200-49c2-99b9-936f69200ae0" />


**Resoning:**
<img width="1097" height="671" alt="finrag_oneprompt" src="https://github.com/user-attachments/assets/53b5077d-5672-49b3-88cd-2efe23ba4a96" />

## How it works

Gatekeeper → Planner → tools → Auditor → Synthesizer. Tools: Librarian (Qdrant), SQL Analyst, Trend Analyst, and Scout (web search).

Generation is **Qwen2.5-7B-Instruct-AWQ** on RunPod (vLLM). Embeddings are FastEmbed BGE-small on Render. Optional MiniLM rerank is a separate GPU worker. Vectors live in Qdrant Cloud.

## Stack

Vite / React · FastAPI · LangGraph · Qdrant · SQLite · Render · RunPod

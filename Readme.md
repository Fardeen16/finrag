# Financial Analyst Agent

An agentic RAG assistant for financial filings. A LangGraph supervisor plans tool calls, audits its own evidence, and replans when a tool fails. The UI streams that walk — not just the final answer.

**Live:** [finrag-zqup.onrender.com](https://finrag-zqup.onrender.com)

## Screenshots

<!-- Drop files into docs/screenshots/ then these will render. -->

![Chat](docs/screenshots/chat.png)

![Reasoning](docs/screenshots/trace.png)

## How it works

Gatekeeper → Planner → tools → Auditor → Synthesizer. Tools: Librarian (Qdrant), SQL Analyst, Trend Analyst, and Scout (web search).

Generation is **Qwen2.5-7B-Instruct-AWQ** on RunPod (vLLM). Embeddings are FastEmbed BGE-small on Render. Optional MiniLM rerank is a separate GPU worker. Vectors live in Qdrant Cloud.

## Stack

Vite / React · FastAPI · LangGraph · Qdrant · SQLite · Render · RunPod

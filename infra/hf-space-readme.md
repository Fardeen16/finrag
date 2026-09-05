---
title: FinRAG
emoji: 📈
colorFrom: gray
colorTo: green
sdk: docker
app_port: 8080
pinned: false
short_description: Agentic RAG over Alphabet 10-K filings with a self-hosted Qwen model
tags:
  - rag
  - finance
  - langgraph
  - fastapi
  - qwen
---

# FinRAG — Agentic financial RAG

Ask questions about **Alphabet Inc.** using its 10-K filing and structured financials.

A LangGraph supervisor (Gatekeeper → Planner → tools → Auditor → Synthesizer) retrieves from Qdrant, runs SQL, and answers with a live trace. Inference is a self-hosted **Qwen2.5-7B-Instruct-AWQ** worker that scales to zero — not a hosted chat API.

## Try these

- What was Alphabet's FY2024 revenue?
- What does the 10-K say about competition in cloud?
- Summarize Item 1A risk factors related to regulation.

## First question can be slow

The UI sleeps when idle. The GPU worker also scales to zero. The first question after a quiet stretch can take a few minutes while both wake up. Later questions on a warm worker are much faster.

Demo scope is FY2024 (and last-two-years) questions. Structured SQL covers 2022–2023; FY2024 figures come from the 10-K text.

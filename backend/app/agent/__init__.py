"""Agent package.

Keep this module lightweight. Scripts that only need embeddings or Qdrant
import `backend.app.agent.embeddings` / `.resources`; a side-effect import of
the LangGraph here used to pull the whole supervisor in and crash Windows
processes that then left the embedded Qdrant lock behind.
"""

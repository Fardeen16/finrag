"""Embedding providers.

The Qdrant collection is 384-d BGE (`BAAI/bge-small-en-v1.5`). Query and
passage vectors from different models are not comparable, so
`assert_dimension_matches` fails closed if the collection size drifts.

Local queries use FastEmbed (ONNX, ~70 MB) instead of sentence-transformers /
PyTorch. Loading torch on a 512 MB Render box is what froze Tool Executor.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

from ..config import get_settings


class FastEmbedEmbeddings(Embeddings):
    """BGE via ONNX. Same 384-d cosine space as the existing Qdrant index."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    @property
    def dimension(self) -> int:
        return int(len(next(self._model.query_embed("dimension probe"))))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(self._model.query_embed(text)).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    settings = get_settings()

    if settings.embedding_provider == "local":
        return FastEmbedEmbeddings(settings.local_embedding_model)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.require_google_key(),
    )


def embedding_dimension() -> int:
    embedder = get_embeddings()
    if isinstance(embedder, FastEmbedEmbeddings):
        return embedder.dimension
    return len(embedder.embed_query("dimension probe"))


def assert_dimension_matches(collection_dimension: int) -> None:
    actual = embedding_dimension()
    if actual != collection_dimension:
        settings = get_settings()
        raise RuntimeError(
            f"Embedding provider '{settings.embedding_provider}' produces {actual}-dim "
            f"vectors but collection '{settings.collection_name}' expects "
            f"{collection_dimension}. Re-index before querying — mismatched vector "
            f"spaces return plausible nonsense rather than failing."
        )

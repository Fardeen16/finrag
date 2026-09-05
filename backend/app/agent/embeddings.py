"""Embedding providers.

This is the one part of the Gemini migration that is not a config change. The
existing Qdrant collection holds 768-dimensional text-embedding-004 vectors.
Embeddings from a different model are not comparable to those, even at matching
dimensionality, so switching providers means recreating the collection.

The failure mode if you switch without re-indexing is nasty: Qdrant happily
returns nearest neighbours in the wrong vector space, so retrieval degrades to
noise while every component reports success. `assert_dimension_matches` exists
to turn that into an error at startup.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

from ..config import get_settings


# BGE retrieval models are trained asymmetrically: queries get an instruction
# prefix, passages do not. Omitting it costs real accuracy, and since the
# asymmetry is invisible at call time it is easy to get wrong in one place only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbeddings(Embeddings):
    """Local embeddings via sentence-transformers.

    Hand-rolled rather than pulling in langchain-huggingface, since
    sentence-transformers is already a dependency for the CrossEncoder reranker.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._query_instruction = (
            BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""
        )

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def _encode(self, texts: List[str]) -> List[List[float]]:
        # Normalised vectors make cosine distance equivalent to a dot product,
        # which is what the collection is configured for.
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([self._query_instruction + text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    settings = get_settings()

    if settings.embedding_provider == "local":
        return SentenceTransformerEmbeddings(settings.local_embedding_model)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.require_google_key(),
    )


def embedding_dimension() -> int:
    embedder = get_embeddings()
    if isinstance(embedder, SentenceTransformerEmbeddings):
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

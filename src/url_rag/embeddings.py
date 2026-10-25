"""Sentence embeddings (all-MiniLM-L6-v2, 384 dimensions). Model loads lazily on first use."""
import os

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = None


def get_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts):
    """Embed a list of strings -> numpy array of shape (n, 384), L2-normalized."""
    return get_embedder().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )


def embed_query(question: str):
    """Embed a single question -> numpy vector of shape (384,), L2-normalized."""
    return get_embedder().encode(
        question, convert_to_numpy=True, normalize_embeddings=True
    )

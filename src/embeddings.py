"""Sentence Transformers embedding model wrapper."""

from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts, batch_size: int = 32):
        """Convert a string or list of strings into embedding vectors."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

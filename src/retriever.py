"""Retrieve relevant document chunks from ChromaDB."""

class RAGRetriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        """Return the top matching chunks, ranked by cosine similarity."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not -1.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between -1 and 1.")

        collection = self.vector_store.collection
        available = collection.count()
        if available == 0:
            return []

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )

        retrieved_docs = []
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for rank, (doc_id, metadata, document, distance) in enumerate(
            zip(ids, metadatas, documents, distances), start=1
        ):
            similarity = 1.0 - float(distance)  # cosine distance -> similarity
            if similarity >= score_threshold:
                retrieved_docs.append(
                    {
                        "id": doc_id,
                        "document": document,
                        "metadata": metadata or {},
                        "distance": float(distance),
                        "similarity_score": similarity,
                        "rank": rank,
                    }
                )

        return retrieved_docs

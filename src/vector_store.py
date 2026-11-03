"""Persistent ChromaDB vector-store management."""

import uuid
from pathlib import Path

import chromadb


class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "pdf_documents",
    ):
        self.persist_directory = str(Path(persist_directory))
        self.collection_name = collection_name
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            # Chroma uses cosine distance so 1 - distance is cosine similarity.
            metadata={
                "hnsw:space": "cosine",
                "description": "Vector store for RAG document chunks",
            },
        )
        print(
            f"Initialized collection '{self.collection_name}' "
            f"with {self.collection.count()} item(s)."
        )

    @staticmethod
    def _clean_metadata(metadata):
        """Chroma metadata values must be simple scalar types."""
        cleaned = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[str(key)] = value
            else:
                cleaned[str(key)] = str(value)
        return cleaned or {"source": "unknown"}

    def add_documents(self, documents, embeddings, batch_size: int = 100):
        """Add document chunks and their embeddings to ChromaDB."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if not documents:
            return 0

        added = 0
        for start in range(0, len(documents), batch_size):
            doc_batch = documents[start : start + batch_size]
            embedding_batch = embeddings[start : start + batch_size]

            ids = [f"doc_{uuid.uuid4().hex}" for _ in doc_batch]
            contents = [doc.page_content for doc in doc_batch]
            metadatas = []

            for index, doc in enumerate(doc_batch, start=start):
                metadata = self._clean_metadata(doc.metadata)
                metadata["doc_index"] = index
                metadata["content_length"] = len(doc.page_content)
                metadatas.append(metadata)

            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=[embedding.tolist() for embedding in embedding_batch],
            )
            added += len(doc_batch)

        print(f"Added {added} chunk(s). Collection now has {self.collection.count()}.")
        return added

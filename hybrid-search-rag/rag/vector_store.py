from typing import List, Tuple, Optional
import numpy as np
from rag.chunker import Chunk


class SimpleVectorStore:
    """
    In-memory dense vector store with cosine similarity search.
    Stores Chunk objects alongside their dense embeddings.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def add_chunks(self, chunks: List[Chunk], vectors: np.ndarray):
        """
        Adds Chunk instances and their corresponding embedding vectors to the index.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of vectors ({len(vectors)})."
            )

        if len(chunks) == 0:
            return

        self.chunks.extend(chunks)

        if self.vectors.size == 0:
            self.vectors = np.array(vectors, dtype=np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def similarity_search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Performs cosine similarity search between query_vector and stored document vectors.

        Returns:
            List of (Chunk, similarity_score) sorted descending by score.
        """
        if len(self.chunks) == 0 or self.vectors.size == 0:
            return []

        q = np.array(query_vector, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_normed = q / q_norm

        # Compute L2 norms for stored vectors
        doc_norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)
        vectors_normed = self.vectors / doc_norms

        # Cosine similarities: dot product of normalized vectors
        scores = np.dot(vectors_normed, q_normed)

        # Get top-k indices
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Tuple[Chunk, float]] = [
            (self.chunks[idx], float(scores[idx])) for idx in top_indices
        ]
        return results

    def clear(self):
        """Clears all stored chunks and vectors."""
        self.chunks = []
        self.vectors = np.empty((0, 0), dtype=np.float32)

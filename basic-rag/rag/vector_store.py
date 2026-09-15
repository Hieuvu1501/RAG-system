from typing import List, Tuple
import numpy as np
from rag.chunker import Chunk


class SimpleVectorStore:
    """
    In-memory dense vector store with cosine similarity search.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def add_chunks(self, chunks: List[Chunk], vectors: np.ndarray):
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

    def similarity_search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if len(self.chunks) == 0 or self.vectors.size == 0:
            return []

        q = np.array(query_vector, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_normed = q / q_norm

        doc_norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)
        vectors_normed = self.vectors / doc_norms

        scores = np.dot(vectors_normed, q_normed)

        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        return [(self.chunks[idx], float(scores[idx])) for idx in top_indices]

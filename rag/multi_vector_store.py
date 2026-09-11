from typing import List, Tuple

import torch

from rag.chunker import Chunk


class MultiVectorStore:
    """
    In-memory multi-vector store with ColBERT-style MaxSim scoring.

    Each document is stored as a 2D tensor of token-level embeddings
    (shape [num_tokens, hidden_dim]). Retrieval computes MaxSim:

        score(Q, D) = sum over q_i in Q of max over d_j in D of cos_sim(q_i, d_j)

    This captures fine-grained token-level semantic matching, which is
    significantly more expressive than single-vector cosine similarity.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: List[torch.Tensor] = []  # List of [num_tokens, dim] tensors

    def add_chunks(self, chunks: List[Chunk], embeddings: List[torch.Tensor]):
        """
        Adds Chunk instances and their corresponding multi-vector embeddings.

        Args:
            chunks: List of Chunk objects.
            embeddings: List of 2D tensors, one per chunk.
                        Each tensor has shape [num_tokens, hidden_dim].
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})."
            )

        if len(chunks) == 0:
            return

        self.chunks.extend(chunks)
        # Detach and move to CPU for storage to free GPU memory
        for emb in embeddings:
            self.embeddings.append(emb.detach().cpu().float())

    def similarity_search(
        self, query_embedding: torch.Tensor, top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Performs MaxSim similarity search between a query multi-vector
        embedding and all stored document multi-vector embeddings.

        MaxSim(Q, D) = sum_{q_i in Q} max_{d_j in D} cos_sim(q_i, d_j)

        Args:
            query_embedding: 2D tensor of shape [num_query_tokens, hidden_dim].
            top_k: Number of top results to return.

        Returns:
            List of (Chunk, score) tuples sorted descending by MaxSim score.
        """
        if len(self.chunks) == 0 or len(self.embeddings) == 0:
            return []

        query = query_embedding.detach().cpu().float()

        # Normalize query token embeddings
        query_norm = torch.nn.functional.normalize(query, p=2, dim=-1)

        scores = []
        for doc_emb in self.embeddings:
            # Normalize document token embeddings
            doc_norm = torch.nn.functional.normalize(doc_emb, p=2, dim=-1)

            # Compute cosine similarity matrix: [num_query_tokens, num_doc_tokens]
            sim_matrix = torch.matmul(query_norm, doc_norm.T)

            # MaxSim: for each query token, take the max similarity across doc tokens
            # Then sum across all query tokens
            max_sim_per_query_token = sim_matrix.max(dim=-1).values
            maxsim_score = max_sim_per_query_token.sum().item()

            scores.append(maxsim_score)

        # Get top-k indices
        k = min(top_k, len(self.chunks))
        score_tensor = torch.tensor(scores)
        top_indices = torch.argsort(score_tensor, descending=True)[:k]

        results: List[Tuple[Chunk, float]] = [
            (self.chunks[idx.item()], scores[idx.item()])
            for idx in top_indices
        ]
        return results

    def clear(self):
        """Clears all stored chunks and embeddings."""
        self.chunks = []
        self.embeddings = []

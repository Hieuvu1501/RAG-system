from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from rag.chunker import Chunk


@dataclass
class FusedResult:
    chunk: Chunk
    rrf_score: float
    sparse_rank: int = -1
    sparse_score: float = 0.0
    dense_rank: int = -1
    dense_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "text": self.chunk.text,
            "rrf_score": round(self.rrf_score, 6),
            "sparse_rank": self.sparse_rank,
            "sparse_score": round(self.sparse_score, 4),
            "dense_rank": self.dense_rank,
            "dense_score": round(self.dense_score, 4),
        }


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Chunk, float]],
    sparse_results: List[Tuple[Chunk, float]],
    k: int = 60,
) -> List[FusedResult]:
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF):
        RRF_Score(doc) = sum( 1 / (k + rank_i(doc)) ) for each retriever list

    Args:
        dense_results: List of (Chunk, score) from dense vector retriever, ordered best-to-worst.
        sparse_results: List of (Chunk, score) from BM25 sparse retriever, ordered best-to-worst.
        k: Smoothing constant (default 60, standard in information retrieval literature).

    Returns:
        List of FusedResult sorted descending by RRF score.
    """
    fused_map: Dict[str, FusedResult] = {}

    # Process Dense Results (1-indexed rank)
    for rank_idx, (chunk, score) in enumerate(dense_results, start=1):
        if chunk.id not in fused_map:
            fused_map[chunk.id] = FusedResult(chunk=chunk, rrf_score=0.0)
        fused_map[chunk.id].rrf_score += 1.0 / (k + rank_idx)
        fused_map[chunk.id].dense_rank = rank_idx
        fused_map[chunk.id].dense_score = score

    # Process Sparse Results (1-indexed rank)
    for rank_idx, (chunk, score) in enumerate(sparse_results, start=1):
        if chunk.id not in fused_map:
            fused_map[chunk.id] = FusedResult(chunk=chunk, rrf_score=0.0)
        fused_map[chunk.id].rrf_score += 1.0 / (k + rank_idx)
        fused_map[chunk.id].sparse_rank = rank_idx
        fused_map[chunk.id].sparse_score = score

    # Sort candidates by combined RRF score descending
    sorted_fused = sorted(fused_map.values(), key=lambda x: x.rrf_score, reverse=True)
    return sorted_fused

"""
Hybrid Search RAG (Retrieval-Augmented Generation) package.
Integrates BM25 Sparse Search, Dense Vector Search, Reciprocal Rank Fusion (RRF),
Cross-Encoder Reranking, and LLM Synthesis.
"""

from rag.chunker import TextChunker, Chunk
from rag.embeddings import GeminiEmbedder
from rag.vector_store import SimpleVectorStore
from rag.bm25_retriever import BM25Retriever
from rag.fusion import reciprocal_rank_fusion
from rag.reranker import CrossEncoderReranker
from rag.pipeline import HybridRAGPipeline

__all__ = [
    "TextChunker",
    "Chunk",
    "GeminiEmbedder",
    "SimpleVectorStore",
    "BM25Retriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "HybridRAGPipeline",
]

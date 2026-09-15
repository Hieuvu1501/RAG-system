"""
Hybrid Search RAG (Retrieval-Augmented Generation) package.
Integrates BM25 Sparse Search, Dense Vector Search, Reciprocal Rank Fusion (RRF),
Cross-Encoder Reranking, LLM Synthesis, and TruLens RAG Triad Evaluation.

Supports two dense embedding backends:
- Vintern-Embedding-1B (local, multi-vector ColBERT-style)
- Gemini Embedding API (cloud, single-vector cosine similarity)
"""

from rag.chunker import TextChunker, Chunk
from rag.embeddings import GeminiEmbedder
from rag.vintern_embeddings import VinternEmbedder
from rag.vector_store import SimpleVectorStore
from rag.multi_vector_store import MultiVectorStore
from rag.bm25_retriever import BM25Retriever
from rag.fusion import reciprocal_rank_fusion
from rag.reranker import CrossEncoderReranker
from rag.evaluator import RAGTriadEvaluator, TriadResult
from rag.pipeline import HybridRAGPipeline

__all__ = [
    "TextChunker",
    "Chunk",
    "GeminiEmbedder",
    "VinternEmbedder",
    "SimpleVectorStore",
    "MultiVectorStore",
    "BM25Retriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "RAGTriadEvaluator",
    "TriadResult",
    "HybridRAGPipeline",
]

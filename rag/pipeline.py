import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from rag.chunker import TextChunker, Chunk
from rag.embeddings import GeminiEmbedder
from rag.vector_store import SimpleVectorStore
from rag.bm25_retriever import BM25Retriever
from rag.fusion import reciprocal_rank_fusion, FusedResult
from rag.reranker import CrossEncoderReranker, RerankedResult

try:
    from google import genai
except ImportError:
    genai = None


class HybridRAGPipeline:
    """
    End-to-end Hybrid Search RAG Pipeline:
    1. Ingestion: Document -> Chunks -> BM25 Index & Vector Store
    2. Dual Retrieval: BM25 Sparse Search + Dense Vector Search
    3. Rank Fusion: Reciprocal Rank Fusion (RRF)
    4. Cross-Encoder Reranking: Top-N Best Chunks
    5. LLM Synthesis: Gemini grounded response generation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        top_k_sparse: Optional[int] = None,
        top_k_dense: Optional[int] = None,
        rrf_k: Optional[int] = None,
        top_n_rerank: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key.strip() == "" or "your_gemini_api_key_here" in self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Please add your key to `.env`."
            )

        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gemini-3.6-flash")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

        self.top_k_sparse = top_k_sparse or int(os.getenv("TOP_K_SPARSE", "5"))
        self.top_k_dense = top_k_dense or int(os.getenv("TOP_K_DENSE", "5"))
        self.rrf_k = rrf_k or int(os.getenv("RRF_K", "60"))
        self.top_n_rerank = top_n_rerank or int(os.getenv("TOP_N_RERANK", "3"))

        # Initialize Sub-Components
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = GeminiEmbedder(api_key=self.api_key, model=self.embedding_model)
        self.vector_store = SimpleVectorStore()
        self.bm25_retriever = BM25Retriever()
        self.reranker = CrossEncoderReranker(api_key=self.api_key, model=self.llm_model)

        if genai is None:
            raise ImportError("Please install `google-genai` first.")
        self.client = genai.Client(api_key=self.api_key)
        self.indexed_chunks: List[Chunk] = []

    def index_document(self, file_path: str):
        """
        Reads, chunks, and indexes a text document into both BM25 and Vector stores.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = self.chunker.chunk_text(content, source=os.path.basename(file_path))
        if not chunks:
            raise ValueError(f"No text chunks could be extracted from {file_path}")

        self.indexed_chunks = chunks

        # 1. Index in BM25
        self.bm25_retriever.index_chunks(chunks)

        # 2. Embed and Index in Dense Vector Store
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_chunks(chunks, embeddings)

    def retrieve(
        self,
        query: str,
        top_k_sparse: Optional[int] = None,
        top_k_dense: Optional[int] = None,
        rrf_k: Optional[int] = None,
        top_n_rerank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes the complete hybrid retrieval flow:
        Sparse (BM25) + Dense (Vector) -> RRF Fusion -> Cross-Encoder Rerank
        """
        k_sparse = top_k_sparse or self.top_k_sparse
        k_dense = top_k_dense or self.top_k_dense
        k_fusion = rrf_k or self.rrf_k
        n_rerank = top_n_rerank or self.top_n_rerank

        # 1. BM25 Sparse Search
        sparse_results = self.bm25_retriever.search(query, top_k=k_sparse)

        # 2. Dense Vector Search
        query_vector = self.embedder.embed_query(query)
        dense_results = self.vector_store.similarity_search(query_vector, top_k=k_dense)

        # 3. Reciprocal Rank Fusion (RRF)
        fused_results: List[FusedResult] = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=k_fusion,
        )

        # 4. Cross-Encoder Reranker
        candidate_pairs = [(res.chunk, res.rrf_score) for res in fused_results]
        reranked_results: List[RerankedResult] = self.reranker.rerank(
            query=query,
            candidate_chunks=candidate_pairs,
            top_n=n_rerank,
        )

        return {
            "query": query,
            "sparse_results": sparse_results,
            "dense_results": dense_results,
            "fused_results": fused_results,
            "reranked_results": reranked_results,
            "top_chunks": [r.chunk for r in reranked_results],
        }

    def generate_answer(self, query: str, retrieved_chunks: List[Chunk]) -> str:
        """
        Generates grounded LLM response using top reranked chunks.
        """
        if not retrieved_chunks:
            return "I could not find any relevant context in the knowledge base to answer your question."

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_blocks.append(f"[Document Chunk {i} - ID: {chunk.id}]\n{chunk.text}")
        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are an accurate, helpful AI assistant. Answer the user's question based strictly on the provided context. If the answer cannot be found in the context, explicitly state that the information is not present. Do not fabricate facts.

Context:
{context_str}

User Question:
{query}

Grounded Answer:"""

        response = self.client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
        )
        text_parts = []
        if hasattr(response, "candidates") and response.candidates:
            for cand in response.candidates:
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if getattr(part, "text", None):
                            text_parts.append(part.text)
        return ("".join(text_parts).strip() if text_parts else response.text.strip())

    def query(
        self,
        question: str,
        top_k_sparse: Optional[int] = None,
        top_k_dense: Optional[int] = None,
        rrf_k: Optional[int] = None,
        top_n_rerank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        End-to-End Query method: Retrieval + Grounded LLM Generation.
        """
        retrieval_data = self.retrieve(
            query=question,
            top_k_sparse=top_k_sparse,
            top_k_dense=top_k_dense,
            rrf_k=rrf_k,
            top_n_rerank=top_n_rerank,
        )

        answer = self.generate_answer(question, retrieval_data["top_chunks"])

        return {
            **retrieval_data,
            "answer": answer,
        }

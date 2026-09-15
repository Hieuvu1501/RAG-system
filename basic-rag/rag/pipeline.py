import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from rag.chunker import TextChunker, Chunk
from rag.embeddings import GeminiEmbedder
from rag.vector_store import SimpleVectorStore

try:
    from google import genai
except ImportError:
    genai = None


class BasicRAGPipeline:
    """
    The simplest possible RAG pipeline:
    1. Ingestion: Document -> Chunks -> Dense Vector Store
    2. Retrieval: Top-K chunks by cosine similarity
    3. Generation: LLM answer grounded in the retrieved chunks

    No hybrid search, no reranking, no fusion — just enough to see how the
    parts of a RAG system fit together before adding any of that complexity.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        top_k: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key.strip() == "" or "your_gemini_api_key_here" in self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured. Please add your key to `.env`.")

        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gemini-3.6-flash")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.top_k = top_k or int(os.getenv("TOP_K", "3"))

        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = GeminiEmbedder(api_key=self.api_key, model=self.embedding_model)
        self.vector_store = SimpleVectorStore()

        if genai is None:
            raise ImportError("Please install `google-genai` first.")
        self.client = genai.Client(api_key=self.api_key)
        self.indexed_chunks: List[Chunk] = []

    def index_document(self, file_path: str):
        """Reads, chunks, embeds, and indexes a text document."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = self.chunker.chunk_text(content, source=os.path.basename(file_path))
        if not chunks:
            raise ValueError(f"No text chunks could be extracted from {file_path}")

        self.indexed_chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_chunks(chunks, embeddings)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[tuple]:
        """Embeds the query and returns the top-k most similar chunks."""
        k = top_k or self.top_k
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.similarity_search(query_vector, top_k=k)

    def generate_answer(self, query: str, retrieved_chunks: List[Chunk]) -> str:
        """Generates an answer grounded strictly in the retrieved chunks."""
        if not retrieved_chunks:
            return "I could not find any relevant context in the knowledge base to answer your question."

        context_blocks = [
            f"[Chunk {i} - ID: {chunk.id}]\n{chunk.text}"
            for i, chunk in enumerate(retrieved_chunks, start=1)
        ]
        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are an accurate, helpful AI assistant. Answer the user's question based strictly on the provided context. If the answer cannot be found in the context, explicitly state that the information is not present. Do not fabricate facts.

Context:
{context_str}

User Question:
{query}

Grounded Answer:"""

        response = self.client.models.generate_content(model=self.llm_model, contents=prompt)
        text_parts = []
        if hasattr(response, "candidates") and response.candidates:
            for cand in response.candidates:
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if getattr(part, "text", None):
                            text_parts.append(part.text)
        return ("".join(text_parts).strip() if text_parts else response.text.strip())

    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """End-to-end: retrieve top-k chunks, then generate a grounded answer."""
        results = self.retrieve(question, top_k=top_k)
        chunks = [c for c, _ in results]
        answer = self.generate_answer(question, chunks)
        return {
            "query": question,
            "retrieved": results,
            "answer": answer,
        }

#!/usr/bin/env python3
"""
LightRAG demo: graph-based Retrieval-Augmented Generation.

Unlike basic-rag (single dense retriever) and hybrid-search-rag (BM25L + dense +
RRF + reranker), LightRAG builds a *knowledge graph* out of the source document at
indexing time (entities + relationships extracted by an LLM), then answers queries
by combining graph traversal with vector search. Query "mode" controls how much of
the graph is used:

  - "naive"  : plain vector search over text chunks (closest to basic-rag)
  - "local"  : entity-centric retrieval (pulls in an entity's direct context)
  - "global" : relationship-centric retrieval (pulls in broader connected context)
  - "hybrid" : combines local + global (LightRAG's recommended default)

This demo wires LightRAG up to Gemini (via `google-genai`) for both the LLM and the
embedding model, since LightRAG's built-in helpers default to OpenAI.

Docs: https://github.com/HKUDS/LightRAG
"""
import asyncio
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv

load_dotenv()

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

try:
    from google import genai
except ImportError:
    genai = None

WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            print("\n[!] GEMINI_API_KEY is not configured in `.env`.")
            print("Copy `.env.example` to `.env` and set your key.\n")
            sys.exit(1)
        if genai is None:
            raise ImportError("Please install `google-genai` first (see requirements.txt).")
        _client = genai.Client(api_key=api_key)
    return _client


async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    """LLM callback LightRAG uses for entity/relationship extraction and query synthesis."""
    client = _get_client()

    parts = []
    if system_prompt:
        parts.append(system_prompt)
    for msg in history_messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append(prompt)
    full_prompt = "\n\n".join(parts)

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=LLM_MODEL,
        contents=full_prompt,
    )
    return response.text.strip()


async def embedding_func(texts):
    """Embedding callback LightRAG uses to embed chunks, entities, and queries."""
    import numpy as np

    client = _get_client()

    def _embed_all():
        vectors = []
        for text in texts:
            resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
            vectors.append(resp.embeddings[0].values)
        return vectors

    vectors = await asyncio.to_thread(_embed_all)
    return np.array(vectors, dtype=np.float32)


async def build_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=embedding_func,
        ),
        llm_model_func=llm_model_func,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="LightRAG graph-based RAG demo")
    parser.add_argument("--data", default="data/knowledge.txt", help="Document to index")
    parser.add_argument("--query", type=str, help="Question to ask (omit for interactive mode)")
    parser.add_argument(
        "--mode",
        choices=["naive", "local", "global", "hybrid"],
        default="hybrid",
        help="LightRAG query mode (default: hybrid)",
    )
    args = parser.parse_args()

    os.makedirs(WORKING_DIR, exist_ok=True)

    print("=" * 68)
    print("        🕸  LIGHTRAG DEMO (Graph-Based RAG)")
    print("=" * 68)

    rag = await build_rag()

    if not os.path.exists(args.data):
        print(f"\n[!] Document not found: {args.data}")
        sys.exit(1)

    print(f"Indexing {args.data} into the knowledge graph (this extracts entities & relations)...")
    with open(args.data, "r", encoding="utf-8") as f:
        content = f.read()
    await rag.ainsert(content)
    print("✅ Indexing complete.\n")

    async def ask(question: str):
        print(f"\nQuery ({args.mode}) > {question}")
        answer = await rag.aquery(question, param=QueryParam(mode=args.mode))
        print("\n💡 Answer:")
        print("-" * 68)
        print(answer)
        print("-" * 68)

    if args.query:
        await ask(args.query)
        return

    print("Type your questions below. Enter 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_prompt = input("\nQuery > ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break
            await ask(user_prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    asyncio.run(main())

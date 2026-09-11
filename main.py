#!/usr/bin/env python3
import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

from dotenv import load_dotenv

load_dotenv()

from rag.pipeline import HybridRAGPipeline


def print_banner():
    backend = os.getenv("EMBEDDING_BACKEND", "vintern").lower()
    if backend == "vintern":
        backend_label = "🧠 Vintern-Embedding-1B (Local Multi-Vector)"
    else:
        backend_label = "☁️  Gemini Embedding API (Cloud Single-Vector)"

    print("=" * 68)
    print("        🚀 HYBRID SEARCH RAG SYSTEM (BM25 + Dense + RRF + Reranker)")
    print(f"        Dense Backend: {backend_label}")
    print("=" * 68)


def check_api_key():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        print("\n[!] GEMINI_API_KEY is not configured in `.env`.")
        print("Please edit `/Users/bill/code/.env` and set your key:")
        print("    GEMINI_API_KEY=AIzaSy...\n")
        print("Get your API key free from Google AI Studio: https://aistudio.google.com/\n")
        sys.exit(1)


def display_results(result: dict, debug: bool = False):
    print("\n" + "=" * 68)
    print(f" Query: {result['query']}")
    print("=" * 68)

    if debug:
        print("\n[Stage 1: BM25 Sparse Search (Top Lexical Matches)]")
        for rank, (chunk, score) in enumerate(result["sparse_results"], start=1):
            snippet = chunk.text[:90].replace("\n", " ") + "..."
            print(f"  {rank}. [BM25: {score:.4f}] ({chunk.id}) {snippet}")

        print("\n[Stage 2: Dense Vector Search (Top Semantic Matches)]")
        for rank, (chunk, score) in enumerate(result["dense_results"], start=1):
            snippet = chunk.text[:90].replace("\n", " ") + "..."
            print(f"  {rank}. [Score: {score:.4f}] ({chunk.id}) {snippet}")

        print("\n[Stage 3: Reciprocal Rank Fusion (RRF Candidates)]")
        for rank, fused in enumerate(result["fused_results"][:5], start=1):
            info = fused.to_dict()
            print(f"  {rank}. [RRF: {info['rrf_score']:.6f}] ({info['chunk_id']}) "
                  f"(Dense Rank: #{info['dense_rank']}, Sparse Rank: #{info['sparse_rank']})")

        print("\n[Stage 4: Cross-Encoder Reranker (Top-N Candidates)]")
        for rank, r in enumerate(result["reranked_results"], start=1):
            snippet = r.chunk.text[:90].replace("\n", " ") + "..."
            print(f"  {rank}. [Rerank Score: {r.rerank_score:.2f} / 10.0] ({r.chunk.id}) {snippet}")

    print("\n💡 Synthesized Answer (Grounded in Top Chunks):")
    print("-" * 68)
    print(result["answer"])
    print("-" * 68)

    if not debug:
        print(f"\n[Sources used: {len(result['top_chunks'])} reranked chunks from hybrid fusion]")
        for i, c in enumerate(result["top_chunks"], start=1):
            print(f"  • [{i}] {c.id}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Search RAG System with BM25, Dense Vector, RRF, and Cross-Encoder Reranker"
    )
    parser.add_argument(
        "--data",
        default="data/knowledge.txt",
        help="Path to the document to index (default: data/knowledge.txt)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single question to query (if omitted, launches interactive prompt)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show intermediate retrieval details: BM25, Dense, RRF, and Cross-Encoder scores",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vintern", "gemini"],
        default=None,
        help="Override embedding backend (default: from EMBEDDING_BACKEND env var)",
    )
    parser.add_argument(
        "--top-k-sparse",
        type=int,
        default=None,
        help="Number of sparse chunks to retrieve",
    )
    parser.add_argument(
        "--top-k-dense",
        type=int,
        default=None,
        help="Number of dense chunks to retrieve",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top chunks after reranking for LLM synthesis",
    )

    args = parser.parse_args()

    # Allow CLI to override the embedding backend
    if args.backend:
        os.environ["EMBEDDING_BACKEND"] = args.backend

    print_banner()
    check_api_key()

    print(f"Loading and indexing knowledge document: {args.data} ...")
    try:
        pipeline = HybridRAGPipeline(
            top_k_sparse=args.top_k_sparse,
            top_k_dense=args.top_k_dense,
            top_n_rerank=args.top_n,
            embedding_backend=args.backend,
        )
        pipeline.index_document(args.data)
        print(f"✅ Indexed {len(pipeline.indexed_chunks)} chunks into BM25 & Vector Store successfully!\n")
    except Exception as e:
        print(f"\n[!] Failed to initialize or index: {e}")
        sys.exit(1)

    if args.query:
        result = pipeline.query(args.query)
        display_results(result, debug=args.debug)
        return

    # Interactive Loop
    print("Type your questions below. Enter 'exit' or 'quit' to stop.")
    print("Tip: add '--debug' when starting main.py to see retriever scores!\n")

    while True:
        try:
            user_prompt = input("\nQuery > ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break

            result = pipeline.query(user_prompt)
            display_results(result, debug=args.debug)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

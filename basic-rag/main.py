#!/usr/bin/env python3
import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

from dotenv import load_dotenv

load_dotenv()

from rag.pipeline import BasicRAGPipeline


def check_api_key():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        print("\n[!] GEMINI_API_KEY is not configured in `.env`.")
        print("Copy `.env.example` to `.env` and set your key:")
        print("    GEMINI_API_KEY=AIzaSy...\n")
        print("Get a free API key from Google AI Studio: https://aistudio.google.com/\n")
        sys.exit(1)


def display_results(result: dict, debug: bool = False):
    print("\n" + "=" * 68)
    print(f" Query: {result['query']}")
    print("=" * 68)

    if debug:
        print("\n[Retrieved Chunks (Cosine Similarity)]")
        for rank, (chunk, score) in enumerate(result["retrieved"], start=1):
            snippet = chunk.text[:90].replace("\n", " ") + "..."
            print(f"  {rank}. [score={score:.4f}] ({chunk.id}) {snippet}")

    print("\n💡 Answer:")
    print("-" * 68)
    print(result["answer"])
    print("-" * 68)
    print()


def main():
    parser = argparse.ArgumentParser(description="Basic RAG pipeline: chunk -> embed -> retrieve -> generate")
    parser.add_argument("--data", default="data/knowledge.txt", help="Path to the document to index")
    parser.add_argument("--query", type=str, help="Single question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--debug", action="store_true", help="Show retrieved chunks and similarity scores")
    args = parser.parse_args()

    check_api_key()

    print("=" * 68)
    print("        📖 BASIC RAG PIPELINE (Chunk -> Embed -> Retrieve -> Generate)")
    print("=" * 68)

    print(f"Loading and indexing document: {args.data} ...")
    try:
        pipeline = BasicRAGPipeline(top_k=args.top_k)
        pipeline.index_document(args.data)
        print(f"✅ Indexed {len(pipeline.indexed_chunks)} chunks into the vector store.\n")
    except Exception as e:
        print(f"\n[!] Failed to initialize or index: {e}")
        sys.exit(1)

    if args.query:
        result = pipeline.query(args.query)
        display_results(result, debug=args.debug)
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
            result = pipeline.query(user_prompt)
            display_results(result, debug=args.debug)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

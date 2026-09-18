#!/usr/bin/env python3
"""
ColVintern-1B-v1 demo: ColPali-style visual document retrieval for Vietnamese.

Unlike the other three demos in this repo, ColVintern doesn't retrieve over
plain text chunks - it embeds whole *document images* (scans, screenshots,
photographed pages) and Vietnamese/English text queries into the same
multi-vector token space, then scores every (query, image) pair with
ColBERT-style MaxSim ("late interaction") via the model's own
`processor.score_multi_vector`. This lets you search visual documents
(tables, forms, posters) directly, with no OCR step.

This script mirrors the official quickstart from the model card almost
exactly: https://huggingface.co/5CD-AI/ColVintern-1B-v1

Usage:
    python3 main.py --query "Phí giao hàng bao nhiêu ?"
    python3 main.py --query "Chuyện gì xảy ra với quốc lộ 5 TP Hải Phòng ?" --top-k 2
    python3 main.py --all                 # score every sample query against every image
    python3 main.py --all --query "Q1" --query "Q2"   # same, but with your own queries
    python3 main.py   # interactive mode over data/*.jpg
"""
import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
except ImportError:
    genai = None

MODEL_NAME = os.getenv("COLVINTERN_MODEL", "5CD-AI/ColVintern-1B-v1")
DATA_DIR = os.getenv("COLVINTERN_DATA_DIR", "data")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")

# The two example queries from the official model card - used by --all when
# no --query is given, so `python3 main.py --all` works out of the box.
SAMPLE_QUERIES = [
    "Chuyện gì xảy ra với quốc lộ 5 TP Hải Phòng ?",
    "Phí giao hàng bao nhiêu ?",
]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def load_model_and_processor(device: str):
    from transformers import AutoModel, AutoProcessor

    print(f"Loading {MODEL_NAME} on device '{device}' (first run downloads ~0.9B params)...")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    return model, processor


def move_to_device(batch: dict, device: str, dtype: torch.dtype) -> dict:
    """
    input_ids stay integer; pixel_values and attention_mask are cast to the
    model's compute dtype - this model consumes attention_mask as a
    multiplicative float weight, not a boolean/int mask (per the official
    quickstart, which calls `.cuda().bfloat16()` on attention_mask too).
    """
    moved = {}
    for key, val in batch.items():
        if not isinstance(val, torch.Tensor):
            moved[key] = val
        elif key == "input_ids":
            moved[key] = val.to(device=device)
        elif val.is_floating_point() or key == "attention_mask":
            moved[key] = val.to(device=device, dtype=dtype)
        else:
            moved[key] = val.to(device=device)
    return moved


def embed_images(model, processor, images, device, dtype):
    batch = move_to_device(processor.process_images(images), device, dtype)
    with torch.no_grad():
        return list(model(**batch))


def embed_queries(model, processor, queries, device, dtype):
    batch = move_to_device(processor.process_queries(queries), device, dtype)
    with torch.no_grad():
        return list(model(**batch))


def get_gemini_client():
    """
    Returns a genai.Client for answer generation, or None if no GEMINI_API_KEY
    is configured. ColVintern itself needs no API key (retrieval is fully
    local) - this is only used for the optional VQA step: asking an LLM to
    read the top-matched image and answer the question in words.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here" or genai is None:
        return None
    return genai.Client(api_key=api_key)


def generate_answer(client, image: Image.Image, question: str) -> str:
    """Asks Gemini to answer the question grounded in the matched document image."""
    prompt = (
        "You are an accurate, helpful assistant. Answer the question based strictly on "
        "the content visible in this document image. If the answer isn't visible in the "
        "image, say so explicitly rather than guessing.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    response = client.models.generate_content(model=LLM_MODEL, contents=[image, prompt])
    return response.text.strip()


def print_score_table(queries, image_paths, scores: torch.Tensor):
    """Prints a query-x-image score matrix, bolding the top score per query."""
    names = [os.path.basename(p) for p in image_paths]
    col_width = max(8, *(len(n) for n in names))
    query_width = max(len(q) for q in queries)

    header = f"{'Query':<{query_width}} | " + " | ".join(f"{n:>{col_width}}" for n in names)
    print(header)
    print("-" * len(header))
    for i, q in enumerate(queries):
        row_scores = scores[i]
        best_idx = int(torch.argmax(row_scores))
        cells = []
        for j, s in enumerate(row_scores.tolist()):
            cell = f"{s:.4f}"
            cells.append(f"{cell + ' *' if j == best_idx else cell:>{col_width}}")
        print(f"{q:<{query_width}} | " + " | ".join(cells))
    print("\n(* = best-matching image for that query)")


def main():
    parser = argparse.ArgumentParser(description="ColVintern-1B-v1 visual document retrieval demo")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory of document images to index")
    parser.add_argument(
        "--query",
        type=str,
        action="append",
        help="Question to ask. Repeat --query for multiple (used with --all); omit for interactive mode.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Batch mode: score every query (--query, repeatable; defaults to the model card's sample "
        "queries if none given) against every image in --data-dir, and print the full score table.",
    )
    parser.add_argument("--top-k", type=int, default=1, help="Number of top-matching images to show (non --all mode)")
    parser.add_argument("--device", default=os.getenv("COLVINTERN_DEVICE", "auto"), help="cuda | mps | cpu | auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    image_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.jpg"))) + sorted(
        glob.glob(os.path.join(args.data_dir, "*.png"))
    )
    if not image_paths:
        print(f"\n[!] No .jpg/.png images found in {args.data_dir}/")
        sys.exit(1)

    print("=" * 68)
    print("        🖼  COLVINTERN-1B-v1 DEMO (Visual Document Retrieval)")
    print("=" * 68)
    print(f"Indexing {len(image_paths)} image(s): {', '.join(os.path.basename(p) for p in image_paths)}")

    model, processor = load_model_and_processor(device)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    image_embeddings = embed_images(model, processor, images, device, dtype)
    print("✅ Images indexed.\n")

    gemini_client = get_gemini_client()
    if gemini_client is None:
        print(
            "[i] No GEMINI_API_KEY configured - showing retrieval results only.\n"
            "    Set GEMINI_API_KEY in .env to also get a generated answer for each query.\n"
        )

    def ask(question: str, top_k: int):
        query_embeddings = embed_queries(model, processor, [question], device, dtype)
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)[0]
        k = min(top_k, len(image_paths))
        top_indices = torch.topk(scores, k=k).indices.tolist()

        print(f"\nQuery > {question}")
        print("-" * 68)
        for rank, idx in enumerate(top_indices, start=1):
            print(f"  {rank}. [score={scores[idx].item():.4f}] {os.path.basename(image_paths[idx])}")
        print("-" * 68)

        if gemini_client is not None:
            best_idx = top_indices[0]
            answer = generate_answer(gemini_client, images[best_idx], question)
            print(f"\n💡 Answer (grounded in {os.path.basename(image_paths[best_idx])}):")
            print("-" * 68)
            print(answer)
            print("-" * 68)

    if args.all:
        queries = args.query or SAMPLE_QUERIES
        query_embeddings = embed_queries(model, processor, queries, device, dtype)
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)
        print(f"Scoring {len(queries)} quer{'y' if len(queries) == 1 else 'ies'} "
              f"against {len(image_paths)} image(s):\n")
        print_score_table(queries, image_paths, scores)
        return

    if args.query:
        for question in args.query:
            ask(question, args.top_k)
        return

    print("Type a question about the indexed images. Enter 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_prompt = input("\nQuery > ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break
            ask(user_prompt, args.top_k)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

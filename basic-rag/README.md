# 📖 Basic RAG Pipeline

> Part of the [RAG demos](../README.md) collection — see also [hybrid-search-rag](../hybrid-search-rag) and [lightrag](../lightrag).

The simplest possible Retrieval-Augmented Generation pipeline: one retriever, one similarity metric, no fusion, no reranking. Read this first, then compare it against `hybrid-search-rag` to see exactly what each added technique (BM25L, RRF, ColBERT-style reranking, TruLens evaluation) buys you.

## 🏛 Pipeline

```
   Document
      │
      ▼
 [Chunker]  (paragraph + sliding window)
      │
      ▼
 [Gemini Embedder]  (gemini-embedding-001)
      │
      ▼
 [Vector Store]  (cosine similarity)
      │
      ▼
   Query ──► embed ──► Top-K similarity search ──► [Gemini LLM] ──► Grounded Answer
```

## 📂 Project Structure

```
basic-rag/
├── .env.example        # Environment configuration template
├── requirements.txt    # google-genai, numpy, python-dotenv
├── data/
│   └── knowledge.txt   # Sample knowledge base (Project Apollo & AGC-1969)
├── rag/
│   ├── chunker.py       # TextChunker (paragraph + sliding window)
│   ├── embeddings.py    # GeminiEmbedder (single-vector dense embeddings)
│   ├── vector_store.py  # SimpleVectorStore (cosine similarity)
│   └── pipeline.py      # BasicRAGPipeline (index -> retrieve -> generate)
├── tests_core.py        # Unit tests for chunking & vector search (no API calls)
├── main.py              # CLI runner
└── README.md
```

## ⚡ Quickstart

```bash
cd basic-rag
cp .env.example .env        # then add your GEMINI_API_KEY
pip install -r requirements.txt
.venv/bin/python -m unittest tests_core.py     # sanity check, no API calls
.venv/bin/python main.py --query "What were the specifications of AGC-1969?" --debug
```

Or run interactively:
```bash
.venv/bin/python main.py
```

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key ([Get one free](https://aistudio.google.com/)) |
| `LLM_MODEL` | `gemini-3.6-flash` | Model used for answer generation |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Model used for dense embeddings |
| `TOP_K` | `3` | Number of chunks retrieved per query |

## What this demo intentionally leaves out

- No sparse/lexical retrieval (BM25) — dense-only, so exact keyword/ID matches can be missed.
- No rank fusion — a single retriever means nothing to fuse.
- No reranking — chunks are used in raw similarity order.
- No answer-quality evaluation.

Each of those is added, one at a time, in [`hybrid-search-rag`](../hybrid-search-rag).

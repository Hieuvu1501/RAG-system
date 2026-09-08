# 🚀 Hybrid Search RAG System

A modular, production-ready implementation of a **Hybrid Search Retrieval-Augmented Generation (RAG)** pipeline combining **BM25 Sparse Retrieval**, **Dense Vector Retrieval**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, and **Gemini LLM Synthesis**.

---

## 🏛 Architecture Diagram

```
                     User Query
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
    [BM25 Sparse Retriever]   [Dense Vector Retriever]
       (Inverted Index)        (Vector DB / Embedding)
             │                       │
             ▼                       ▼
    Top-K Lexical Chunks      Top-K Semantic Chunks
             │                       │
             └───────────┬───────────┘
                         ▼
              [Rank Fusion (ex: RRF)]
                         │
                         ▼
              [Cross-Encoder Reranker]
                         │
                         ▼
                 Top-N Best Chunks
                         │
                         ▼
                  [LLM Synthesis]
```

### Why Hybrid Search?
- **Dense Vector Search** captures conceptual similarity and semantic nuances, but can struggle with exact serial numbers, technical acronyms, or rare keywords.
- **BM25 Sparse Search** excels at exact keyword matching, technical codes (`AGC-1969`, `SA-506`), and proper nouns.
- **Reciprocal Rank Fusion (RRF)** mathematically normalizes and merges ranked results from both paradigms without arbitrary score scaling.
- **Cross-Encoder Reranker** scores full query-document cross-attention pairs to select the true top-N most informative passages.
- **LLM Synthesis** grounds the final answer strictly on the reranked context with citations.

---

## 📂 Project Structure

```
/Users/bill/code/
├── .env.example          # Environment variables template
├── .env                  # Active environment file (put your GEMINI_API_KEY here)
├── requirements.txt      # Python dependencies
├── data/
│   └── knowledge.txt     # Sample knowledge base with technical codes & concepts
├── rag/
│   ├── __init__.py       # Package exports
│   ├── chunker.py        # TextChunker (paragraph + sliding window)
│   ├── embeddings.py     # GeminiEmbedder (gemini-embedding-001)
│   ├── vector_store.py   # SimpleVectorStore (cosine similarity dense index)
│   ├── bm25_retriever.py # BM25Retriever (BM25Okapi sparse lexical index)
│   ├── fusion.py         # Reciprocal Rank Fusion (RRF) algorithm
│   ├── reranker.py       # CrossEncoderReranker (relevance scoring)
│   └── pipeline.py       # HybridRAGPipeline (end-to-end orchestrator)
├── tests_core.py         # Unit tests (Chunker, BM25, Cosine Similarity, RRF)
├── main.py               # Interactive CLI runner
└── README.md             # Documentation
```

---

## ⚡ Quickstart

### 1. Configure `.env`
Open [.env](.env) and insert your Gemini API Key (get one free at [Google AI Studio](https://aistudio.google.com/)):

```env
GEMINI_API_KEY=AIzaSy...
LLM_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-001
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Core Verification Tests
Run the standalone tests to verify BM25, Vector Search, and RRF logic (no API calls required):
```bash
python3 -m unittest tests_core.py
```

### 4. Ask Questions!

**Interactive Mode:**
```bash
python3 main.py
```

**Single Query with Debug View:**
```bash
python3 main.py --query "What were the specifications of AGC-1969?" --debug
```

**Query with Custom Data:**
```bash
python3 main.py --data path/to/your/document.txt --query "Your question here"
```

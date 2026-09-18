# 🚀 Hybrid Search RAG System

> Part of the [RAG demos](../README.md) collection — see also [basic-rag](../basic-rag) and [lightrag](../lightrag).

A modular, production-grade implementation of a **Hybrid Search Retrieval-Augmented Generation (RAG)** pipeline combining **BM25L Sparse Retrieval**, **Dense Vector Retrieval** (with a ColBERT-style multi-vector MaxSim backend), **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, and **Gemini LLM Synthesis**.

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
                         │
                         ▼
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  Context Relevance  Groundedness  Answer Relevance
        [TruLens RAG Triad Evaluation]
```

### Why Hybrid Search?
- **Dense Vector Search** captures conceptual similarity and semantic meaning, but can struggle with exact serial numbers, technical acronyms, or rare keywords. The `vintern` backend stores per-token multi-vector embeddings and scores them with **ColBERT-style MaxSim** late interaction instead of single-vector cosine similarity (see [`rag/multi_vector_store.py`](rag/multi_vector_store.py)).
- **BM25L Sparse Search** excels at exact keyword matching, technical codes (`AGC-1969`, `SA-506`), numbers, and proper nouns. Unlike classic BM25Okapi, **BM25L** (Lv & Zhai, 2011) adds a length-normalized pseudo-tf and a `delta` lower bound so long-but-relevant chunks aren't unfairly penalized.
- **Reciprocal Rank Fusion (RRF)** merges and normalizes ranked results from both sparse and dense retrievers without arbitrary score scaling:
  $$RRF\_Score(d) = \sum_{r \in \{\text{Dense}, \text{Sparse}\}} \frac{1}{k + \text{rank}_r(d)}$$
- **Cross-Encoder Reranker** scores full query-document cross-attention pairs (0.00 – 10.00 scale) to select the true top-$N$ most informative passages.
- **LLM Synthesis** generates the final answer grounded strictly in the top reranked context with zero hallucinations.
- **TruLens RAG Triad** evaluates every answer for Context Relevance, Groundedness, and Answer Relevance (LLM-as-a-judge).

---

## 📂 Project Structure

```
hybrid-search-rag/
├── .env.example          # Environment configuration template
├── .env                  # Active environment file (GEMINI_API_KEY & model options)
├── requirements.txt      # Project dependencies (google-genai, rank-bm25, numpy, python-dotenv)
├── data/
│   ├── knowledge.txt     # Technical knowledge base on Project Apollo & AGC-1969
│   └── example1.txt      # Googlecar vehicle manual dataset
├── rag/
│   ├── __init__.py       # Package exports
│   ├── chunker.py        # TextChunker (paragraph, sliding window & code assignments)
│   ├── embeddings.py     # GeminiEmbedder (gemini-embedding-001 via google-genai)
│   ├── vintern_embeddings.py # VinternEmbedder (local multi-vector / ColBERT-style tokens)
│   ├── vector_store.py   # SimpleVectorStore (cosine similarity dense index)
│   ├── multi_vector_store.py # MultiVectorStore (ColBERT-style MaxSim late interaction)
│   ├── bm25_retriever.py # BM25Retriever (BM25L sparse lexical index)
│   ├── fusion.py         # Reciprocal Rank Fusion (RRF) algorithm
│   ├── reranker.py       # CrossEncoderReranker (relevance scoring & top-N filtering)
│   ├── evaluator.py      # RAGTriadEvaluator (TruLens-style Context/Groundedness/Answer triad)
│   └── pipeline.py       # HybridRAGPipeline (end-to-end orchestrator)
├── tests_core.py         # Unit tests (Chunker, BM25L, Cosine Similarity, RRF)
├── main.py               # Interactive CLI runner with debug output
└── README.md             # Documentation
```

---

## ⚙️ Configuration (`.env`)

Configure your settings in `.env` (or copy from `.env.example`):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key ([Get one from Google AI Studio](https://aistudio.google.com/)) |
| `LLM_MODEL` | `gemini-3.6-flash` | Gemini model for cross-encoder reranking & synthesis |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model for dense vector search |
| `TOP_K_SPARSE` | `5` | Number of lexical chunks to retrieve via BM25 |
| `TOP_K_DENSE` | `5` | Number of semantic chunks to retrieve via Vector Store |
| `RRF_K` | `60` | Reciprocal Rank Fusion smoothing constant |
| `TOP_N_RERANK` | `3` | Number of best chunks passed to LLM synthesis |
| `ENABLE_TRIAD_EVAL` | `true` | Enable/disable TruLens RAG Triad evaluation after each query |
| `TRIAD_EVAL_MODEL` | `gemini-3.6-flash` | Gemini model used as LLM-as-a-Judge for triad scoring |

---

## ⚡ Quickstart

All commands below assume you are inside this folder:
```bash
cd hybrid-search-rag
```

### 1. Configure `.env`
Copy [.env.example](.env.example) to `.env` and add your API key:
```env
GEMINI_API_KEY=AIzaSy...
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Core Verification Tests
Run the standalone unit tests (verifies BM25, Vector Search, and RRF logic with zero API calls):
```bash
.venv/bin/python -m unittest tests_core.py
```

---

## 💬 Usage Examples

### 1. Vehicle Manual Dataset (`data/example1.txt`)

**Ask a question with stage-by-stage debug view:**
```bash
.venv/bin/python main.py --data data/example1.txt --query "How do I defrost the windshield in my Googlecar?" --debug
```

**Ask about touchscreen navigation & music:**
```bash
.venv/bin/python main.py --data data/example1.txt --query "How do I play music or get directions using the touchscreen?" --debug
```

**Ask about gear shifting in slippery conditions:**
```bash
.venv/bin/python main.py --data data/example1.txt --query "What gear position should I use for driving in snow?" --debug
```

**Interactive chat mode on `data/example1.txt`:**
```bash
.venv/bin/python main.py --data data/example1.txt
```

---

### 2. Apollo Knowledge Base (`data/knowledge.txt`)

**Query exact technical specifications:**
```bash
.venv/bin/python main.py --query "What were the specifications of AGC-1969?" --debug
```

**Interactive chat session:**
```bash
.venv/bin/python main.py
```

---

### 3. Custom Retrieval Parameters

Tune retrieval depths directly via CLI flags:
```bash
.venv/bin/python main.py \
  --data data/knowledge.txt \
  --query "Who walked on the Moon during Apollo 11?" \
  --top-k-sparse 8 \
  --top-k-dense 8 \
  --top-n 4 \
  --debug
```

---

## 🔍 Understanding the Pipeline Stages (`--debug`)

When running with `--debug`, the CLI outputs each stage of the pipeline:

1. **Stage 1: BM25 Sparse Search** — Displays the top lexical matches scored by BM25L.
2. **Stage 2: Dense Vector Search** — Displays the top semantic matches scored by Cosine Similarity.
3. **Stage 3: Reciprocal Rank Fusion (RRF)** — Shows the fused score and individual retriever ranks for each candidate chunk.
4. **Stage 4: Cross-Encoder Reranker** — Evaluates deep query-document relevance (scored out of 10.0) and re-orders candidates.
5. **Stage 5: Grounded Answer** — Synthesizes a factual response based strictly on the top reranked chunks.
6. **📊 TruLens RAG Triad** — Automatically evaluates the response quality with three metrics.

---

## 📊 TruLens RAG Triad Evaluation

Every response is automatically evaluated using the **TruLens RAG Triad** — an LLM-as-a-Judge methodology that detects hallucinations and assesses quality across three dimensions:

| Metric | What it measures | Failure mode |
| :--- | :--- | :--- |
| **Context Relevance** | Are the retrieved chunks relevant to the query? | Retrieval noise / off-topic chunks |
| **Groundedness** | Is every claim in the answer supported by the context? | LLM hallucination |
| **Answer Relevance** | Does the answer directly address the user's question? | Vague or off-topic generation |

Scores are in **[0.0, 1.0]**. A **Composite Score** is the mean of all three.

### Example Output

```
📊 TruLens RAG Triad Evaluation
──────────────────────────────────────────────────
  Context Relevance:  0.87  █████████░  ✅
  Groundedness:       0.95  █████████▌  ✅
  Answer Relevance:   0.90  █████████   ✅
  ──────────────────────────────────────────────
  Composite Score:    0.91  █████████░
──────────────────────────────────────────────────
```

### Interpretation Guide

| Score Range | Status | Meaning |
| :--- | :--- | :--- |
| ≥ 0.80 | ✅ Pass | High quality, low hallucination risk |
| 0.50 – 0.79 | ⚠️ Warning | Investigate retrieval or generation quality |
| < 0.50 | ❌ Fail | High risk of hallucination or irrelevant output |

### Disabling Evaluation

To skip the triad evaluation for faster responses, set in `.env`:
```env
ENABLE_TRIAD_EVAL=false
```

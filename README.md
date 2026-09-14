# 🚀 Hybrid Search RAG System

A modular, production-grade implementation of a **Hybrid Search Retrieval-Augmented Generation (RAG)** pipeline combining **BM25 Sparse Retrieval**, **Dense Vector Retrieval**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, and **Gemini LLM Synthesis**.

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
- **Dense Vector Search** captures conceptual similarity and semantic meaning, but can struggle with exact serial numbers, technical acronyms, or rare keywords.
- **BM25 Sparse Search** excels at exact keyword matching, technical codes (`AGC-1969`, `SA-506`), numbers, and proper nouns.
- **Reciprocal Rank Fusion (RRF)** merges and normalizes ranked results from both sparse and dense retrievers without arbitrary score scaling:
  $$RRF\_Score(d) = \sum_{r \in \{\text{Dense}, \text{Sparse}\}} \frac{1}{k + \text{rank}_r(d)}$$
- **Cross-Encoder Reranker** scores full query-document cross-attention pairs (0.00 – 10.00 scale) to select the true top-$N$ most informative passages.
- **LLM Synthesis** generates the final answer grounded strictly in the top reranked context with zero hallucinations.

---

## 📂 Project Structure

```
/Users/bill/code/
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
│   ├── vector_store.py   # SimpleVectorStore (cosine similarity dense index)
│   ├── bm25_retriever.py # BM25Retriever (BM25Okapi sparse lexical index)
│   ├── fusion.py         # Reciprocal Rank Fusion (RRF) algorithm
│   ├── reranker.py       # CrossEncoderReranker (relevance scoring & top-N filtering)
│   └── pipeline.py       # HybridRAGPipeline (end-to-end orchestrator)
├── tests_core.py         # Unit tests (Chunker, BM25, Cosine Similarity, RRF)
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

---

## ⚡ Quickstart

### 1. Configure `.env`
Open [.env](.env) and add your API key:
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
python3 -m unittest tests_core.py
```

---

## 💬 Usage Examples

### 1. Vehicle Manual Dataset (`data/example1.txt`)

**Ask a question with stage-by-stage debug view:**
```bash
python3 main.py --data data/example1.txt --query "How do I defrost the windshield in my Googlecar?" --debug
```

**Ask about touchscreen navigation & music:**
```bash
python3 main.py --data data/example1.txt --query "How do I play music or get directions using the touchscreen?" --debug
```

**Ask about gear shifting in slippery conditions:**
```bash
python3 main.py --data data/example1.txt --query "What gear position should I use for driving in snow?" --debug
```

**Interactive chat mode on `data/example1.txt`:**
```bash
python3 main.py --data data/example1.txt
```

---

### 2. Apollo Knowledge Base (`data/knowledge.txt`)

**Query exact technical specifications:**
```bash
python3 main.py --query "What were the specifications of AGC-1969?" --debug
```

**Interactive chat session:**
```bash
python3 main.py
```

---

### 3. Custom Retrieval Parameters

Tune retrieval depths directly via CLI flags:
```bash
python3 main.py \
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

1. **Stage 1: BM25 Sparse Search** — Displays the top lexical matches scored by BM25Okapi.
2. **Stage 2: Dense Vector Search** — Displays the top semantic matches scored by Cosine Similarity.
3. **Stage 3: Reciprocal Rank Fusion (RRF)** — Shows the fused score and individual retriever ranks for each candidate chunk.
4. **Stage 4: Cross-Encoder Reranker** — Evaluates deep query-document relevance (scored out of 10.0) and re-orders candidates.
5. **Stage 5: Grounded Answer** — Synthesizes a factual response based strictly on the top reranked chunks.

# 🕸 LightRAG Demo (Graph-Based RAG)

> Part of the [RAG demos](../README.md) collection — see also [basic-rag](../basic-rag) and [hybrid-search-rag](../hybrid-search-rag).

[LightRAG](https://github.com/HKUDS/LightRAG) takes a different approach from the other two demos in this repo: instead of only chunking + embedding text, it uses an LLM to extract **entities and relationships** from the document at indexing time and builds a **knowledge graph**. Queries are then answered by combining graph traversal with vector search, which tends to do better on questions that span multiple chunks or require connecting facts (e.g. "how are X and Y related?").

This demo wires LightRAG up to **Gemini** (via `google-genai`) for both the LLM and embedding calls, since LightRAG's built-in helper functions default to OpenAI.

## 🏛 How it differs from the other two demos

| | [basic-rag](../basic-rag) | [hybrid-search-rag](../hybrid-search-rag) | lightrag (this folder) |
| :--- | :--- | :--- | :--- |
| Indexing | Chunk + embed | Chunk + embed + BM25L index | Chunk + embed + **LLM-extracted entity/relationship graph** |
| Retrieval | Dense top-K | BM25L + Dense → RRF → Cross-encoder rerank | Vector search + **graph traversal** (`naive`/`local`/`global`/`hybrid` modes) |
| Best for | Learning the RAG basics | Exact keywords/IDs + semantic recall in one pass | Multi-hop questions connecting entities/relations |
| Evaluation | None | TruLens RAG Triad | None (bring your own) |

## Query modes

- `naive` — plain vector search over text chunks (closest to basic-rag).
- `local` — entity-centric: pulls in an entity's direct neighborhood.
- `global` — relationship-centric: pulls in broader connected context.
- `hybrid` — combines local + global (LightRAG's recommended default).

## 📂 Project Structure

```
lightrag/
├── .env.example      # Environment configuration template
├── requirements.txt  # lightrag-hku, google-genai, numpy, python-dotenv
├── data/
│   └── knowledge.txt # Sample knowledge base (Project Apollo & AGC-1969)
├── main.py           # Builds the graph, indexes a document, and answers queries
└── README.md
```

`main.py` creates a `rag_storage/` directory on first run — that's LightRAG's on-disk knowledge graph, vector index, and KV cache. It's gitignored; delete it to force a clean re-index.

## ⚡ Quickstart

```bash
cd lightrag
cp .env.example .env        # then add your GEMINI_API_KEY
pip install -r requirements.txt
python3 main.py --query "What were the specifications of AGC-1969?" --mode hybrid
```

Or run interactively:
```bash
python3 main.py
```

Try comparing modes on the same question:
```bash
python3 main.py --query "How does the AGC relate to the Apollo missions?" --mode naive
python3 main.py --query "How does the AGC relate to the Apollo missions?" --mode hybrid
```

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key ([Get one free](https://aistudio.google.com/)) |
| `LLM_MODEL` | `gemini-3.6-flash` | Model used for entity extraction & answer synthesis |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Model used for embeddings |
| `EMBEDDING_DIM` | `3072` | Must match `EMBEDDING_MODEL`'s output dimensionality |
| `LIGHTRAG_WORKING_DIR` | `./rag_storage` | Where the graph/vector/KV storage is persisted |

## Notes

- Indexing is slower and uses more LLM calls than the other two demos, because entity/relationship extraction runs an LLM pass over the document — expect it to take noticeably longer than `basic-rag` or `hybrid-search-rag` on the same file.
- `lightrag-hku`'s public API has changed across versions; if `main.py` errors on `LightRAG(...)`, `EmbeddingFunc`, or `initialize_pipeline_status`, check the signatures for the version pinned in your `requirements.txt` against the [LightRAG README](https://github.com/HKUDS/LightRAG).

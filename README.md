# RAG Demos

Three progressively more advanced Retrieval-Augmented Generation demos, each in its own self-contained folder (own `requirements.txt`, `.env.example`, and `README.md`). All three answer questions over the same sample knowledge base (`data/knowledge.txt` — Project Apollo & the AGC-1969 guidance computer) so you can compare them directly, and all three use Google Gemini as the LLM.

| Folder | What it demonstrates |
| :--- | :--- |
| [`basic-rag/`](basic-rag) | The minimal RAG pipeline: chunk → embed → cosine-similarity retrieve → generate. Start here. |
| [`hybrid-search-rag/`](hybrid-search-rag) | Production-style hybrid retrieval: **BM25L** sparse search + dense vector search (with a **ColBERT-style** MaxSim multi-vector backend) fused with **Reciprocal Rank Fusion (RRF)**, then **cross-encoder reranking**, then a **TruLens RAG Triad** evaluator (Context Relevance / Groundedness / Answer Relevance) scoring every answer. |
| [`lightrag/`](lightrag) | [LightRAG](https://github.com/HKUDS/LightRAG): graph-based RAG that extracts entities/relationships into a knowledge graph at indexing time, then retrieves via graph traversal + vector search (`naive`/`local`/`global`/`hybrid` modes). |

## Quickstart

Each folder is independent — `cd` into it, install its own dependencies, and run:

```bash
cd basic-rag          # or hybrid-search-rag, or lightrag
cp .env.example .env   # then add your GEMINI_API_KEY
pip install -r requirements.txt
python3 main.py --query "What were the specifications of AGC-1969?"
```

See each folder's own README for full details, configuration options, and example queries.

## Why three folders?

Each demo adds one layer of sophistication on top of the previous one, so you can see exactly what each technique buys you:

1. **basic-rag** — a single dense retriever. Fast to understand, but weak on exact keywords/IDs and has no way to judge answer quality.
2. **hybrid-search-rag** — adds a sparse lexical retriever (BM25L) alongside the dense one, fuses their rankings (RRF), reranks with a cross-encoder, and grades every answer with an LLM-as-judge triad (TruLens-style).
3. **lightrag** — replaces flat chunk retrieval with a knowledge graph, so multi-hop questions that connect several entities/facts can be answered more reliably.

None of the three share code — each is meant to be read end-to-end on its own.

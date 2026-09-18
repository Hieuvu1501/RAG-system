# RAG Demos

Five progressively more advanced Retrieval-Augmented Generation demos, each in its own self-contained folder (own `requirements.txt`, `.env.example`, and `README.md`). Four of them answer questions over the same sample knowledge base (`data/knowledge.txt` — Project Apollo & the AGC-1969 guidance computer) so you can compare them directly, and use Google Gemini as the LLM. The fifth (`colvintern`) is a different retrieval problem entirely — searching document *images* instead of text.

| Folder | What it demonstrates |
| :--- | :--- |
| [`basic-rag/`](basic-rag) | The minimal RAG pipeline, hand-rolled: chunk → embed → cosine-similarity retrieve → generate. Start here. |
| [`langchain-rag/`](langchain-rag) | The exact same minimal pipeline as `basic-rag`, rebuilt with [LangChain](https://python.langchain.com/)'s abstractions (`RecursiveCharacterTextSplitter`, `InMemoryVectorStore`, an LCEL chain) instead of hand-rolled code — a direct framework-vs-no-framework comparison. |
| [`hybrid-search-rag/`](hybrid-search-rag) | Production-style hybrid retrieval: **BM25L** sparse search + dense vector search (with a **ColBERT-style** MaxSim multi-vector backend, via `Vintern-Embedding-1B`) fused with **Reciprocal Rank Fusion (RRF)**, then **cross-encoder reranking**, then a **TruLens RAG Triad** evaluator (Context Relevance / Groundedness / Answer Relevance) scoring every answer. |
| [`lightrag/`](lightrag) | [LightRAG](https://github.com/HKUDS/LightRAG): graph-based RAG that extracts entities/relationships into a knowledge graph at indexing time, then retrieves via graph traversal + vector search (`naive`/`local`/`global`/`hybrid` modes). |
| [`colvintern/`](colvintern) | [ColVintern-1B-v1](https://huggingface.co/5CD-AI/ColVintern-1B-v1): ColPali-style **visual document retrieval** — embeds document images and text queries into a shared multi-vector space and scores them with MaxSim, no OCR or text chunking involved. Retrieval is fully local, no API key; an optional Gemini step reads the matched image to answer in words. |

## Quickstart

Each folder is independent — `cd` into it, install its own dependencies, and run:

```bash
cd basic-rag          # or hybrid-search-rag, or lightrag
cp .env.example .env   # then add your GEMINI_API_KEY
pip install -r requirements.txt
python3 main.py --query "What were the specifications of AGC-1969?"
```

See each folder's own README for full details, configuration options, and example queries.

## Why five folders?

**basic-rag**, **hybrid-search-rag**, and **lightrag** each add one layer of sophistication on top of the previous one, so you can see exactly what each technique buys you:

1. **basic-rag** — a single dense retriever, hand-rolled. Fast to understand, but weak on exact keywords/IDs and has no way to judge answer quality.
2. **hybrid-search-rag** — adds a sparse lexical retriever (BM25L) alongside the dense one, fuses their rankings (RRF), reranks with a cross-encoder, and grades every answer with an LLM-as-judge triad (TruLens-style).
3. **lightrag** — replaces flat chunk retrieval with a knowledge graph, so multi-hop questions that connect several entities/facts can be answered more reliably.

The other two are siblings rather than next steps in that progression:

- **langchain-rag** sits right next to **basic-rag**: same pipeline, same models, same data — the only difference is hand-rolled code vs. a framework's abstractions. Read them side by side to see what LangChain buys you (less boilerplate, swappable components) and what it costs you (the mechanics are hidden inside the library instead of visible in your own code).
- **colvintern** retrieves over document *images* instead of chunked text, using the same multi-vector/MaxSim idea that powers `hybrid-search-rag`'s dense backend.

None of the five share code — each is meant to be read end-to-end on its own.

# 🦜 LangChain RAG Demo

> Part of the [RAG demos](../README.md) collection — see also [basic-rag](../basic-rag), [hybrid-search-rag](../hybrid-search-rag), [lightrag](../lightrag), and [colvintern](../colvintern).

The same retrieve-then-generate pipeline as [`basic-rag`](../basic-rag) — same knowledge base, same Gemini models — but built with [LangChain](https://python.langchain.com/)'s abstractions instead of hand-rolled code. This demo exists specifically to compare against `basic-rag` line by line and see what a framework buys you (and what it hides).

## 🏛 Pipeline

```
   Document
      │
      ▼
 [RecursiveCharacterTextSplitter]
      │
      ▼
 [GoogleGenerativeAIEmbeddings]  (models/gemini-embedding-001)
      │
      ▼
 [InMemoryVectorStore]  (cosine similarity, via .as_retriever())
      │
      ▼
   Query ──► LCEL chain: {context: retriever|format_docs, question} | prompt | ChatGoogleGenerativeAI | StrOutputParser
```

## basic-rag vs. langchain-rag — line-by-line comparison

| Step | [`basic-rag`](../basic-rag) (hand-rolled) | `langchain-rag` (this folder) |
| :--- | :--- | :--- |
| Chunking | `rag/chunker.py`: custom `TextChunker` (paragraph split + sliding window) | `RecursiveCharacterTextSplitter` |
| Embeddings | `rag/embeddings.py`: `GeminiEmbedder` wrapping raw `google-genai` calls | `GoogleGenerativeAIEmbeddings` |
| Vector store | `rag/vector_store.py`: `SimpleVectorStore` with manual numpy cosine similarity | `InMemoryVectorStore.as_retriever()` |
| Prompt assembly | An f-string built by hand in `pipeline.py` | `ChatPromptTemplate.from_template(...)` |
| LLM call | `client.models.generate_content(...)`, manual response-text extraction | `ChatGoogleGenerativeAI` |
| Wiring it together | Explicit `index_document()` / `retrieve()` / `generate_answer()` methods | One LCEL chain: `{...} | prompt | llm | StrOutputParser()` |

Same idea, same models, same data. `basic-rag` shows you every mechanic explicitly; `langchain-rag` shows the same result composed from framework primitives with the mechanics hidden inside the library.

## 📂 Project Structure

```
langchain-rag/
├── .env.example       # Environment configuration template
├── requirements.txt   # langchain, langchain-core, langchain-text-splitters, langchain-google-genai, python-dotenv
├── data/
│   └── knowledge.txt  # Sample knowledge base (Project Apollo & AGC-1969)
├── main.py            # LCEL RAG chain: split -> embed -> retrieve -> generate
└── README.md
```

## ⚡ Quickstart

```bash
cd langchain-rag
cp .env.example .env        # then add your GEMINI_API_KEY
pip install -r requirements.txt
.venv/bin/python main.py --query "What were the specifications of AGC-1969?" --debug
```

Or run interactively:
```bash
python3 main.py
```

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key ([Get one free](https://aistudio.google.com/)) |
| `LLM_MODEL` | `gemini-3.6-flash` | Model used for answer generation |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Model used for dense embeddings |
| `TOP_K` | `3` | Number of chunks retrieved per query |

## What this demo intentionally leaves out

Just like `basic-rag`: no BM25/sparse retrieval, no rank fusion, no reranking, no answer-quality evaluation. It's the LangChain equivalent of the *simplest* demo in this repo, not the hybrid one — a fair one-to-one comparison. LangChain can absolutely do BM25 ensembles, rerankers, and multi-vector retrieval too (via `EnsembleRetriever`, `ContextualCompressionRetriever`, etc.), but that's a bigger demo than "should I use a framework for the basics" calls for.

## Is LangChain worth it?

For learning the mechanics of RAG — chunking strategy, similarity math, prompt construction — no; that's exactly what `basic-rag` and `hybrid-search-rag` are for, and LangChain hides those mechanics behind its abstractions. For building something fast once you already understand the mechanics — swapping vector stores, LLM providers, or adding memory/agents without rewriting plumbing — LangChain (or similar frameworks) starts to pay for itself.

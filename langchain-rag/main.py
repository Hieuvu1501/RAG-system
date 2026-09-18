#!/usr/bin/env python3
"""
LangChain RAG demo: the same retrieve-then-generate pipeline as basic-rag,
built with LangChain's abstractions instead of hand-rolled code.

Compare this file against ../basic-rag/rag/pipeline.py line by line:
- TextChunker + custom paragraph/sliding-window logic  -> RecursiveCharacterTextSplitter
- GeminiEmbedder + manual embed_content() calls         -> GoogleGenerativeAIEmbeddings
- SimpleVectorStore + manual cosine similarity          -> InMemoryVectorStore.as_retriever()
- Manual prompt string + client.models.generate_content -> ChatPromptTemplate | ChatGoogleGenerativeAI
- Hand-wired index_document()/retrieve()/generate_answer -> one LCEL chain (the `|` pipe operator)

Same knowledge base, same Gemini models, same retrieve-then-generate idea -
just composed through a framework instead of written by hand. Neither
approach is "more correct"; this exists so you can see the trade-off
directly: LangChain buys you less boilerplate and swappable components, at
the cost of the mechanics (chunking rules, similarity math, prompt assembly)
being hidden inside the library instead of visible in your own code.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
TOP_K = int(os.getenv("TOP_K", "3"))

PROMPT = ChatPromptTemplate.from_template(
    """You are an accurate, helpful AI assistant. Answer the user's question based strictly on \
the provided context. If the answer cannot be found in the context, explicitly state that the \
information is not present. Do not fabricate facts.

Context:
{context}

User Question:
{question}

Grounded Answer:"""
)


def check_api_key():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        print("\n[!] GEMINI_API_KEY is not configured in `.env`.")
        print("Copy `.env.example` to `.env` and set your key.")
        print("Get a free API key from Google AI Studio: https://aistudio.google.com/\n")
        sys.exit(1)


def build_chain(vector_store: InMemoryVectorStore, top_k: int):
    """
    The canonical LangChain LCEL RAG chain: retriever feeds formatted context
    into the prompt, the prompt feeds the LLM, the LLM's output is parsed to
    a plain string. The `|` operator pipes each step's output into the next.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=os.getenv("GEMINI_API_KEY"))

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[Chunk {i} - ID: {d.metadata.get('chunk_id', i)}]\n{d.page_content}"
            for i, d in enumerate(docs, start=1)
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever, rag_chain


def index_document(file_path: str, chunk_size: int = 400, chunk_overlap: int = 60) -> InMemoryVectorStore:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(content)
    documents = [
        Document(page_content=chunk, metadata={"chunk_id": f"{os.path.basename(file_path)}#chunk-{i:04d}"})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=os.getenv("GEMINI_API_KEY"))
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)
    return vector_store, len(documents)


def display_result(question: str, retrieved: list[Document], answer: str, debug: bool):
    print("\n" + "=" * 68)
    print(f" Query: {question}")
    print("=" * 68)

    if debug:
        print("\n[Retrieved Chunks (LangChain InMemoryVectorStore, cosine similarity)]")
        for rank, doc in enumerate(retrieved, start=1):
            snippet = doc.page_content[:90].replace("\n", " ") + "..."
            print(f"  {rank}. ({doc.metadata.get('chunk_id', rank)}) {snippet}")

    print("\n💡 Answer:")
    print("-" * 68)
    print(answer)
    print("-" * 68)
    print()


def main():
    parser = argparse.ArgumentParser(description="LangChain RAG pipeline: split -> embed -> retrieve -> generate")
    parser.add_argument("--data", default="data/knowledge.txt", help="Path to the document to index")
    parser.add_argument("--query", type=str, help="Single question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--debug", action="store_true", help="Show retrieved chunks")
    args = parser.parse_args()

    check_api_key()
    top_k = args.top_k or TOP_K

    print("=" * 68)
    print("        🦜 LANGCHAIN RAG PIPELINE (Split -> Embed -> Retrieve -> Generate)")
    print("=" * 68)

    print(f"Loading and indexing document: {args.data} ...")
    try:
        vector_store, num_chunks = index_document(args.data)
        retriever, rag_chain = build_chain(vector_store, top_k)
        print(f"✅ Indexed {num_chunks} chunks into an in-memory LangChain vector store.\n")
    except Exception as e:
        print(f"\n[!] Failed to initialize or index: {e}")
        sys.exit(1)

    def ask(question: str):
        retrieved = retriever.invoke(question)
        answer = rag_chain.invoke(question)
        display_result(question, retrieved, answer, args.debug)

    if args.query:
        ask(args.query)
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
            ask(user_prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

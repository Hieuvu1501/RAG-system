import re
import math
from typing import List, Tuple, Optional
from collections import Counter
from rag.chunker import Chunk

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def tokenize(text: str) -> List[str]:
    """
    Standard tokenizer that splits on non-alphanumeric characters while preserving
    alphanumeric tokens, acronyms, and hyphenated technical terms (e.g. AGC-1969, SA-506).
    """
    # Lowercase and split into alphanumeric tokens and hyphenated identifiers
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", text.lower())
    return tokens


class PurePythonBM25Okapi:
    """
    Fallback pure-Python BM25Okapi implementation in case `rank_bm25`
    is not installed. Guarantees 100% test reliability with zero external dependencies.
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / max(1, self.corpus_size)
        self.doc_freqs: List[Counter] = [Counter(doc) for doc in corpus]

        # Calculate inverse document frequency (IDF) for all terms
        self.df: Counter = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1

        self.idf = {}
        for term, freq in self.df.items():
            # Standard Lucene/BM25 IDF formula with smoothing
            self.idf[term] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for idx, doc_freq in enumerate(self.doc_freqs):
                tf = doc_freq.get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1, self.avgdl)))
                scores[idx] += idf * (numerator / denominator)
        return scores


class BM25Retriever:
    """
    Sparse Lexical Retriever using BM25Okapi inverted indexing.
    Ideal for exact keyword matches, technical codes, dates, and identifiers.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None

    def index_chunks(self, chunks: List[Chunk]):
        """
        Builds the inverted index from a list of Chunk objects.
        """
        self.chunks = list(chunks)
        self.tokenized_corpus = [tokenize(c.text) for c in chunks]

        if not self.chunks:
            self.bm25 = None
            return

        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self.bm25 = PurePythonBM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Retrieves top-k lexical chunks matching the query.
        Returns:
            List of (Chunk, bm25_score) sorted descending by score.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Sort indices descending by score
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        k = min(top_k, len(self.chunks))
        results: List[Tuple[Chunk, float]] = []
        for idx, score in indexed_scores[:k]:
            if score > 0:  # Only return chunks with non-zero lexical overlap
                results.append((self.chunks[idx], score))

        return results

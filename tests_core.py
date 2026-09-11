import unittest
import numpy as np
import torch
from rag.chunker import TextChunker, Chunk
from rag.vector_store import SimpleVectorStore
from rag.multi_vector_store import MultiVectorStore
from rag.bm25_retriever import BM25Retriever, tokenize
from rag.fusion import reciprocal_rank_fusion
from rag.reranker import CrossEncoderReranker


class TestHybridRAGCore(unittest.TestCase):

    def test_chunker(self):
        chunker = TextChunker(chunk_size=60, chunk_overlap=15)
        text = (
            "Section 1: Apollo guidance systems were developed by MIT.\n\n"
            "Section 2: The Saturn V was an American super heavy-lift rocket capable of lifting massive payloads into orbit."
        )
        chunks = chunker.chunk_text(text, source="test_doc")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))
        self.assertTrue(chunks[0].id.startswith("test_doc#chunk-"))
        self.assertIn("MIT", chunks[0].text)

    def test_vector_store_cosine_similarity(self):
        store = SimpleVectorStore()
        chunks = [
            Chunk(id="c1", text="Machine learning and artificial intelligence"),
            Chunk(id="c2", text="Deep space exploration and rocketry"),
            Chunk(id="c3", text="Cooking delicious Italian pasta"),
        ]
        # Vectors: c1 along x-axis, c2 along y-axis, c3 along z-axis
        vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        store.add_chunks(chunks, vecs)

        # Query very close to c2 (rocketry)
        q = np.array([0.05, 0.95, 0.0], dtype=np.float32)
        results = store.similarity_search(q, top_k=2)

        self.assertEqual(len(results), 2)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.id, "c2")
        self.assertGreater(score, 0.9)

    def test_multi_vector_store_maxsim(self):
        """Tests ColBERT-style MaxSim retrieval with synthetic multi-vector embeddings."""
        store = MultiVectorStore()
        chunks = [
            Chunk(id="mv1", text="Machine learning and artificial intelligence"),
            Chunk(id="mv2", text="Deep space exploration and rocketry"),
            Chunk(id="mv3", text="Cooking delicious Italian pasta"),
        ]

        # Create synthetic multi-vector embeddings (each doc has 3 tokens, dim=4)
        # mv1: tokens point along x-axis (AI/ML domain)
        emb1 = torch.tensor([
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0],
        ], dtype=torch.float32)

        # mv2: tokens point along y-axis (space domain)
        emb2 = torch.tensor([
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.8, 0.2, 0.0],
            [0.1, 0.7, 0.2, 0.0],
        ], dtype=torch.float32)

        # mv3: tokens point along z-axis (cooking domain)
        emb3 = torch.tensor([
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.8, 0.2],
            [0.0, 0.1, 0.7, 0.2],
        ], dtype=torch.float32)

        store.add_chunks(chunks, [emb1, emb2, emb3])

        # Query with tokens pointing toward y-axis (space domain) → should match mv2
        query_emb = torch.tensor([
            [0.05, 0.95, 0.0, 0.0],
            [0.0, 0.85, 0.15, 0.0],
        ], dtype=torch.float32)

        results = store.similarity_search(query_emb, top_k=3)

        self.assertEqual(len(results), 3)
        # mv2 (space) should be the top result
        self.assertEqual(results[0][0].id, "mv2")
        # MaxSim score should be positive
        self.assertGreater(results[0][1], 0.0)
        # mv2 should score higher than mv1 and mv3
        self.assertGreater(results[0][1], results[1][1])
        self.assertGreater(results[0][1], results[2][1])

    def test_multi_vector_store_empty(self):
        """Tests that empty MultiVectorStore returns no results."""
        store = MultiVectorStore()
        query_emb = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        results = store.similarity_search(query_emb, top_k=5)
        self.assertEqual(len(results), 0)

    def test_multi_vector_store_clear(self):
        """Tests that clearing the MultiVectorStore removes all data."""
        store = MultiVectorStore()
        chunks = [Chunk(id="c1", text="test")]
        embs = [torch.tensor([[1.0, 0.0]], dtype=torch.float32)]
        store.add_chunks(chunks, embs)
        self.assertEqual(len(store.chunks), 1)

        store.clear()
        self.assertEqual(len(store.chunks), 0)
        self.assertEqual(len(store.embeddings), 0)

    def test_bm25_retriever_exact_match(self):
        retriever = BM25Retriever()
        chunks = [
            Chunk(id="c1", text="The model designation is AGC-1969 with Block II specs."),
            Chunk(id="c2", text="The rover traveled across the lunar surface."),
            Chunk(id="c3", text="Liquid fuel was used for Saturn-V-SA-506."),
        ]
        retriever.index_chunks(chunks)

        # Search for exact technical code
        res1 = retriever.search("AGC-1969", top_k=2)
        self.assertTrue(len(res1) > 0)
        self.assertEqual(res1[0][0].id, "c1")

        # Search for rocket designation
        res2 = retriever.search("Saturn-V-SA-506", top_k=2)
        self.assertTrue(len(res2) > 0)
        self.assertEqual(res2[0][0].id, "c3")

    def test_reciprocal_rank_fusion(self):
        c1 = Chunk(id="chunk-A", text="A")
        c2 = Chunk(id="chunk-B", text="B")
        c3 = Chunk(id="chunk-C", text="C")

        # Dense ranks: A (rank 1), B (rank 2)
        dense_results = [(c1, 0.95), (c2, 0.80)]

        # Sparse ranks: B (rank 1), C (rank 2)
        sparse_results = [(c2, 4.2), (c3, 3.1)]

        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

        # Candidate B appeared in both (rank 2 in dense, rank 1 in sparse)
        # score(B) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.03252
        # score(A) = 1/(60+1) = 1/61 = 0.01639
        # score(C) = 1/(60+2) = 1/62 = 0.01612
        # Therefore, B should win the top rank!
        self.assertEqual(fused[0].chunk.id, "chunk-B")
        self.assertEqual(fused[1].chunk.id, "chunk-A")
        self.assertEqual(fused[2].chunk.id, "chunk-C")

    def test_reranker_fallback(self):
        reranker = CrossEncoderReranker(api_key=None)
        c1 = Chunk(id="c1", text="Doc 1")
        c2 = Chunk(id="c2", text="Doc 2")
        candidates = [(c1, 0.05), (c2, 0.03)]

        # Without API key, falls back gracefully to RRF order
        reranked = reranker.rerank("any query", candidates, top_n=2)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].chunk.id, "c1")


if __name__ == "__main__":
    unittest.main()

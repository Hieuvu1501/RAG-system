import unittest
import numpy as np

from rag.chunker import TextChunker
from rag.vector_store import SimpleVectorStore


class TestTextChunker(unittest.TestCase):
    def test_chunks_short_paragraphs_as_is(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_text("Hello world.\n\nAnother short paragraph.", source="doc")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "Hello world.")

    def test_sliding_window_on_long_paragraph(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        long_text = "a" * 50
        chunks = chunker.chunk_text(long_text, source="doc")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), 20)

    def test_empty_text_returns_no_chunks(self):
        chunker = TextChunker()
        self.assertEqual(chunker.chunk_text("   "), [])

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=10, chunk_overlap=10)


class TestSimpleVectorStore(unittest.TestCase):
    def test_cosine_similarity_ranks_closest_vector_first(self):
        chunker = TextChunker()
        chunks = chunker.chunk_text("first\n\nsecond\n\nthird", source="doc")

        store = SimpleVectorStore()
        vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        store.add_chunks(chunks, vectors)

        results = store.similarity_search(np.array([1, 0]), top_k=1)
        self.assertEqual(results[0][0].text, "first")

    def test_empty_store_returns_no_results(self):
        store = SimpleVectorStore()
        self.assertEqual(store.similarity_search(np.array([1, 0])), [])


if __name__ == "__main__":
    unittest.main()

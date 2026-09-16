from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        snippet = self.text[:60].replace("\n", " ") + ("..." if len(self.text) > 60 else "")
        return f"Chunk(id={self.id!r}, text={snippet!r})"


class TextChunker:
    """
    Splits a text document into fixed-size overlapping chunks by paragraph,
    falling back to a sliding character window for long paragraphs.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str = "document") -> List[Chunk]:
        text = text.strip()
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        raw_chunks: List[str] = []

        for p in paragraphs:
            if len(p) <= self.chunk_size:
                raw_chunks.append(p)
                continue

            start = 0
            step = self.chunk_size - self.chunk_overlap
            while start < len(p):
                end = min(start + self.chunk_size, len(p))
                piece = p[start:end].strip()
                if piece:
                    raw_chunks.append(piece)
                if end == len(p):
                    break
                start += step

        chunks: List[Chunk] = []
        for idx, chunk_str in enumerate(raw_chunks):
            chunks.append(
                Chunk(
                    id=f"{source}#chunk-{idx:04d}",
                    text=chunk_str,
                    metadata={"source": source, "chunk_index": idx},
                )
            )
        return chunks

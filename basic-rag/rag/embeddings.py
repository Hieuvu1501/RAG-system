import os
from typing import List, Optional
import numpy as np

try:
    from google import genai
except ImportError:
    genai = None


class GeminiEmbedder:
    """
    Generates single-vector dense embeddings using Google Gemini's embedding
    model via the official `google-genai` SDK.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-embedding-001"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

        if not self.api_key or self.api_key.strip() == "" or "your_gemini_api_key_here" in self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your .env file or environment."
            )

        if genai is None:
            raise ImportError(
                "The `google-genai` library is not installed. Install it via `pip install -r requirements.txt`."
            )

        self.client = genai.Client(api_key=self.api_key)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of texts, returning a 2D array of shape (N, D)."""
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        embeddings: List[List[float]] = []
        for text in texts:
            response = self.client.models.embed_content(model=self.model, contents=text)
            embeddings.append(response.embeddings[0].values)

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string, returning a 1D array of shape (D,)."""
        return self.embed_texts([query])[0]

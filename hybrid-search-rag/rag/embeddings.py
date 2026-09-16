import os
from typing import List, Optional
import numpy as np

try:
    from google import genai
except ImportError:
    genai = None


class GeminiEmbedder:
    """
    Generates dense semantic vector embeddings using Google Gemini's embedding models
    via the official `google-genai` SDK.
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
        """
        Generates vector embeddings for a list of text strings.
        Returns a 2D numpy array of shape (N, D).
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        embeddings: List[List[float]] = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
            )
            # Response embeddings are accessed via response.embeddings[0].values
            embeddings.append(response.embeddings[0].values)

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates vector embedding for a single query string.
        Returns a 1D numpy array of shape (D,).
        """
        vecs = self.embed_texts([query])
        return vecs[0]

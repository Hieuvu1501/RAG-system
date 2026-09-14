import os
import re
import json
from typing import List, Optional
from dataclasses import dataclass

try:
    from google import genai
except ImportError:
    genai = None


@dataclass
class TriadResult:
    """
    Stores the three RAG Triad scores and the composite average.

    Scores are in the range [0.0, 1.0]:
    - 0.0: Completely fails the metric
    - 1.0: Perfectly passes the metric

    Attributes:
        context_relevance:  Avg relevance of retrieved chunks to the query.
        groundedness:       Degree to which the answer is supported by the context.
        answer_relevance:   Degree to which the answer addresses the query.
        composite:          Mean of all three scores.
    """
    context_relevance: float
    groundedness: float
    answer_relevance: float
    composite: float

    def to_dict(self) -> dict:
        return {
            "context_relevance": round(self.context_relevance, 4),
            "groundedness": round(self.groundedness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "composite": round(self.composite, 4),
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float value to [lo, hi]."""
    return max(lo, min(hi, value))


def _extract_score(text: str, fallback: float = 0.5) -> float:
    """
    Parses a numeric score from a Gemini response.
    Accepts JSON {"score": 0.87} or bare floats/ints anywhere in the text.
    """
    text = text.strip()

    # Try JSON first: {"score": 0.87} or {"score": "0.87"}
    try:
        clean = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        clean = re.sub(r"\n?```$", "", clean)
        data = json.loads(clean)
        if isinstance(data, dict) and "score" in data:
            return _clamp(float(data["score"]))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: first float/int in the response
    match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", text)
    if match:
        return _clamp(float(match.group(1)))

    return fallback


def _safe_llm_call(client, model: str, prompt: str) -> str:
    """
    Executes an LLM call and safely extracts text from candidates,
    avoiding thought_signature warnings.
    """
    response = client.models.generate_content(model=model, contents=prompt)
    parts = []
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        parts.append(part.text)
    return "".join(parts).strip() if parts else response.text.strip()


class RAGTriadEvaluator:
    """
    TruLens-style RAG Triad Evaluator powered by Gemini LLM-as-a-Judge.

    Evaluates three hallucination-detection metrics on every RAG response:

    1. Context Relevance  — Are the retrieved chunks relevant to the query?
    2. Groundedness       — Is every claim in the answer supported by the context?
    3. Answer Relevance   — Does the answer directly address the user's question?

    All scores are in [0.0, 1.0]. A composite score is the mean of all three.
    """

    CONTEXT_RELEVANCE_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Task: Assess whether the retrieved passage is relevant to the user's query.
Score from 0.0 (completely irrelevant) to 1.0 (directly and completely relevant).

Query:
"{query}"

Retrieved Passage:
"{chunk}"

Instructions:
- Score 1.0 if the passage directly contains information needed to answer the query.
- Score 0.5 if the passage is topically related but only partially useful.
- Score 0.0 if the passage has no meaningful connection to the query.

Respond ONLY with valid JSON: {{"score": <float between 0.0 and 1.0>}}"""

    GROUNDEDNESS_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Task: Assess whether the answer is fully supported by (grounded in) the provided context.
Score from 0.0 (answer contains claims with no support in the context) to 1.0 (every claim is directly supported).

Context:
"{context}"

Answer:
"{answer}"

Instructions:
- Score 1.0 if every factual claim in the answer can be traced back to the context.
- Score 0.5 if some claims are supported but others are inferred or unsupported.
- Score 0.0 if the answer introduces facts that are absent from or contradicted by the context.
- Do NOT penalize for paraphrasing or slight rewording of context.

Respond ONLY with valid JSON: {{"score": <float between 0.0 and 1.0>}}"""

    ANSWER_RELEVANCE_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Task: Assess whether the answer directly and completely addresses the user's original query.
Score from 0.0 (completely off-topic) to 1.0 (fully and directly answers the query).

Query:
"{query}"

Answer:
"{answer}"

Instructions:
- Score 1.0 if the answer directly and completely resolves the user's question.
- Score 0.5 if the answer is partially relevant but misses key aspects.
- Score 0.0 if the answer does not address the question at all.

Respond ONLY with valid JSON: {{"score": <float between 0.0 and 1.0>}}"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # api_key=None  → read from environment (normal runtime use)
        # api_key=""    → explicitly disabled (useful in tests / offline mode)
        self.api_key = os.getenv("GEMINI_API_KEY") if api_key is None else api_key
        self.model = model or os.getenv("TRIAD_EVAL_MODEL", os.getenv("LLM_MODEL", "gemini-3.6-flash"))
        self.client = None

        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def score_context_relevance(self, query: str, chunks: List[str]) -> float:
        """
        Computes context relevance as the average per-chunk relevance score.

        Args:
            query:  The user query string.
            chunks: List of retrieved document chunk texts.

        Returns:
            Float in [0.0, 1.0] — average relevance score across all chunks.
        """
        if not self.client or not chunks:
            return 0.5  # Neutral fallback

        scores = []
        for chunk in chunks:
            prompt = self.CONTEXT_RELEVANCE_PROMPT.format(
                query=query.strip(),
                chunk=chunk.strip()[:1500],  # Limit to avoid context overflow
            )
            try:
                raw = _safe_llm_call(self.client, self.model, prompt)
                scores.append(_extract_score(raw))
            except Exception:
                scores.append(0.5)

        return _clamp(sum(scores) / len(scores)) if scores else 0.5

    def score_groundedness(self, answer: str, chunks: List[str]) -> float:
        """
        Scores how well the answer is grounded in the retrieved context.

        Args:
            answer: The LLM-generated answer.
            chunks: List of retrieved document chunk texts used as context.

        Returns:
            Float in [0.0, 1.0].
        """
        if not self.client or not answer or not chunks:
            return 0.5

        context = "\n\n".join(chunks)[:3000]  # Limit to avoid overflow
        prompt = self.GROUNDEDNESS_PROMPT.format(
            context=context.strip(),
            answer=answer.strip(),
        )
        try:
            raw = _safe_llm_call(self.client, self.model, prompt)
            return _extract_score(raw)
        except Exception:
            return 0.5

    def score_answer_relevance(self, query: str, answer: str) -> float:
        """
        Scores how directly the answer addresses the user's query.

        Args:
            query:  The user query string.
            answer: The LLM-generated answer.

        Returns:
            Float in [0.0, 1.0].
        """
        if not self.client or not answer:
            return 0.5

        prompt = self.ANSWER_RELEVANCE_PROMPT.format(
            query=query.strip(),
            answer=answer.strip(),
        )
        try:
            raw = _safe_llm_call(self.client, self.model, prompt)
            return _extract_score(raw)
        except Exception:
            return 0.5

    def evaluate(
        self,
        query: str,
        context_chunks: List[str],
        answer: str,
    ) -> TriadResult:
        """
        Runs all three RAG Triad evaluations and returns a TriadResult.

        Args:
            query:          The user's original query.
            context_chunks: List of retrieved chunk texts passed to the LLM.
            answer:         The LLM-generated grounded answer.

        Returns:
            TriadResult with context_relevance, groundedness, answer_relevance, composite.
        """
        context_relevance = self.score_context_relevance(query, context_chunks)
        groundedness = self.score_groundedness(answer, context_chunks)
        answer_relevance = self.score_answer_relevance(query, answer)
        composite = _clamp((context_relevance + groundedness + answer_relevance) / 3.0)

        return TriadResult(
            context_relevance=context_relevance,
            groundedness=groundedness,
            answer_relevance=answer_relevance,
            composite=composite,
        )

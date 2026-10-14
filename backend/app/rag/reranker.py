import os
import re
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass
from app.rag.chunker import Chunk

try:
    from google import genai
except ImportError:
    genai = None


@dataclass
class RerankedResult:
    chunk: Chunk
    rerank_score: float
    rrf_score: float
    original_rank: int


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker that assesses deep semantic relevance of (Query, Document) pairs.
    Uses Google Gemini (via GEMINI_API_KEY) with structured relevance scoring,
    with an optional hook for local sentence-transformers CrossEncoder.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.6-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Tuple[Chunk, float]],
        top_n: int = 3,
    ) -> List[RerankedResult]:
        """
        Reranks candidate chunks based on cross-attention / relevance scoring.

        Args:
            query: The user query.
            candidate_chunks: List of (Chunk, rrf_score) from rank fusion.
            top_n: Number of top chunks to return after reranking.

        Returns:
            List of RerankedResult sorted descending by rerank_score.
        """
        if not candidate_chunks:
            return []

        # If only 1 candidate, or no client available, return as-is
        if len(candidate_chunks) == 1 or self.client is None:
            results = [
                RerankedResult(
                    chunk=chunk,
                    rerank_score=score,
                    rrf_score=score,
                    original_rank=i + 1,
                )
                for i, (chunk, score) in enumerate(candidate_chunks[:top_n])
            ]
            return results

        # Format candidates for the Cross-Encoder prompt
        candidates_text = ""
        for i, (chunk, _) in enumerate(candidate_chunks):
            snippet = chunk.text.replace("\n", " ")
            candidates_text += f"[{i}] {snippet}\n"

        prompt = f"""You are an expert retrieval reranker (Cross-Encoder).
Task: Given a user query and numbered candidate passages, evaluate and score the relevance of EACH passage to the query on a continuous scale from 0.00 (completely irrelevant) to 10.00 (directly and accurately answers the query).

Query: "{query}"

Candidate Passages:
{candidates_text}

Respond strictly with valid JSON array of objects, each containing "index" (integer) and "score" (float between 0.0 and 10.0), with no extra text or markdown code fences:
[
  {{"index": 0, "score": 8.5}},
  ...
]"""

        scored_results: List[RerankedResult] = []
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            # Extract text safely from candidate parts to avoid non-text warning
            text_parts = []
            if hasattr(response, "candidates") and response.candidates:
                for cand in response.candidates:
                    if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                        for part in cand.content.parts:
                            if getattr(part, "text", None):
                                text_parts.append(part.text)
            raw_text = "".join(text_parts).strip() if text_parts else response.text.strip()

            # Clean markdown JSON block if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)

            parsed = json.loads(raw_text)
            score_dict = {item["index"]: float(item["score"]) for item in parsed if "index" in item}

            for idx, (chunk, rrf_score) in enumerate(candidate_chunks):
                # If model missed an index, default score to scaled rrf
                score = score_dict.get(idx, rrf_score * 10.0)
                scored_results.append(
                    RerankedResult(
                        chunk=chunk,
                        rerank_score=score,
                        rrf_score=rrf_score,
                        original_rank=idx + 1,
                    )
                )

            # Sort by reranker score descending
            scored_results.sort(key=lambda x: x.rerank_score, reverse=True)
            return scored_results[:top_n]

        except Exception as e:
            # Graceful fallback: maintain RRF ordering if reranking API call fails
            fallback_results = [
                RerankedResult(
                    chunk=chunk,
                    rerank_score=score,
                    rrf_score=score,
                    original_rank=i + 1,
                )
                for i, (chunk, score) in enumerate(candidate_chunks[:top_n])
            ]
            return fallback_results

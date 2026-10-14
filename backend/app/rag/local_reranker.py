"""
Cross-encoder reranking, running fully locally via sentence-transformers -
no API call, so it doesn't count against the Gemini quota. Default is
itdainb/PhoRanker - a Vietnamese-specific reranker (~100M params) that beats
the general multilingual bge-reranker-v2-m3 (568M params) on Vietnamese
benchmarks while being ~5x smaller and ~4x faster - smaller model AND better
quality for this project's use case, not the usual size/quality trade-off.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.rag.chunker import Chunk
from app.rag.device import resolve_device


@dataclass
class RerankedResult:
    chunk: Chunk
    rerank_score: float
    rrf_score: float
    rrf_rank: int  # 1-indexed position after RRF fusion, before this reranking step


class LocalCrossEncoderReranker:
    def __init__(self, model_name: str = "itdainb/PhoRanker", device: Optional[str] = None, max_length: int = 256):
        self.model_name = model_name
        self.device = resolve_device(device or "auto")
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from sentence_transformers import CrossEncoder

        # bfloat16 on GPU/MPS roughly halves this model's resident memory vs
        # the sentence-transformers default (float32) - matches the dtype
        # VinternEmbedder already uses on those devices. CPU keeps float32
        # (no bf16 speedup there, and it's the safer default for numerical
        # stability on CPU kernels).
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self._model = CrossEncoder(
            self.model_name, device=self.device, max_length=self.max_length, automodel_args={"torch_dtype": dtype}
        )

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> Optional[List[float]]:
        """
        Scores arbitrary (text_a, text_b) pairs with the same local
        cross-encoder used for reranking, as sigmoid-normalized [0,1]
        relevance-ish scores. Returns None if the model fails to load/run.
        Reused for the RAG Triad's groundedness/answer-relevance proxies so
        those don't need a separate model or an API call.
        """
        if not pairs:
            return []
        try:
            self._load()
            raw_scores = self._model.predict(pairs)
        except Exception:
            return None
        return [1 / (1 + math.exp(-float(raw))) for raw in raw_scores]

    def rerank(self, query: str, candidate_chunks: List[Tuple[Chunk, float]], top_n: int = 3) -> List[RerankedResult]:
        if not candidate_chunks:
            return []

        if len(candidate_chunks) == 1:
            chunk, score = candidate_chunks[0]
            return [RerankedResult(chunk=chunk, rerank_score=score, rrf_score=score, rrf_rank=1)]

        pairs = [(query, chunk.text) for chunk, _ in candidate_chunks]
        scores = self.score_pairs(pairs)
        if scores is None:
            # Graceful fallback: maintain RRF ordering if the local model fails to load/run
            return [
                RerankedResult(chunk=chunk, rerank_score=score, rrf_score=score, rrf_rank=i + 1)
                for i, (chunk, score) in enumerate(candidate_chunks[:top_n])
            ]

        # rrf_rank is each chunk's 1-indexed position in candidate_chunks - the
        # RRF-fused order from retrieve() - captured *before* the sort below,
        # so it still reflects "where RRF placed it", not where reranking did.
        results = [
            RerankedResult(chunk=chunk, rerank_score=score, rrf_score=rrf_score, rrf_rank=i + 1)
            for i, ((chunk, rrf_score), score) in enumerate(zip(candidate_chunks, scores))
        ]
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_n]

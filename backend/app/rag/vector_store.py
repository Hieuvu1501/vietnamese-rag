from typing import List, Tuple

import torch
import torch.nn.functional as F

from app.rag.chunker import Chunk


def mean_pool(token_embeddings: torch.Tensor) -> torch.Tensor:
    """Collapses Vintern's [num_tokens, dim] output into one L2-normalised [dim] vector."""
    emb = token_embeddings.detach().cpu().float()
    # Vintern multiplies its output by the attention mask, so padded positions are exactly zero.
    real = emb[emb.norm(dim=-1) > 0]
    return F.normalize((real if len(real) else emb).mean(dim=0), dim=0)


class VectorStore:
    """
    One vector per chunk (mean of Vintern-Embedding-1B's token vectors), searched by cosine similarity.
    ~150x smaller than keeping every token vector for MaxSim, but Vintern was trained for token-level
    MaxSim, so averaging gives up some retrieval precision.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.matrix = torch.empty(0)  # [num_chunks, dim] once populated

    def add_chunks(self, chunks: List[Chunk], embeddings: List[torch.Tensor]):
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)}).")
        if not chunks:
            return
        new = torch.stack([mean_pool(e) for e in embeddings])
        self.matrix = new if self.matrix.numel() == 0 else torch.cat([self.matrix, new])
        self.chunks.extend(chunks)

    def similarity_search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores = self.matrix @ mean_pool(query_embedding)
        top = torch.topk(scores, min(top_k, len(self.chunks)))
        return [(self.chunks[i], s) for s, i in zip(top.values.tolist(), top.indices.tolist())]

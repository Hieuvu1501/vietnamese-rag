"""
LangChain BaseRetriever wrappers around our Vietnamese-tuned retrieval code.

Vintern-Embedding-1B needs its own processor (separate doc/query encoding) and
outputs one vector per token, which app.rag.vector_store mean-pools into one
vector per chunk - neither fits LangChain's `Embeddings` interface directly.
`BaseRetriever` only requires "text in, ranked Documents out", so wrapping each
retriever this way lets them compose with LangChain's `EnsembleRetriever` for RRF.
"""
from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.rag.bm25_retriever import BM25Retriever
from app.rag.chunker import Chunk
from app.rag.vector_store import VectorStore
from app.rag.vintern_embeddings import VinternEmbedder


def chunk_to_document(chunk: Chunk) -> Document:
    return Document(page_content=chunk.text, metadata={**chunk.metadata, "chunk_id": chunk.id})


class BM25LRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapping our Vietnamese word-segmented BM25L index."""

    bm25: Any
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = self.bm25.search(query, top_k=self.top_k)
        return [chunk_to_document(chunk) for chunk, _score in results]

    @classmethod
    def from_chunks(cls, chunks: List[Chunk], top_k: int = 5, **bm25_kwargs) -> "BM25LRetriever":
        bm25 = BM25Retriever(**bm25_kwargs)
        bm25.index_chunks(chunks)
        return cls(bm25=bm25, top_k=top_k)


class VinternDenseRetriever(BaseRetriever):
    """
    LangChain BaseRetriever wrapping Vintern-Embedding-1B dense retrieval
    (one mean-pooled vector per chunk, cosine similarity via VectorStore).
    """

    embedder: Any
    store: Any
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        query_vec = self.embedder.embed_query(query)
        results = self.store.similarity_search(query_vec, top_k=self.top_k)
        return [chunk_to_document(chunk) for chunk, _score in results]

    @classmethod
    def from_chunks(cls, chunks: List[Chunk], embedder: VinternEmbedder, top_k: int = 5) -> "VinternDenseRetriever":
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)
        store = VectorStore()
        store.add_chunks(chunks, embeddings)
        return cls(embedder=embedder, store=store, top_k=top_k)


class GraphRetriever(BaseRetriever):
    """
    LangChain BaseRetriever surfacing chunks connected to entities mentioned
    in the query via the knowledge graph (see app.rag.graph_store) - fuses
    graph-based retrieval into the same RRF ensemble as BM25L/dense search.
    """

    graph_store: Any
    chunk_by_id: Any
    top_k: int = 5
    hops: int = 1

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        chunk_ids = self.graph_store.related_chunk_ids(query, hops=self.hops)
        docs = [
            chunk_to_document(self.chunk_by_id[chunk_id]) for chunk_id in chunk_ids if chunk_id in self.chunk_by_id
        ]
        return docs[: self.top_k]

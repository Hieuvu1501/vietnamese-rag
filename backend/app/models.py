from typing import List, Optional

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    uploaded_at: str
    status: str
    num_chunks: int
    error: Optional[str] = None


class ChatRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    chunk_id: str
    snippet: str
    source_document: Optional[str] = None
    rerank_score: Optional[float] = None
    rrf_rank: Optional[int] = None  # position after RRF fusion, before reranking


class TriadScores(BaseModel):
    context_relevance: float
    groundedness: Optional[float] = None
    answer_relevance: Optional[float] = None
    composite: Optional[float] = None


class Timings(BaseModel):
    retrieval_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    graph_lookup_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    triad_eval_ms: Optional[float] = None
    total_ms: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    triad_scores: Optional[TriadScores] = None
    timings: Optional[Timings] = None
    graph_facts: Optional[List[str]] = None  # "subject predicate object" strings, only when ENABLE_GRAPH_RAG is on


class GraphNode(BaseModel):
    id: str  # normalized entity name, same key the retriever matches on
    label: str
    degree: int
    document_ids: List[int]


class GraphEdge(BaseModel):
    source: str
    target: str
    predicate: str
    count: int  # identical (subject, predicate, object) facts merged, e.g. from a re-uploaded document
    document_ids: List[int]


class GraphDocument(BaseModel):
    id: int
    filename: str


class GraphOut(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    documents: List[GraphDocument]


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: str
    top_chunk_ids: Optional[List[str]] = None
    triad_scores: Optional[TriadScores] = None
    timings: Optional[Timings] = None
    graph_facts: Optional[List[str]] = None

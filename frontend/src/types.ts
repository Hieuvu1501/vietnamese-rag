export interface DocumentOut {
  id: number
  filename: string
  file_type: string
  uploaded_at: string
  status: 'processing' | 'indexed' | 'failed'
  num_chunks: number
  error: string | null
}

export interface SourceChunk {
  chunk_id: string
  snippet: string
  source_document: string | null
  rerank_score: number | null
  rrf_rank: number | null
}

export interface TriadScores {
  context_relevance: number
  // null when no answer was generated (e.g. Gemini quota exhausted) - these
  // need an answer to score, unlike context_relevance.
  groundedness: number | null
  answer_relevance: number | null
  composite: number | null
}

export interface Timings {
  retrieval_ms: number | null
  rerank_ms: number | null
  graph_lookup_ms: number | null
  generation_ms: number | null
  triad_eval_ms: number | null
  total_ms: number | null
}

export interface ChatResponse {
  answer: string
  sources: SourceChunk[]
  triad_scores: TriadScores | null
  timings: Timings | null
  // "subject predicate object" strings from the knowledge graph, only
  // populated when the backend's ENABLE_GRAPH_RAG is on.
  graph_facts: string[] | null
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: SourceChunk[]
  triad_scores?: TriadScores | null
  timings?: Timings | null
  graph_facts?: string[] | null
  pending?: boolean
}

export interface GraphNode {
  id: string
  label: string
  degree: number
  document_ids: number[]
}

export interface GraphEdge {
  source: string
  target: string
  predicate: string
  count: number
  document_ids: number[]
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  documents: { id: number; filename: string }[]
}

export interface MessageOut {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
  // History only stores which chunk IDs were used, not full snippets/scores -
  // so historical messages can't show a "sources" breakdown the way a live
  // answer can, only that sources existed.
  top_chunk_ids: string[] | null
  triad_scores: TriadScores | null
  timings: Timings | null
  graph_facts: string[] | null
}

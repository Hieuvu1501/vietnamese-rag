import logging
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever

from app import config
from app.ingestion.normalize import nfc_normalize
from app.rag.bm25_retriever import BM25Retriever
from app.rag.chunker import Chunk
from app.rag.evaluator import TriadResult
from app.rag.graph_store import GraphExtractor, GraphStore, Triple
from app.rag.langchain_retrievers import BM25LRetriever, GraphRetriever, VinternDenseRetriever
from app.rag.local_reranker import LocalCrossEncoderReranker
from app.rag.vector_store import VectorStore
from app.rag.vintern_embeddings import VinternEmbedder

try:
    from google import genai
except ImportError:
    genai = None

try:
    from google.genai.errors import ClientError as GeminiClientError
except ImportError:
    GeminiClientError = None

log = logging.getLogger("knowledge_base")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _describe_generation_error(e: Exception) -> str:
    if GeminiClientError is not None and isinstance(e, GeminiClientError) and getattr(e, "code", None) == 429:
        return (
            "No answer was generated: the Gemini API quota is exhausted (free tier caps requests per day). "
            "The retrieved context below is still shown. Wait for the quota to reset, upgrade your Gemini API "
            "billing plan, or set ENABLE_TRIAD_EVAL=false in backend/.env to use fewer requests per question."
        )
    return f"No answer was generated: Gemini API error ({e})."


class KnowledgeBase:
    """
    Incremental version of RAG-full's RAGFullPipeline: instead of a one-shot
    index built from a single file, documents can be added one at a time
    (as they're uploaded) and the retrieval indexes grow incrementally.

    - BM25 has corpus-global IDF, so it's fully rebuilt from the in-memory
      chunk list on every add (cheap: CPU, in-memory, no I/O).
    - The Vintern dense store (VectorStore) supports true incremental
      append, so it's never rebuilt except at startup (`rebuild_from_db`).
    """

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured. Please set it in backend/.env")
        if genai is None:
            raise ImportError("Please install `google-genai` first.")

        self.embedder = VinternEmbedder()
        self.reranker = LocalCrossEncoderReranker(
            model_name=config.RERANKER_MODEL, device=config.RERANKER_DEVICE, max_length=config.RERANKER_MAX_LENGTH
        )
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        # None when GraphRAG is disabled - no extraction calls, no graph retriever.
        self.graph_store: Optional[GraphStore] = GraphStore() if config.ENABLE_GRAPH_RAG else None
        self.graph_extractor: Optional[GraphExtractor] = (
            GraphExtractor(self.client, config.LLM_MODEL) if config.ENABLE_GRAPH_RAG else None
        )

        self.all_chunks: List[Chunk] = []
        self._chunk_by_id: Dict[str, Chunk] = {}

        self.bm25 = BM25Retriever()
        self.store = VectorStore()
        self._wire_retrievers()

        # Vintern-Embedding-1B runs on MPS/CUDA, which isn't safe to hit from
        # multiple threads at once (FastAPI runs sync routes in a threadpool,
        # so concurrent requests are otherwise possible) - serialize all
        # index mutation and model inference through one lock.
        self._lock = threading.RLock()

    def _wire_retrievers(self) -> None:
        self.bm25_retriever = BM25LRetriever(bm25=self.bm25, top_k=config.TOP_K_SPARSE)
        self.dense_retriever = VinternDenseRetriever(embedder=self.embedder, store=self.store, top_k=config.TOP_K_DENSE)
        retrievers: List[Any] = [self.bm25_retriever, self.dense_retriever]
        weights = [config.RRF_WEIGHT_SPARSE, config.RRF_WEIGHT_DENSE]

        if self.graph_store is not None:
            self.graph_retriever = GraphRetriever(
                graph_store=self.graph_store,
                chunk_by_id=self._chunk_by_id,
                top_k=config.TOP_K_GRAPH,
                hops=config.GRAPH_HOPS,
            )
            retrievers.append(self.graph_retriever)
            weights.append(config.RRF_WEIGHT_GRAPH)

        self.ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)

    # -- indexing ---------------------------------------------------------

    def add_document(self, chunks: List[Chunk]) -> None:
        """
        Embeds and indexes new chunks. Embedding runs in fixed-size batches
        (config.EMBED_BATCH_SIZE), not all chunks in one call - a single
        unbatched call means peak memory for the forward pass grows with
        corpus size (this bit us directly: 132 chunks embedded fine, but
        rebuild_from_db() re-embedding a 349-chunk corpus in one shot pushed
        an already memory-constrained machine into an unrecoverable stall).
        Batching bounds peak memory to one batch's worth regardless of how
        large the total corpus gets.
        """
        if not chunks:
            return
        with self._lock:
            self.all_chunks.extend(chunks)
            self._chunk_by_id.update({c.id: c for c in chunks})

            self.bm25.index_chunks(self.all_chunks)

            batch_size = config.EMBED_BATCH_SIZE
            num_batches = (len(chunks) + batch_size - 1) // batch_size
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                log.info("Embedding batch %d/%d (%d chunks)...", i // batch_size + 1, num_batches, len(batch))
                embeddings = self.embedder.embed_texts([c.text for c in batch])
                self.store.add_chunks(batch, embeddings)

    def add_graph_triples(self, chunks: List[Chunk]) -> List[Triple]:
        """
        Extracts (subject, predicate, object) triples from new chunks via
        Gemini, batched (config.GRAPH_EXTRACT_BATCH_SIZE) to bound both API
        calls and prompt size, and adds them to the in-memory graph. Returns
        the extracted triples so the caller can persist them to SQLite.
        """
        if not chunks or self.graph_extractor is None or self.graph_store is None:
            return []
        with self._lock:
            all_triples: List[Triple] = []
            batch_size = config.GRAPH_EXTRACT_BATCH_SIZE
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                triples = self.graph_extractor.extract_batch(batch)
                self.graph_store.add_triples(triples)
                all_triples.extend(triples)
            return all_triples

    def rebuild_from_db(self) -> None:
        """Recomputes every index from SQLite-stored chunks and triples (server startup, or after a delete)."""
        import json

        from app import db

        with self._lock:
            self.all_chunks = []
            self._chunk_by_id = {}
            self.bm25 = BM25Retriever()
            self.store = VectorStore()
            # Graph rebuild replays already-extracted triples from SQLite - no Gemini calls.
            self.graph_store = GraphStore() if config.ENABLE_GRAPH_RAG else None
            self._wire_retrievers()

            rows = db.all_chunks_raw()
            chunks = [
                Chunk(id=row["id"], text=row["text"], metadata=json.loads(row["metadata_json"]))
                for row in rows
            ]
            if chunks:
                self.add_document(chunks)

            num_triples = 0
            if self.graph_store is not None:
                triple_rows = db.all_triples_raw()
                self.graph_store.add_triples(
                    [
                        Triple(subject=r["subject"], predicate=r["predicate"], object=r["object"], chunk_id=r["chunk_id"])
                        for r in triple_rows
                    ]
                )
                num_triples = len(triple_rows)

            log.info("Rebuilt knowledge base: %d chunks, %d graph triples", len(chunks), num_triples)

    # -- retrieval + generation --------------------------------------------

    def retrieve(self, query: str) -> Dict[str, Any]:
        query = nfc_normalize(query)
        timings: Dict[str, float] = {}

        with self._lock:
            t0 = time.perf_counter()
            fused_docs = self.ensemble_retriever.invoke(query) if self.all_chunks else []
            timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

            candidate_pairs = [
                (self._chunk_by_id[doc.metadata["chunk_id"]], 1.0 / (rank + 1))
                for rank, doc in enumerate(fused_docs)
                if doc.metadata["chunk_id"] in self._chunk_by_id
            ]
            t0 = time.perf_counter()
            reranked = (
                self.reranker.rerank(query, candidate_pairs, top_n=config.TOP_N_RERANK) if candidate_pairs else []
            )
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
            top_chunks = [r.chunk for r in reranked]

            top_triples: List[Triple] = []
            if self.graph_store is not None:
                t0 = time.perf_counter()
                top_triples = self.graph_store.related_triples(
                    query, hops=config.GRAPH_HOPS, max_triples=config.GRAPH_MAX_TRIPLES_IN_PROMPT
                )
                timings["graph_lookup_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "query": query,
            "reranked_results": reranked,
            "top_chunks": top_chunks,
            "top_triples": top_triples,
            "timings": timings,
        }

    def generate_answer(self, query: str, top_chunks: List[Chunk], top_triples: Optional[List[Triple]] = None) -> str:
        """The one Gemini call per question - everything else (rerank, triad eval) runs locally."""
        if not top_chunks and not top_triples:
            return "I could not find any relevant context to answer your question."

        context_blocks = [f"[Text Chunk {i} - ID: {c.id}]\n{c.text}" for i, c in enumerate(top_chunks, start=1)]
        context_str = "\n\n".join(context_blocks) if context_blocks else "(no text context retrieved)"

        graph_note = ""
        if top_triples:
            facts = "\n".join(f"- {t.subject} {t.predicate} {t.object}" for t in top_triples)
            graph_note = f"\n\nKnowledge graph facts about entities mentioned in the question:\n{facts}"

        prompt = f"""You are an accurate, helpful AI assistant. Answer the user's question based strictly on the provided context (text and/or knowledge graph facts). If the answer cannot be found in the context, explicitly state that the information is not present. Do not fabricate facts.

Text Context:
{context_str}{graph_note}

User Question:
{query}

Grounded Answer:"""

        response = self.client.models.generate_content(model=config.LLM_MODEL, contents=prompt)
        text_parts = []
        if hasattr(response, "candidates") and response.candidates:
            for cand in response.candidates:
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if getattr(part, "text", None):
                            text_parts.append(part.text)
        return "".join(text_parts).strip() if text_parts else response.text.strip()

    def _evaluate_triad_locally(
        self, query: str, reranked: list, top_chunks: List[Chunk], answer: Optional[str]
    ) -> Optional[TriadResult]:
        """
        RAG Triad scoring with zero extra API calls, reusing the same local
        cross-encoder used for reranking instead of an LLM judge:
        - context_relevance: the rerank_score already computed for each top
          chunk against the query IS a query-passage relevance judgment -
          just averaged, no recomputation needed. Needs no answer, so it's
          always available once there are retrieved chunks - including when
          answer generation itself failed (e.g. Gemini quota exhausted).
        - groundedness: the cross-encoder scores (context, answer) as a pair
          - a proxy for "does this context support this answer", not a true
            claim-by-claim entailment check the way an LLM judge would do.
        - answer_relevance: the cross-encoder scores (query, answer) directly
          - exactly the kind of relevance judgment it was trained for.

        Trade-off: a retrieval cross-encoder wasn't trained for NLI-style
        entailment or answer-quality judgment, so groundedness/answer_relevance
        here are heuristic proxies, less semantically rigorous than an LLM
        reading and reasoning about the claims. context_relevance is solid
        since it's literally what the model was trained to score.
        """
        if not top_chunks:
            return None

        context_relevance = _clamp01(sum(r.rerank_score for r in reranked) / len(reranked)) if reranked else 0.5

        if not answer:
            return TriadResult(context_relevance=context_relevance)

        context_str = "\n\n".join(c.text for c in top_chunks)
        pair_scores = self.reranker.score_pairs([(context_str, answer), (query, answer)])
        if pair_scores is None:
            return TriadResult(context_relevance=context_relevance)
        groundedness, answer_relevance = pair_scores
        groundedness = _clamp01(groundedness)
        answer_relevance = _clamp01(answer_relevance)
        composite = _clamp01((context_relevance + groundedness + answer_relevance) / 3.0)

        return TriadResult(
            context_relevance=context_relevance,
            groundedness=groundedness,
            answer_relevance=answer_relevance,
            composite=composite,
        )

    def query(self, question: str) -> Dict[str, Any]:
        t_start = time.perf_counter()
        retrieval_data = self.retrieve(question)
        timings = retrieval_data["timings"]

        t0 = time.perf_counter()
        try:
            answer = self.generate_answer(
                question, retrieval_data["top_chunks"], retrieval_data["top_triples"]
            )
            generation_failed = False
        except Exception as e:
            answer = _describe_generation_error(e)
            generation_failed = True
        timings["generation_ms"] = (time.perf_counter() - t0) * 1000

        triad_scores = None
        if config.ENABLE_TRIAD_EVAL:
            t0 = time.perf_counter()
            with self._lock:
                triad_scores = self._evaluate_triad_locally(
                    question,
                    retrieval_data["reranked_results"],
                    retrieval_data["top_chunks"],
                    answer if not generation_failed else None,
                )
            timings["triad_eval_ms"] = (time.perf_counter() - t0) * 1000

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        return {**retrieval_data, "answer": answer, "triad_scores": triad_scores}


knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global knowledge_base
    if knowledge_base is None:
        knowledge_base = KnowledgeBase()
    return knowledge_base

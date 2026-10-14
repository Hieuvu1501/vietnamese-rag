import json

from fastapi import APIRouter

from app import db
from app.knowledge_base import get_knowledge_base
from app.models import ChatRequest, ChatResponse, MessageOut, SourceChunk, Timings, TriadScores

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    kb = get_knowledge_base()
    # kb.query() never raises for a failed Gemini call (e.g. quota exhausted) -
    # it degrades gracefully instead: `answer` becomes an explanatory message,
    # but `sources` and `triad_scores.context_relevance` are still populated
    # from local retrieval/reranking, which don't depend on Gemini at all.
    result = kb.query(req.question)

    sources = [
        SourceChunk(
            chunk_id=r.chunk.id,
            snippet=r.chunk.text[:280],
            source_document=r.chunk.metadata.get("filename"),
            rerank_score=r.rerank_score,
            rrf_rank=r.rrf_rank,
        )
        for r in result["reranked_results"]
    ]

    triad = result.get("triad_scores")
    triad_scores = TriadScores(**triad.to_dict()) if triad is not None else None

    timings_dict = result.get("timings")
    timings = Timings(**timings_dict) if timings_dict else None

    graph_facts = [f"{t.subject} {t.predicate} {t.object}" for t in result.get("top_triples") or []] or None

    db.insert_message(role="user", content=req.question)
    db.insert_message(
        role="assistant",
        content=result["answer"],
        top_chunk_ids=[s.chunk_id for s in sources],
        triad=triad.to_dict() if triad is not None else None,
        timings=timings_dict,
        graph_facts=graph_facts,
    )

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        triad_scores=triad_scores,
        timings=timings,
        graph_facts=graph_facts,
    )


@router.get("/chat/history", response_model=list[MessageOut])
def chat_history():
    rows = db.list_messages()
    out = []
    for row in rows:
        triad = json.loads(row["triad_json"]) if row["triad_json"] else None
        row_timings = json.loads(row["timings_json"]) if row["timings_json"] else None
        out.append(
            MessageOut(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
                top_chunk_ids=json.loads(row["top_chunk_ids_json"]) if row["top_chunk_ids_json"] else None,
                triad_scores=TriadScores(**triad) if triad else None,
                timings=Timings(**row_timings) if row_timings else None,
                graph_facts=json.loads(row["graph_facts_json"]) if row["graph_facts_json"] else None,
            )
        )
    return out

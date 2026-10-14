from typing import Any, Dict, Tuple

from fastapi import APIRouter

from app import db
from app.models import GraphOut
from app.rag.graph_store import normalize_entity

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("", response_model=GraphOut)
def get_graph():
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    documents: Dict[int, str] = {}

    for row in db.all_triples_raw():
        s_key, o_key = normalize_entity(row["subject"]), normalize_entity(row["object"])
        if not s_key or not o_key:
            continue
        doc_id = row["document_id"]
        documents[doc_id] = row["filename"]

        for key, label in ((s_key, row["subject"]), (o_key, row["object"])):
            node = nodes.setdefault(key, {"id": key, "label": label, "degree": 0, "document_ids": set()})
            node["document_ids"].add(doc_id)

        edge = edges.setdefault(
            (s_key, normalize_entity(row["predicate"]), o_key),
            {"source": s_key, "target": o_key, "predicate": row["predicate"], "count": 0, "document_ids": set()},
        )
        edge["count"] += 1
        edge["document_ids"].add(doc_id)

    for edge in edges.values():
        nodes[edge["source"]]["degree"] += 1
        nodes[edge["target"]]["degree"] += 1

    return {
        "nodes": [{**n, "document_ids": sorted(n["document_ids"])} for n in nodes.values()],
        "edges": [{**e, "document_ids": sorted(e["document_ids"])} for e in edges.values()],
        "documents": [{"id": doc_id, "filename": name} for doc_id, name in sorted(documents.items())],
    }

"""
Lightweight GraphRAG: extracts (subject, predicate, object) triples from
chunks via Gemini at indexing time and answers relationship-style queries
("how is X connected to Y") by matching entities mentioned in the query
against the graph and walking a few hops of neighbors. The graph itself is
just an in-memory NetworkX graph rebuilt from SQLite-stored triples on
startup (see KnowledgeBase.rebuild_from_db) - no separate graph database.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx

from app.rag.chunker import Chunk

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract factual relationships (subject, predicate, object) from each numbered passage below. Only extract relationships that are explicitly stated - do not infer or guess. Keep subject/object as short entity names (people, places, organizations, concepts), and predicate as a short verb phrase. Replace pronouns (he, she, it, they, ông, bà, anh, chị, nó, họ...) with the full name of the entity they refer to, and always use the same name for the same entity. Write entity names and predicates in the passage's own language - do not translate.

{passages}

Respond ONLY with valid JSON, a list of objects, one per relationship found, each with keys "passage" (the passage number), "subject", "predicate", "object". If a passage has no clear relationships, omit it. Example:
[{{"passage": 1, "subject": "Nguyễn Huệ", "predicate": "thành lập", "object": "nhà Tây Sơn"}}]"""


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    chunk_id: str


def _safe_llm_text(client, model: str, prompt: str) -> str:
    response = client.models.generate_content(model=model, contents=prompt)
    parts = []
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        parts.append(part.text)
    return "".join(parts).strip() if parts else response.text.strip()


def _parse_json_array(text: str):
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, list) else None


class GraphExtractor:
    """Extracts triples from chunks via Gemini, batched to bound API calls and prompt size."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def extract_batch(self, chunks: List[Chunk]) -> List[Triple]:
        if not chunks:
            return []
        passages = "\n\n".join(f"[{i + 1}] {c.text[:1500]}" for i, c in enumerate(chunks))
        prompt = EXTRACTION_PROMPT.format(passages=passages)
        try:
            raw = _safe_llm_text(self.client, self.model, prompt)
            data = _parse_json_array(raw)
        except Exception:
            logger.warning("Graph triple extraction call failed; skipping this batch.")
            return []
        if not data:
            return []

        triples: List[Triple] = []
        for item in data:
            try:
                idx = int(item["passage"]) - 1
                subject = str(item["subject"]).strip()
                predicate = str(item["predicate"]).strip()
                obj = str(item["object"]).strip()
            except (KeyError, ValueError, TypeError):
                continue
            if not (0 <= idx < len(chunks)) or not subject or not predicate or not obj:
                continue
            triples.append(Triple(subject=subject, predicate=predicate, object=obj, chunk_id=chunks[idx].id))
        return triples


def normalize_entity(name: str) -> str:
    return " ".join(name.strip().lower().split())


class GraphStore:
    """
    In-memory knowledge graph built from extracted triples. Entities are
    graph nodes (keyed by normalized name); each edge carries the predicate
    and the chunk_id it was extracted from, so a matched entity/relationship
    can be traced back to retrievable text.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_chunk_ids: Dict[str, Set[str]] = {}
        self.entity_display_name: Dict[str, str] = {}

    def add_triples(self, triples: List[Triple]) -> None:
        for t in triples:
            s_key, o_key = normalize_entity(t.subject), normalize_entity(t.object)
            if not s_key or not o_key:
                continue
            self.entity_display_name.setdefault(s_key, t.subject)
            self.entity_display_name.setdefault(o_key, t.object)
            self.graph.add_edge(s_key, o_key, predicate=t.predicate, chunk_id=t.chunk_id)
            self.entity_chunk_ids.setdefault(s_key, set()).add(t.chunk_id)
            self.entity_chunk_ids.setdefault(o_key, set()).add(t.chunk_id)

    def _match_seed_entities(self, query: str, max_seeds: int = 5) -> List[str]:
        """Finds known entities mentioned in the query via case-insensitive substring match."""
        query_norm = normalize_entity(query)
        if not query_norm:
            return []
        # len >= 3 filters out single-syllable/generic keys that would match almost any query.
        matches = [key for key in self.entity_display_name if len(key) >= 3 and key in query_norm]
        matches.sort(key=len, reverse=True)  # prefer longer, more specific entity names first
        return matches[:max_seeds]

    def related_chunk_ids(self, query: str, hops: int = 1, max_seeds: int = 5) -> Set[str]:
        chunk_ids: Set[str] = set()
        undirected = self.graph.to_undirected(as_view=True)
        for seed in self._match_seed_entities(query, max_seeds=max_seeds):
            if seed not in self.graph:
                continue
            neighborhood = nx.ego_graph(undirected, seed, radius=hops)
            for node in neighborhood.nodes:
                chunk_ids |= self.entity_chunk_ids.get(node, set())
        return chunk_ids

    def related_triples(self, query: str, hops: int = 1, max_seeds: int = 5, max_triples: int = 8) -> List[Triple]:
        seeds = self._match_seed_entities(query, max_seeds=max_seeds)
        if not seeds:
            return []
        undirected = self.graph.to_undirected(as_view=True)
        seen_nodes: Set[str] = set()
        for seed in seeds:
            if seed in self.graph:
                seen_nodes |= set(nx.ego_graph(undirected, seed, radius=hops).nodes)

        triples: List[Triple] = []
        for u, v, data in self.graph.edges(data=True):
            if u in seen_nodes and v in seen_nodes:
                triples.append(
                    Triple(
                        subject=self.entity_display_name.get(u, u),
                        predicate=data["predicate"],
                        object=self.entity_display_name.get(v, v),
                        chunk_id=data["chunk_id"],
                    )
                )
                if len(triples) >= max_triples:
                    break
        return triples

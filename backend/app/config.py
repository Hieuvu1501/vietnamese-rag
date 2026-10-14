import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (BACKEND_DIR.parent / "data").resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "rag.db"
FRONTEND_DIST_DIR = (BACKEND_DIR.parent / "frontend" / "dist").resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")

TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "5"))
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "5"))
TOP_N_RERANK = int(os.getenv("TOP_N_RERANK", "3"))
RRF_WEIGHT_SPARSE = float(os.getenv("RRF_WEIGHT_SPARSE", "0.5"))
RRF_WEIGHT_DENSE = float(os.getenv("RRF_WEIGHT_DENSE", "0.5"))
ENABLE_TRIAD_EVAL = os.getenv("ENABLE_TRIAD_EVAL", "true").lower() not in ("false", "0", "no")

# Reranking runs locally (no API call) via a sentence-transformers cross-encoder.
# itdainb/PhoRanker: Vietnamese-specific, ~100M params (~5.6x smaller than
# bge-reranker-v2-m3's 568M) and scores higher on Vietnamese benchmarks - both
# smaller and better for this project, not a size/quality trade-off.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "itdainb/PhoRanker")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "256"))

# GraphRAG (off by default): extracts (subject, predicate, object) triples
# from chunks via Gemini at indexing time, building a small in-memory graph
# for relationship-style queries ("how is X connected to Y"). This adds a
# real Gemini API cost at upload time (batched, GRAPH_EXTRACT_BATCH_SIZE
# chunks per call) - a cost this project otherwise only pays per question.
ENABLE_GRAPH_RAG = os.getenv("ENABLE_GRAPH_RAG", "false").lower() not in ("false", "0", "no")
GRAPH_EXTRACT_BATCH_SIZE = int(os.getenv("GRAPH_EXTRACT_BATCH_SIZE", "8"))
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "1"))
TOP_K_GRAPH = int(os.getenv("TOP_K_GRAPH", "5"))
GRAPH_MAX_TRIPLES_IN_PROMPT = int(os.getenv("GRAPH_MAX_TRIPLES_IN_PROMPT", "8"))
# Weight for the graph retriever in the RRF ensemble - only added when
# ENABLE_GRAPH_RAG is on. RRF_WEIGHT_SPARSE/RRF_WEIGHT_DENSE above don't need
# adjusting for correctness, but for interpretability aim for all enabled
# weights to sum to ~1.0 (e.g. 0.4/0.4/0.2).
RRF_WEIGHT_GRAPH = float(os.getenv("RRF_WEIGHT_GRAPH", "0.2"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

# Chunks per Vintern embedding forward pass. Bounds peak memory for a single
# call regardless of total corpus size - matters most for rebuild_from_db(),
# which otherwise embeds the entire stored corpus in one unbatched call.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

ALLOWED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".docx", ".pdf"}

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

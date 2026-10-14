import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config, db
from app.knowledge_base import get_knowledge_base
from app.routers import chat, documents, graph

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Vietnamese RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (documents.router, chat.router, graph.router):
    app.include_router(router)
    # Same routes under /api so the built frontend (which calls /api/*) works
    # when served by this app; the Vite dev server strips /api via its proxy.
    app.include_router(router, prefix="/api", include_in_schema=False)


@app.on_event("startup")
def on_startup():
    db.init_db()
    db.fail_orphaned_processing_documents()
    get_knowledge_base().rebuild_from_db()


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the built React app (`npm run build` in frontend/) at the root. Mounted
# last so every API route above takes precedence.
if config.FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIST_DIR), html=True), name="frontend")

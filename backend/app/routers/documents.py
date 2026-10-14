import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app import config, db
from app.ingestion.extractors import UnsupportedFileTypeError, extract
from app.knowledge_base import get_knowledge_base
from app.models import DocumentOut
from app.rag.chunker import Chunk, TextChunker

router = APIRouter(prefix="/documents", tags=["documents"])

chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)


@router.get("", response_model=list[DocumentOut])
def list_documents():
    return db.list_documents()


@router.post("", response_model=DocumentOut)
async def upload_document(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    document_id = db.create_document(filename=file.filename, file_type=ext.lstrip("."))
    upload_dir = config.UPLOADS_DIR / str(document_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    src_path = upload_dir / f"original{ext}"

    try:
        with src_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        result = extract(src_path)

        chunks: list[Chunk] = []
        for page in result.pages:
            source = f"doc{document_id}" if page.page_number is None else f"doc{document_id}_p{page.page_number}"
            extra_metadata = {"document_id": document_id, "filename": file.filename, "page_number": page.page_number}
            chunks.extend(chunker.chunk_text(page.text, source=source, extra_metadata=extra_metadata))

        if chunks:
            db.insert_chunks(document_id, chunks)

        kb = get_knowledge_base()
        kb.add_document(chunks)

        triples = kb.add_graph_triples(chunks)
        if triples:
            db.insert_triples(document_id, triples)

        db.mark_document_done(document_id, num_chunks=len(chunks))
    except UnsupportedFileTypeError as e:
        db.mark_document_failed(document_id, str(e))
        raise HTTPException(400, str(e))
    except Exception as e:
        db.mark_document_failed(document_id, str(e))
        raise HTTPException(500, f"Failed to process {file.filename}: {e}")

    return next(d for d in db.list_documents() if d["id"] == document_id)


@router.delete("/{document_id}")
def delete_document(document_id: int):
    docs = db.list_documents()
    if not any(d["id"] == document_id for d in docs):
        raise HTTPException(404, "Document not found")

    upload_dir = config.UPLOADS_DIR / str(document_id)
    db.delete_document(document_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)

    get_knowledge_base().rebuild_from_db()
    return {"status": "deleted", "id": document_id}

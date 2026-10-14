import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        snippet = self.text[:60].replace("\n", " ") + ("..." if len(self.text) > 60 else "")
        return f"Chunk(id={self.id!r}, text={snippet!r})"


class TextChunker:
    """
    Splits text documents into manageable overlapping chunks.
    Preserves document structure (paragraphs, sections) and attaches consistent chunk IDs.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 60):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_text(
        self, text: str, source: str = "document", extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Splits text into chunks by paragraphs or sliding window, returning Chunk dataclasses.
        """
        text = text.strip()
        if not text:
            return []

        # Detect if file uses document variable lines (e.g. DOCUMENT1 = "...")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        has_assignments = any(re.match(r"^[A-Za-z0-9_]+\s*=\s*[\"']", l) for l in lines)

        if has_assignments:
            paragraphs = []
            for l in lines:
                m = re.match(r"^[A-Za-z0-9_]+\s*=\s*[\"'](.*)[\"']\s*$", l)
                if m:
                    paragraphs.append(m.group(1).strip())
                elif not re.match(r"^documents\s*=\s*\[", l):
                    paragraphs.append(l)
        else:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        raw_chunks: List[str] = self._splitter.split_text("\n\n".join(paragraphs))

        # Build Chunk objects with uniform IDs and metadata
        chunks: List[Chunk] = []
        base_meta = extra_metadata or {}
        for idx, chunk_str in enumerate(raw_chunks):
            chunk_id = f"{source}#chunk-{idx:04d}"
            meta = {
                **base_meta,
                "source": source,
                "chunk_index": idx,
                "char_length": len(chunk_str),
            }
            chunks.append(Chunk(id=chunk_id, text=chunk_str, metadata=meta))

        return chunks

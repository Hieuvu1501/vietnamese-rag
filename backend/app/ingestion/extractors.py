from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.ingestion.normalize import nfc_normalize


@dataclass
class PageText:
    page_number: Optional[int]
    text: str


@dataclass
class ExtractionResult:
    pages: List[PageText] = field(default_factory=list)


class UnsupportedFileTypeError(ValueError):
    pass


def extract(src_path: Path) -> ExtractionResult:
    ext = src_path.suffix.lower()

    if ext in (".txt", ".md"):
        return _extract_text(src_path)
    if ext in (".html", ".htm"):
        return _extract_html(src_path)
    if ext == ".docx":
        return _extract_docx(src_path)
    if ext == ".pdf":
        return _extract_pdf(src_path)
    if ext == ".doc":
        raise UnsupportedFileTypeError(
            "Legacy .doc (pre-2007 Word binary format) is not supported - please save as .docx and re-upload."
        )
    raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")


def _extract_text(src_path: Path) -> ExtractionResult:
    text = src_path.read_text(encoding="utf-8", errors="replace")
    return ExtractionResult(pages=[PageText(None, nfc_normalize(text))])


def _extract_html(src_path: Path) -> ExtractionResult:
    from bs4 import BeautifulSoup

    html = src_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return ExtractionResult(pages=[PageText(None, nfc_normalize(text))])


def _extract_docx(src_path: Path) -> ExtractionResult:
    import docx

    document = docx.Document(str(src_path))
    text = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
    return ExtractionResult(pages=[PageText(None, nfc_normalize(text))])


def _extract_pdf(src_path: Path) -> ExtractionResult:
    import pymupdf as fitz

    pages: List[PageText] = []
    doc = fitz.open(str(src_path))
    try:
        for page_index in range(doc.page_count):
            text = doc.load_page(page_index).get_text("text")
            if text.strip():
                pages.append(PageText(page_index + 1, nfc_normalize(text)))
    finally:
        doc.close()

    return ExtractionResult(pages=pages)

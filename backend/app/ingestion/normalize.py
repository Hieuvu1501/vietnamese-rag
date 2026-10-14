import unicodedata


def nfc_normalize(text: str) -> str:
    """
    Normalizes to NFC (precomposed) form. PDFs and other sources often emit
    Vietnamese diacritics as NFD (decomposed, e.g. base letter + combining
    mark); indexing/searching with mixed normalization silently breaks exact
    term matches (BM25) and query/document consistency (dense embeddings).
    Applied once at ingestion so every downstream consumer sees one form.
    """
    return unicodedata.normalize("NFC", text)

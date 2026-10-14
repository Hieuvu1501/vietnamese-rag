import re
import math
from typing import List, Tuple
from collections import Counter
from app.rag.chunker import Chunk

try:
    from rank_bm25 import BM25L
except ImportError:
    BM25L = None

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

# Unicode word boundary: Vietnamese diacritics (e.g. "ê", "ộ", "ữ") are outside
# the ASCII a-zA-Z range, so a plain [a-zA-Z0-9]+ tokenizer silently drops every
# Vietnamese word. \w in Python's `re` is Unicode-aware by default, which keeps
# diacritics intact.
_WORD_RE = re.compile(r"\w+(?:-\w+)*", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """
    Vietnamese-aware tokenizer for BM25L:
    1. Word-segments with pyvi (so "thành_phố" is one token, not "thành" + "phố" -
       Vietnamese words are often multi-syllable, and syllable-level tokens lose
       the meaning boundary BM25's term-overlap scoring relies on).
    2. Falls back to Unicode-aware syllable splitting if pyvi isn't installed.
    Also preserves hyphenated technical identifiers (e.g. AGC-1969, SA-506).
    """
    text = text.lower()
    if ViTokenizer is not None:
        segmented = ViTokenizer.tokenize(text)
        return _WORD_RE.findall(segmented)
    return _WORD_RE.findall(text)


class PurePythonBM25L:
    """
    Fallback pure-Python BM25L implementation in case `rank_bm25` is not
    installed. BM25L (Lv & Zhai, 2011) fixes BM25Okapi's bias against long
    documents via a length-normalized pseudo-frequency `ctd` and a free
    `delta` parameter.
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75, delta: float = 0.5):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / max(1, self.corpus_size)
        self.doc_freqs: List[Counter] = [Counter(doc) for doc in corpus]

        self.df: Counter = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1

        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(self.corpus_size + 1) - math.log(freq + 0.5)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for idx, doc_freq in enumerate(self.doc_freqs):
                tf = doc_freq.get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[idx]
                ctd = tf / (1 - self.b + self.b * (doc_len / max(1, self.avgdl)))
                scores[idx] += idf * (self.k1 + 1) * (ctd + self.delta) / (self.k1 + ctd + self.delta)
        return scores


class BM25Retriever:
    """
    Sparse Lexical Retriever using Vietnamese word-segmented BM25L indexing.
    Ideal for exact keyword matches, technical codes, dates, and proper nouns
    that dense embeddings can blur together.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 0.5):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None

    def index_chunks(self, chunks: List[Chunk]):
        self.chunks = list(chunks)
        self.tokenized_corpus = [tokenize(c.text) for c in chunks]

        if not self.chunks:
            self.bm25 = None
            return

        if BM25L is not None:
            self.bm25 = BM25L(self.tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
        else:
            self.bm25 = PurePythonBM25L(self.tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        k = min(top_k, len(self.chunks))
        results: List[Tuple[Chunk, float]] = []
        for idx, score in indexed_scores[:k]:
            if score > 0:
                results.append((self.chunks[idx], score))
        return results

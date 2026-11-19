from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from .sources import SearchResult

_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "if", "then", "else", "what", "who",
    "where", "when", "why", "how", "which", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
    "me", "my", "tell", "explain", "about", "please",
}


def tokenize(text: str) -> list[str]:
    """Extract lowercased alphanumeric words excluding common stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


@dataclass
class ScoredPassage:
    """A scored passage candidate."""

    passage: str
    source_result: SearchResult
    score: float


class BM25Reranker:
    """Okapi BM25 ranker for selecting the most relevant factual evidence."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def rank(self, query: str, results: Sequence[SearchResult]) -> list[ScoredPassage]:
        query_tokens = tokenize(query)
        if not query_tokens or not results:
            return [
                ScoredPassage(
                    passage=r.extract,
                    source_result=r,
                    score=0.1,
                )
                for r in results
            ]

        # Break results into sentence/short passage units
        passages: list[tuple[str, SearchResult]] = []
        for r in results:
            sentences = re.split(r"(?<=[.!?])\s+", r.extract.strip())
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) >= 20:
                    passages.append((s_clean, r))
            if not sentences or all(len(s.strip()) < 20 for s in sentences):
                passages.append((r.extract.strip(), r))

        if not passages:
            return []

        # Corpus stats
        n_docs = len(passages)
        tokenized_docs = [tokenize(p[0]) for p in passages]
        doc_lens = [len(td) for td in tokenized_docs]
        avg_doc_len = sum(doc_lens) / max(1, n_docs)

        # Document frequencies for query tokens
        df: dict[str, int] = {}
        for q in set(query_tokens):
            df[q] = sum(1 for td in tokenized_docs if q in td)

        scored: list[ScoredPassage] = []
        for (passage_text, src_result), doc_tokens, doc_len in zip(passages, tokenized_docs, doc_lens):
            score = 0.0
            if doc_len == 0:
                scored.append(ScoredPassage(passage_text, src_result, 0.0))
                continue

            for q in query_tokens:
                n_q = df.get(q, 0)
                # Smoothed Okapi BM25 IDF
                idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
                freq = doc_tokens.count(q)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, avg_doc_len)))
                term_score = idf * (freq * (self.k1 + 1.0)) / max(1e-6, denom)
                score += term_score

            scored.append(ScoredPassage(passage_text, src_result, score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

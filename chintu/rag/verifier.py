from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from .reranker import ScoredPassage, tokenize

logger = logging.getLogger("chintu.rag.verifier")

ABSTENTION_MESSAGE = "I couldn't find enough relevant information to answer that reliably."


@dataclass
class VerifiedAnswer:
    """The outcome of the grounding and verification gate."""

    answer: str
    confidence: float
    verified: bool
    source_title: str
    source_url: str


class GroundingVerifier:
    """Validate that answers are factually supported by retrieved evidence and formulate concise spoken responses."""

    def __init__(self, min_confidence: float = 0.25) -> None:
        self.min_confidence = min_confidence

    def verify_and_format(self, query: str, ranked_passages: Sequence[ScoredPassage]) -> VerifiedAnswer:
        if not ranked_passages:
            return VerifiedAnswer(
                answer=ABSTENTION_MESSAGE,
                confidence=0.0,
                verified=False,
                source_title="",
                source_url="",
            )

        top = ranked_passages[0]
        query_tokens = set(tokenize(query))

        # Check for direct Open-Meteo weather response
        if top.source_result.source == "open-meteo":
            return VerifiedAnswer(
                answer=top.passage,
                confidence=1.0,
                verified=True,
                source_title=top.source_result.title,
                source_url=top.source_result.url,
            )

        # Count query terms in the evidence itself, not just its title.
        passage_tokens = set(tokenize(top.passage))
        overlap = query_tokens.intersection(passage_tokens)

        # Coverage ratio of core query terms
        coverage = len(overlap) / max(1, len(query_tokens))

        # Scale BM25 score to a [0, 1] confidence range
        bm25_scaled = min(1.0, top.score / max(1.0, len(query_tokens) * 2.0))
        confidence = 0.5 * coverage + 0.5 * bm25_scaled

        if coverage < 0.6 or confidence < self.min_confidence:
            logger.info(
                "Answer failed grounding verification: coverage=%.2f confidence=%.2f (min=%.2f)",
                coverage,
                confidence,
                self.min_confidence,
            )
            return VerifiedAnswer(
                answer=ABSTENTION_MESSAGE,
                confidence=confidence,
                verified=False,
                source_title=top.source_result.title,
                source_url=top.source_result.url,
            )

        # Select up to 2-3 concise sentences from the evidence
        selected_sentences: list[str] = [top.passage]
        # Include a second highly scored sentence if from the same source
        for p in ranked_passages[1:3]:
            if p.source_result.url == top.source_result.url and p.score > 0.4:
                if p.passage not in selected_sentences and len(" ".join(selected_sentences)) < 280:
                    selected_sentences.append(p.passage)

        answer_text = " ".join(selected_sentences).strip()

        # Clean trailing references like [1], [2], [citation needed]
        answer_text = re.sub(r"\[\d+\]|\[citation needed\]", "", answer_text, flags=re.IGNORECASE)
        answer_text = re.sub(r"\s+", " ", answer_text).strip()

        return VerifiedAnswer(
            answer=answer_text,
            confidence=confidence,
            verified=True,
            source_title=top.source_result.title,
            source_url=top.source_result.url,
        )

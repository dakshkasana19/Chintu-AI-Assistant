from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import AssistantConfig
from .reranker import BM25Reranker
from .sources import DuckDuckGoClient, OpenMeteoClient, SearchResult, WikipediaClient
from .verifier import ABSTENTION_MESSAGE, GroundingVerifier, VerifiedAnswer

logger = logging.getLogger("chintu.rag.pipeline")


@dataclass
class RAGResult:
    """The complete result of an information retrieval and verification request."""

    query: str
    answer: str
    confidence: float
    verified: bool
    source_title: str
    source_url: str


class RAGPipeline:
    """Orchestrate keyless retrieval sources, BM25 reranking, and grounding verification."""

    def __init__(
        self,
        config: AssistantConfig | None = None,
        weather_client: OpenMeteoClient | None = None,
        wiki_client: WikipediaClient | None = None,
        ddg_client: DuckDuckGoClient | None = None,
        reranker: BM25Reranker | None = None,
        verifier: GroundingVerifier | None = None,
    ) -> None:
        cfg = config or AssistantConfig()
        timeout = cfg.rag_timeout_seconds
        max_results = cfg.rag_max_results
        min_confidence = cfg.rag_min_confidence

        self.weather_client = weather_client or OpenMeteoClient(timeout=timeout)
        self.wiki_client = wiki_client or WikipediaClient(timeout=timeout)
        self.ddg_client = ddg_client or DuckDuckGoClient(timeout=timeout, max_results=max_results)
        self.reranker = reranker or BM25Reranker()
        self.verifier = verifier or GroundingVerifier(min_confidence=min_confidence)

    def answer(self, query: str) -> RAGResult:
        """Process an information query, retrieve evidence, and verify the answer."""
        query_clean = query.strip()
        if not query_clean:
            return RAGResult(
                query=query,
                answer="Please ask a question.",
                confidence=0.0,
                verified=False,
                source_title="",
                source_url="",
            )

        # 1. Specialized weather flow
        if self.weather_client.is_weather_query(query_clean):
            location = self.weather_client.extract_location(query_clean)
            if not location:
                return RAGResult(
                    query=query_clean,
                    answer="Which location should I check the weather for?",
                    confidence=0.0,
                    verified=False,
                    source_title="",
                    source_url="",
                )
            weather_res = self.weather_client.fetch_weather(location)
            if weather_res is not None:
                return RAGResult(
                    query=query_clean,
                    answer=weather_res.extract,
                    confidence=1.0,
                    verified=True,
                    source_title=weather_res.title,
                    source_url=weather_res.url,
                )
            return RAGResult(
                query=query_clean,
                answer="I couldn't retrieve current weather for that location.",
                confidence=0.0,
                verified=False,
                source_title="",
                source_url="",
            )

        # 2. General encyclopedic and web retrieval
        results: list[SearchResult] = []

        # Query Wikipedia
        wiki_res = self.wiki_client.search_and_summarize(query_clean)
        if wiki_res is not None:
            results.append(wiki_res)

        # Query DuckDuckGo
        ddg_results = self.ddg_client.search(query_clean)
        results.extend(ddg_results)

        if not results:
            logger.info("No retrieval results returned")
            return RAGResult(
                query=query_clean,
                answer=ABSTENTION_MESSAGE,
                confidence=0.0,
                verified=False,
                source_title="",
                source_url="",
            )

        # 3. BM25 Reranking
        ranked_passages = self.reranker.rank(query_clean, results)

        # 4. Grounding verification
        verified_answer: VerifiedAnswer = self.verifier.verify_and_format(query_clean, ranked_passages)

        return RAGResult(
            query=query_clean,
            answer=verified_answer.answer,
            confidence=verified_answer.confidence,
            verified=verified_answer.verified,
            source_title=verified_answer.source_title,
            source_url=verified_answer.source_url,
        )

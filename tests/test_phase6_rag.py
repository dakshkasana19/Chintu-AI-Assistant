from __future__ import annotations

from unittest.mock import MagicMock, patch


from chintu.config import AssistantConfig
from chintu.rag import (
    ABSTENTION_MESSAGE,
    BM25Reranker,
    GroundingVerifier,
    OpenMeteoClient,
    RAGPipeline,
    RAGResult,
    ScoredPassage,
    SearchResult,
    WikipediaClient,
)
from chintu.state import AssistantState


def test_open_meteo_query_detection() -> None:
    assert OpenMeteoClient.is_weather_query("what is the weather in Mumbai")
    assert OpenMeteoClient.is_weather_query("temperature in Tokyo today")
    assert OpenMeteoClient.is_weather_query("how hot is it in London")
    assert not OpenMeteoClient.is_weather_query("who is the President of India")
    assert not OpenMeteoClient.is_weather_query("open youtube")


def test_open_meteo_location_extraction() -> None:
    assert OpenMeteoClient.extract_location("weather in London today").lower() == "london"
    assert OpenMeteoClient.extract_location("what is the temperature for Tokyo").lower() == "tokyo"
    assert OpenMeteoClient.extract_location("weather at New York right now").lower() == "new york"
    assert OpenMeteoClient.extract_location("what is the weather?") == ""
    assert OpenMeteoClient.extract_location("weather near me") == ""


def test_rag_pipeline_asks_for_location_instead_of_assuming_one() -> None:
    weather_client = MagicMock()
    weather_client.is_weather_query.return_value = True
    weather_client.extract_location.return_value = ""
    pipeline = RAGPipeline(weather_client=weather_client)

    result = pipeline.answer("what is the weather?")

    assert result.answer == "Which location should I check the weather for?"
    assert not result.verified
    weather_client.fetch_weather.assert_not_called()


def test_rag_pipeline_abstains_when_weather_source_fails() -> None:
    weather_client = MagicMock()
    weather_client.is_weather_query.return_value = True
    weather_client.extract_location.return_value = "Paris"
    weather_client.fetch_weather.return_value = None
    pipeline = RAGPipeline(weather_client=weather_client)

    result = pipeline.answer("weather in Paris")

    assert result.answer == "I couldn't retrieve current weather for that location."
    assert not result.verified


def test_open_meteo_rejects_incomplete_current_conditions() -> None:
    client = OpenMeteoClient()
    geo_response = MagicMock()
    geo_response.status_code = 200
    geo_response.json.return_value = {
        "results": [{"name": "Paris", "latitude": 48.85, "longitude": 2.35}]
    }
    weather_response = MagicMock()
    weather_response.status_code = 200
    weather_response.json.return_value = {"current": {}}

    with patch("requests.get", side_effect=[geo_response, weather_response]):
        assert client.fetch_weather("Paris") is None


def test_open_meteo_fetch_weather_mocked() -> None:
    client = OpenMeteoClient()

    geo_response = MagicMock()
    geo_response.status_code = 200
    geo_response.json.return_value = {
        "results": [{"name": "Paris", "latitude": 48.85, "longitude": 2.35, "country": "France"}]
    }

    weather_response = MagicMock()
    weather_response.status_code = 200
    weather_response.json.return_value = {
        "current": {
            "temperature_2m": 18.5,
            "apparent_temperature": 17.8,
            "relative_humidity_2m": 55,
            "weather_code": 1,
            "wind_speed_10m": 12.0,
        }
    }

    with patch("requests.get", side_effect=[geo_response, weather_response]):
        result = client.fetch_weather("Paris")

    assert result is not None
    assert "Paris" in result.title
    assert "18.5°C" in result.extract
    assert "mainly clear" in result.extract
    assert "open-meteo" in result.source


def test_wikipedia_client_mocked() -> None:
    client = WikipediaClient()

    search_resp = MagicMock()
    search_resp.status_code = 200
    search_resp.json.return_value = {
        "query": {"search": [{"title": "Python (programming language)"}]}
    }

    summary_resp = MagicMock()
    summary_resp.status_code = 200
    summary_resp.json.return_value = {
        "title": "Python (programming language)",
        "extract": "Python is a high-level, general-purpose programming language created by Guido van Rossum.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python_(programming_language)"}},
    }

    with patch("requests.get", side_effect=[search_resp, summary_resp]):
        result = client.search_and_summarize("who created Python programming")

    assert result is not None
    assert "Python" in result.title
    assert "Guido van Rossum" in result.extract
    assert result.source == "wikipedia"


def test_bm25_reranker_prioritizes_matching_terms() -> None:
    reranker = BM25Reranker()
    results = [
        SearchResult(
            title="Random Topic",
            extract="The quick brown fox jumps over the lazy dog in the park.",
            url="https://example.com/fox",
            source="test",
        ),
        SearchResult(
            title="French Capital",
            extract="Paris is the capital and most populous city of France.",
            url="https://example.com/paris",
            source="test",
        ),
    ]

    ranked = reranker.rank("what is the capital of France", results)

    assert len(ranked) >= 2
    assert "Paris" in ranked[0].passage
    assert ranked[0].score > ranked[-1].score


def test_grounding_verifier_accepts_good_coverage() -> None:
    verifier = GroundingVerifier(min_confidence=0.25)
    passage = ScoredPassage(
        passage="Paris is the capital of France and has a rich history.",
        source_result=SearchResult(
            title="Paris",
            extract="Paris is the capital of France.",
            url="https://example.com",
            source="wikipedia",
        ),
        score=2.5,
    )

    verified = verifier.verify_and_format("capital of France", [passage])

    assert verified.verified
    assert verified.confidence >= 0.25
    assert "Paris is the capital of France" in verified.answer


def test_grounding_verifier_abstains_on_poor_coverage() -> None:
    verifier = GroundingVerifier(min_confidence=0.5)
    passage = ScoredPassage(
        passage="Apples and oranges are delicious orchard fruits.",
        source_result=SearchResult(
            title="Fruit",
            extract="Apples and oranges are delicious orchard fruits.",
            url="https://example.com",
            source="test",
        ),
        score=0.1,
    )

    verified = verifier.verify_and_format("quantum computing superposition", [passage])

    assert not verified.verified
    assert verified.answer == ABSTENTION_MESSAGE


def test_grounding_verifier_does_not_count_title_only_overlap() -> None:
    verifier = GroundingVerifier(min_confidence=0.1)
    passage = ScoredPassage(
        passage="This page contains unrelated information about a topic.",
        source_result=SearchResult(
            title="Quantum computing and superposition",
            extract="This page contains unrelated information about a topic.",
            url="https://example.com",
            source="test",
        ),
        score=10.0,
    )

    verified = verifier.verify_and_format(
        "quantum computing superposition",
        [passage],
    )

    assert not verified.verified
    assert verified.answer == ABSTENTION_MESSAGE


def test_rag_pipeline_end_to_end_weather() -> None:
    weather_client = MagicMock()
    weather_client.is_weather_query.return_value = True
    weather_client.extract_location.return_value = "Berlin"
    weather_client.fetch_weather.return_value = SearchResult(
        title="Weather in Berlin",
        extract="The current weather in Berlin is clear with 20.0°C.",
        url="https://open-meteo.com",
        source="open-meteo",
    )

    pipeline = RAGPipeline(weather_client=weather_client)
    res = pipeline.answer("weather in Berlin")

    assert res.verified
    assert "Berlin is clear with 20.0°C" in res.answer
    assert res.confidence == 1.0


def test_rag_pipeline_abstains_when_no_sources() -> None:
    weather_client = MagicMock()
    weather_client.is_weather_query.return_value = False
    wiki_client = MagicMock()
    wiki_client.search_and_summarize.return_value = None
    ddg_client = MagicMock()
    ddg_client.search.return_value = []

    pipeline = RAGPipeline(
        weather_client=weather_client,
        wiki_client=wiki_client,
        ddg_client=ddg_client,
    )
    res = pipeline.answer("some obscure unknown query")

    assert not res.verified
    assert res.answer == ABSTENTION_MESSAGE


def test_runtime_includes_source_attribution_for_verified_answers() -> None:
    from types import SimpleNamespace

    from chintu_assistant import ChintuRuntime

    responses: list[str] = []

    class Signal:
        def emit(self, value: str) -> None:
            responses.append(value)

    bridge = SimpleNamespace(
        status_changed=Signal(),
        response_changed=Signal(),
    )
    runtime = ChintuRuntime(bridge, AssistantConfig(tts_enabled=False))
    runtime.state.transition(AssistantState.WAKE_CONFIRM)
    runtime.state.transition(AssistantState.LISTENING)
    runtime.state.transition(AssistantState.PROCESSING)
    runtime._rag = MagicMock()
    runtime._rag.answer.return_value = RAGResult(
        query="capital of France",
        answer="Paris is the capital of France.",
        confidence=0.9,
        verified=True,
        source_title="Paris - Wikipedia",
        source_url="https://en.wikipedia.org/wiki/Paris",
    )

    runtime._answer_information_request("capital of France")

    assert responses[-1] == "Paris is the capital of France. Source: Paris - Wikipedia."

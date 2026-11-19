from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger("chintu.rag.sources")

_USER_AGENT = "Chintu-AI-Assistant/1.0 (local voice assistant; contact: local)"

_WMO_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    62: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


@dataclass
class SearchResult:
    """A single retrieved evidence candidate."""

    title: str
    extract: str
    url: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class OpenMeteoClient:
    """Keyless public weather client using Open-Meteo geocoding and forecast APIs."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    @staticmethod
    def is_weather_query(query: str) -> bool:
        lower = query.lower()
        patterns = [
            r"\bweather\b",
            r"\btemperature\b",
            r"\bforecast\b",
            r"\bhow hot\b",
            r"\bhow cold\b",
            r"\bis it raining\b",
            r"\bwill it rain\b",
            r"\bhumidity in\b",
        ]
        return any(re.search(p, lower) for p in patterns)

    @staticmethod
    def extract_location(query: str) -> str:
        lower = query.lower()
        if re.search(r"\b(?:here|near me|my location)\b", lower):
            return ""
        # Look for "in <city>", "for <city>", "at <city>"
        match = re.search(r"\b(?:in|for|at|of)\s+([a-zA-Z\s]+?)(?:\s+(?:today|tomorrow|right now|currently))?[\?\.]*$", lower)
        if match:
            return match.group(1).strip()
        # Fallback: remove weather trigger words
        cleaned = re.sub(
            r"\b(?:what is the|what's the|tell me the|how is the|weather|temperature|forecast|today|tomorrow|currently|right now|in|for|at|please)\b",
            "",
            lower,
        ).strip(" \t\r\n?!.,")
        return cleaned

    def fetch_weather(self, location: str) -> SearchResult | None:
        try:
            # 1. Geocode location name
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote(location)}&count=1&language=en&format=json"
            geo_resp = requests.get(geo_url, headers={"User-Agent": _USER_AGENT}, timeout=self.timeout)
            if geo_resp.status_code != 200:
                logger.warning("Geocoding request failed: HTTP %d", geo_resp.status_code)
                return None

            geo_data = geo_resp.json()
            results = geo_data.get("results")
            if not results:
                logger.info("Geocoding request returned no matching location")
                return None

            place = results[0]
            lat = place["latitude"]
            lon = place["longitude"]
            place_name = place.get("name", location.title())
            country = place.get("country", "")

            # 2. Fetch current weather
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
            )
            weather_resp = requests.get(weather_url, headers={"User-Agent": _USER_AGENT}, timeout=self.timeout)
            if weather_resp.status_code != 200:
                logger.warning("Weather request failed: HTTP %d", weather_resp.status_code)
                return None

            weather_data = weather_resp.json()
            current = weather_data.get("current")
            if not isinstance(current, dict):
                logger.warning("Weather response is missing current conditions")
                return None
            try:
                temp = float(current["temperature_2m"])
                feels_like = float(current.get("apparent_temperature", temp))
                humidity = float(current["relative_humidity_2m"])
                wind = float(current["wind_speed_10m"])
                code = int(current["weather_code"])
            except (KeyError, TypeError, ValueError):
                logger.warning("Weather response contains invalid current conditions")
                return None
            if (
                not all(math.isfinite(value) for value in (temp, feels_like, humidity, wind))
                or not 0 <= humidity <= 100
                or wind < 0
            ):
                logger.warning("Weather response contains out-of-range current conditions")
                return None
            condition = _WMO_WEATHER_CODES.get(code, "conditions unavailable")

            location_display = f"{place_name}, {country}" if country else place_name
            extract = (
                f"The current weather in {location_display} is {condition} with a temperature of {temp:.1f}°C "
                f"(feels like {feels_like:.1f}°C), {humidity}% humidity, and wind speed of {wind:.1f} km/h."
            )

            return SearchResult(
                title=f"Weather in {location_display}",
                extract=extract,
                url=f"https://open-meteo.com/en/forecast?latitude={lat}&longitude={lon}",
                source="open-meteo",
                metadata={
                    "temperature": temp,
                    "condition": condition,
                    "humidity": humidity,
                    "wind": wind,
                    "location": location_display,
                },
            )
        except Exception as exc:
            logger.warning("Open-Meteo request failed (%s)", type(exc).__name__)
            return None


class WikipediaClient:
    """Keyless public encyclopedic client using the Wikipedia REST API."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def search_and_summarize(self, query: str) -> SearchResult | None:
        try:
            # Clean query for search
            clean_query = re.sub(
                r"^(?:who is|what is|where is|tell me about|explain|define)\s+",
                "",
                query.strip(),
                flags=re.IGNORECASE,
            ).strip("? .")

            if not clean_query:
                clean_query = query.strip("? .")

            # 1. Search for title
            search_url = (
                f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                f"&srsearch={quote(clean_query)}&srlimit=1&utf8=&format=json"
            )
            search_resp = requests.get(search_url, headers={"User-Agent": _USER_AGENT}, timeout=self.timeout)
            if search_resp.status_code != 200:
                logger.warning("Wikipedia search failed: HTTP %d", search_resp.status_code)
                return None

            search_data = search_resp.json()
            search_list = search_data.get("query", {}).get("search", [])
            if not search_list:
                return None

            top_title = search_list[0]["title"]

            # 2. Fetch page summary
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(top_title.replace(' ', '_'))}"
            summary_resp = requests.get(summary_url, headers={"User-Agent": _USER_AGENT}, timeout=self.timeout)
            if summary_resp.status_code != 200:
                logger.warning("Wikipedia summary failed: HTTP %d", summary_resp.status_code)
                return None

            summary_data = summary_resp.json()
            extract = summary_data.get("extract")
            page_url = summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")

            if not extract or len(extract.strip()) < 15:
                return None

            return SearchResult(
                title=summary_data.get("title", top_title),
                extract=extract.strip(),
                url=page_url or f"https://en.wikipedia.org/wiki/{quote(top_title.replace(' ', '_'))}",
                source="wikipedia",
            )
        except Exception as exc:
            logger.warning("Wikipedia request failed (%s)", type(exc).__name__)
            return None


class DuckDuckGoClient:
    """Keyless web search client using DuckDuckGo."""

    def __init__(self, timeout: float = 6.0, max_results: int = 4) -> None:
        self.timeout = timeout
        self.max_results = max_results

    def search(self, query: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        try:
            from ddgs import DDGS

            ddgs_results = list(DDGS(timeout=self.timeout).text(query, max_results=self.max_results))
            for item in ddgs_results:
                title = item.get("title", "").strip()
                body = item.get("body", "").strip()
                href = item.get("href", "").strip()
                if body and len(body) >= 15:
                    results.append(
                        SearchResult(
                            title=title,
                            extract=body,
                            url=href,
                            source="duckduckgo",
                        )
                    )
        except Exception as exc:
            logger.warning("DuckDuckGo request failed (%s)", type(exc).__name__)

        return results

"""Keyless public data sources for the travel planner: geocoding, weather, Wikivoyage and OpenStreetMap places.

Only a fixed set of hosts is ever contacted, requests are small and time-limited, and answers are cached so a
demo does not hammer shared community services (Nominatim asks for at most one request per second).
"""

from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import urlsplit

import requests

ALLOWED_HOSTS = {
    "geocoding-api.open-meteo.com", "api.open-meteo.com", "archive-api.open-meteo.com",
    "en.wikivoyage.org", "en.wikipedia.org", "nominatim.openstreetmap.org",
}
USER_AGENT = "clef-routing-poc/0.1 (local research prototype)"
MAX_BYTES = 2_000_000
CACHE_SECONDS = 6 * 3600


class WebError(RuntimeError):
    """A public data source could not be reached or answered unexpectedly."""


class PublicWeb:
    def __init__(self, session: requests.Session | None = None, clock: Any = time.monotonic) -> None:
        self._session = session or requests.Session()
        self._cache: dict[tuple[str, tuple], tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._nominatim_lock = threading.Lock()
        self._nominatim_last = 0.0
        self._clock = clock

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
            raise WebError("That host is not on the allowed list")
        key = (url, tuple(sorted((params or {}).items())))
        with self._lock:
            hit = self._cache.get(key)
            if hit and self._clock() - hit[0] < CACHE_SECONDS:
                return hit[1]
        if parts.hostname == "nominatim.openstreetmap.org":
            with self._nominatim_lock:
                wait = 1.1 - (self._clock() - self._nominatim_last)
                if wait > 0:
                    time.sleep(wait)
                self._nominatim_last = self._clock()
        try:
            response = self._session.get(url, params=params, timeout=(5, 12), allow_redirects=False,
                                         headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        except requests.RequestException as error:
            raise WebError(f"{parts.hostname} is not reachable ({type(error).__name__})") from error
        if response.status_code != 200 or len(response.content) > MAX_BYTES:
            raise WebError(f"{parts.hostname} answered HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as error:
            raise WebError(f"{parts.hostname} returned something other than JSON") from error
        with self._lock:
            self._cache[key] = (self._clock(), data)
        return data

    # ------------------------------------------------------------------ sources
    def geocode(self, place: str) -> dict[str, Any]:
        data = self.get_json("https://geocoding-api.open-meteo.com/v1/search",
                             {"name": place.strip()[:80], "count": 1, "language": "en"})
        results = data.get("results") or []
        if not results:
            raise WebError(f"No place called {place!r} was found")
        r = results[0]
        return {"name": r["name"], "country": r.get("country", ""), "country_code": r.get("country_code", ""),
                "lat": r["latitude"], "lon": r["longitude"], "timezone": r.get("timezone", ""),
                "population": r.get("population")}

    def weather(self, lat: float, lon: float, start: str, end: str, *, forecast: bool) -> list[dict[str, Any]]:
        host = "api.open-meteo.com/v1/forecast" if forecast else "archive-api.open-meteo.com/v1/archive"
        data = self.get_json(f"https://{host}", {
            "latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": start, "end_date": end,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", "timezone": "auto"})
        daily = data.get("daily") or {}
        days = []

        def at(key: str, i: int) -> Any:
            values = daily.get(key) or []
            return values[i] if i < len(values) else None

        for i, day in enumerate(daily.get("time") or []):
            hi, lo, rain = at("temperature_2m_max", i), at("temperature_2m_min", i), at("precipitation_sum", i)
            if hi is not None and lo is not None:
                days.append({"date": day, "high_c": round(hi), "low_c": round(lo), "rain_mm": round(rain or 0, 1)})
        return days

    def wikivoyage_intro(self, title: str) -> dict[str, str]:
        for host in ("en.wikivoyage.org", "en.wikipedia.org"):
            data = self.get_json(f"https://{host}/w/api.php", {
                "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1,
                "titles": title.strip()[:80], "format": "json"})
            pages = (data.get("query") or {}).get("pages") or {}
            for page in pages.values():
                text = (page.get("extract") or "").strip()
                if text and "missing" not in page:
                    return {"source": host, "title": page.get("title", title), "text": text[:1100]}
        raise WebError(f"No guide text was found for {title!r}")

    def hotels(self, lat: float, lon: float, limit: int = 14) -> list[dict[str, Any]]:
        box = 0.06
        data = self.get_json("https://nominatim.openstreetmap.org/search", {
            "q": "hotel", "format": "jsonv2", "limit": limit, "extratags": 1, "bounded": 1,
            "viewbox": f"{lon - box},{lat + box},{lon + box},{lat - box}"})
        out = []
        for item in data:
            name = item.get("name") or ""
            if name and item.get("type") in ("hotel", "guest_house", "hostel", "apartment", "motel"):
                tags = item.get("extratags") or {}
                out.append({"name": name[:80], "kind": item["type"], "stars": tags.get("stars"),
                            "lat": float(item["lat"]), "lon": float(item["lon"]), "website": tags.get("website")})
        return out

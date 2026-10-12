"""The travel planner's tools: real public data and web search where it exists, clearly labelled estimates where it does not.

Real: destination guides (Wikivoyage/Wikipedia), weather (Open-Meteo forecast or the same dates last year), hotel
names and star ratings (OpenStreetMap), and open web search with citations.
Simulated: flight fares and times, hotel nightly rates, and "holds". There is no free public booking API, so these
are deterministic, labelled "illustrative", and nothing can be booked.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from agent_demo.support import check_expectations
from agent_demo.web import PublicWeb, WebError

DATA = json.loads((Path(__file__).parent / "data" / "travel_places.json").read_text(encoding="utf-8-sig"))
AIRPORTS: dict[str, dict[str, Any]] = DATA["airports"]
DESTINATIONS: list[dict[str, Any]] = DATA["destinations"]
CARRIERS: list[str] = DATA["carriers"]["default"]
MAX_TRIP_DAYS = 21
MAX_SEARCHES_PER_LANE = 8
STAR_RATE = {1: 35, 2: 55, 3: 85, 4: 140, 5: 260}
INTERESTS = ("food", "culture", "nature", "beaches", "nightlife", "family", "shopping", "adventure")
SIM_NOTE = "Illustrative simulated data, not bookable. Confirm real prices and availability with the airline or hotel."

TOOLS: list[dict[str, Any]] = [
    {"name": "search_web", "description": "Search the open web and get a cited summary, for sights, restaurants, events, entry rules or prices.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "A specific search such as 'best traditional restaurants in Lisbon'."}},
         "required": ["query"]}},
    {"name": "get_destination_overview", "description": "Get a travel guide summary, coordinates, timezone and nearest airport for a place.",
     "parameters": {"type": "object", "properties": {
         "destination": {"type": "string", "description": "City or place name, for example Lisbon."}}, "required": ["destination"]}},
    {"name": "get_weather", "description": "Get the weather for a place and dates: a forecast when near, otherwise typical weather from the same dates last year.",
     "parameters": {"type": "object", "properties": {
         "destination": {"type": "string", "description": "City name."},
         "start_date": {"type": "string", "description": "First day, YYYY-MM-DD."},
         "end_date": {"type": "string", "description": "Last day, YYYY-MM-DD."}},
         "required": ["destination", "start_date", "end_date"]}},
    {"name": "suggest_destinations", "description": "Suggest the best-fitting cities for the dates and interests when no destination is chosen, ranked using weather.",
     "parameters": {"type": "object", "properties": {
         "start_date": {"type": "string", "description": "First day, YYYY-MM-DD."},
         "end_date": {"type": "string", "description": "Last day, YYYY-MM-DD."},
         "interests": {"type": "array", "items": {"type": "string"}, "description": "Any of food, culture, nature, beaches, nightlife, family, shopping, adventure."},
         "origin": {"type": "string", "description": "Departure city, used to prefer nearer places."}},
         "required": ["start_date", "end_date"]}},
    {"name": "search_flights", "description": "Find round-trip flight offers between two cities on given dates (simulated fares).",
     "parameters": {"type": "object", "properties": {
         "origin": {"type": "string", "description": "Departure city or IATA code."},
         "destination": {"type": "string", "description": "Arrival city or IATA code."},
         "depart_date": {"type": "string", "description": "Outbound date, YYYY-MM-DD."},
         "return_date": {"type": "string", "description": "Return date, YYYY-MM-DD."},
         "travelers": {"type": "integer", "description": "Number of travelers."}},
         "required": ["origin", "destination", "depart_date", "return_date"]}},
    {"name": "find_hotels", "description": "Find real hotels in a city with estimated nightly rates for the dates.",
     "parameters": {"type": "object", "properties": {
         "destination": {"type": "string", "description": "City name."},
         "check_in": {"type": "string", "description": "YYYY-MM-DD."}, "check_out": {"type": "string", "description": "YYYY-MM-DD."},
         "guests": {"type": "integer", "description": "Number of guests."},
         "budget": {"type": "string", "enum": ["budget", "mid-range", "premium"], "description": "Price level wanted."}},
         "required": ["destination", "check_in", "check_out"]}},
    {"name": "hold_flight", "description": "Place a hold on a flight offer returned by search_flights.",
     "parameters": {"type": "object", "properties": {
         "offer_id": {"type": "string", "description": "Offer id such as FL-1A2B3C."}}, "required": ["offer_id"]}},
    {"name": "hold_hotel", "description": "Place a hold on a hotel offer returned by find_hotels.",
     "parameters": {"type": "object", "properties": {
         "offer_id": {"type": "string", "description": "Offer id such as HT-1A2B3C."}}, "required": ["offer_id"]}},
]


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def _iso(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{name} must be a date like 2026-11-12") from error


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def season_factor(day: date) -> float:
    return {12: 1.25, 7: 1.2, 8: 1.2, 6: 1.1, 9: 1.05, 4: 1.1, 5: 1.05, 1: 0.9, 2: 0.9, 3: 0.95}.get(day.month, 1.0)


class SharedResults:
    """Tool answers shared by both lanes of a run, so they see identical facts and a search is paid for once."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self.stats = {"web_searches": 0, "search_input_tokens": 0, "search_output_tokens": 0}

    def once(self, key: str, compute: Callable[[], Any]) -> Any:
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self._values:
                self._values[key] = ("ok", compute())
        return self._values[key][1]


class TravelDesk:
    """One lane's view of the world: shared facts, plus its own offers and holds."""

    def __init__(self, llm: Any, search_model: str, web: PublicWeb, shared: SharedResults,
                 today: Callable[[], date] = date.today) -> None:
        self.llm, self.search_model, self.web, self.shared, self._today = llm, search_model, web, shared, today
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []
        self.offers: dict[str, dict[str, Any]] = {}
        self.holds: list[dict[str, Any]] = []
        self._searches = 0

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            result: dict[str, Any] = {"error": f"Unknown tool {name}"}
        else:
            try:
                result = handler(**args)
            except TypeError as error:
                result = {"error": f"Bad arguments for {name}: {error}"}
            except (ValueError, WebError) as error:
                result = {"error": str(error)}
            except Exception as error:  # noqa: BLE001 - a tool failure must reach the model as an error, not crash a lane
                result = {"error": f"{name} failed ({type(error).__name__})"}
        with self._lock:
            self.calls.append({"name": name, "arguments": args, "result": result, "ok": "error" not in result})
        return result

    def _shared(self, name: str, args: dict[str, Any], compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        return self.shared.once(f"{name}:{json.dumps(args, sort_keys=True, default=str)}", compute)

    # ------------------------------------------------------------------ helpers
    def _dates(self, start: Any, end: Any, *, past_ok: bool = False) -> tuple[date, date]:
        a, b = _iso(start, "start date"), _iso(end, "end date")
        if b < a:
            raise ValueError("The end date is before the start date")
        if (b - a).days > MAX_TRIP_DAYS:
            raise ValueError(f"Trips are limited to {MAX_TRIP_DAYS} days")
        if not past_ok and a < self._today():
            raise ValueError("The start date is in the past")
        return a, b

    def _airport(self, text: str) -> str:
        key = text.strip()
        if key.upper() in AIRPORTS:
            return key.upper()
        base = key.split("(")[0].strip().lower()
        for code, a in AIRPORTS.items():
            if a["city"].lower() == base:
                return code
        place = self.web.geocode(base)
        best = min(AIRPORTS.items(), key=lambda kv: distance_km((place["lat"], place["lon"]), (kv[1]["lat"], kv[1]["lon"])))
        if distance_km((place["lat"], place["lon"]), (best[1]["lat"], best[1]["lon"])) > 250:
            raise ValueError(f"No airport in the demo catalog near {text}")
        return best[0]

    # --------------------------------------------------------------------- tools
    def _tool_search_web(self, query: str) -> dict[str, Any]:
        query = query.strip()[:200]
        if not query:
            raise ValueError("query must not be empty")
        with self._lock:
            self._searches += 1
            over = self._searches > MAX_SEARCHES_PER_LANE
        if over:
            raise ValueError("The search limit for this trip has been reached; use the results you already have")

        def compute() -> dict[str, Any]:
            result = self.llm.search(self.search_model, query)
            self.shared.stats["web_searches"] += 1
            self.shared.stats["search_input_tokens"] += result.input_tokens
            self.shared.stats["search_output_tokens"] += result.output_tokens
            seen, sources = set(), []
            for s in result.sources:
                if s["url"] not in seen:
                    seen.add(s["url"])
                    sources.append(s)
            return {"query": query, "answer": result.text[:2200], "sources": sources[:6]}
        return self._shared("search_web", {"q": query.lower()}, compute)

    def _tool_get_destination_overview(self, destination: str) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            place = self.web.geocode(destination)
            try:
                guide = self.web.wikivoyage_intro(place["name"])
            except WebError:
                guide = {"source": "none", "title": place["name"], "text": ""}
            near = min(AIRPORTS.items(), key=lambda kv: distance_km((place["lat"], place["lon"]), (kv[1]["lat"], kv[1]["lon"])))
            km = distance_km((place["lat"], place["lon"]), (near[1]["lat"], near[1]["lon"]))
            return {**place, "airport": near[0] if km < 150 else None, "guide_source": guide["source"], "guide": guide["text"]}
        return self._shared("overview", {"d": destination.strip().lower()}, compute)

    def _tool_get_weather(self, destination: str, start_date: str, end_date: str) -> dict[str, Any]:
        start, end = self._dates(start_date, end_date, past_ok=True)

        def compute() -> dict[str, Any]:
            place = self.web.geocode(destination)
            near = start >= self._today() and end <= self._today() + timedelta(days=15)
            if near:
                days, basis = self.web.weather(place["lat"], place["lon"], start.isoformat(), end.isoformat(), forecast=True), "forecast"
            else:
                last_year = (start.replace(year=start.year - 1) if not (start.month == 2 and start.day == 29) else start.replace(year=start.year - 1, day=28))
                days = self.web.weather(place["lat"], place["lon"], last_year.isoformat(),
                                        (last_year + (end - start)).isoformat(), forecast=False)
                basis = "typical: the same dates last year"
            if not days:
                raise WebError("No weather data came back for those dates")
            return {"destination": place["name"], "basis": basis, "avg_high_c": round(sum(d["high_c"] for d in days) / len(days)),
                    "avg_low_c": round(sum(d["low_c"] for d in days) / len(days)),
                    "rainy_days": sum(d["rain_mm"] >= 1 for d in days), "days": days[:10]}
        return self._shared("weather", {"d": destination.strip().lower(), "s": start_date, "e": end_date}, compute)

    def _tool_suggest_destinations(self, start_date: str, end_date: str, interests: Any = None,
                                   origin: str = "") -> dict[str, Any]:
        start, end = self._dates(start_date, end_date)
        wanted = {str(i).lower() for i in (interests if isinstance(interests, list) else [interests] if interests else [])} & set(INTERESTS)

        def compute() -> dict[str, Any]:
            home = None
            if origin.strip():
                try:
                    place = self.web.geocode(origin.split("(")[0])
                    home = (place["lat"], place["lon"])
                except WebError:
                    home = None
            pool = []
            for d in DESTINATIONS:
                if origin and d["city"].lower() in origin.lower():
                    continue
                a = AIRPORTS[d["iata"]]
                km = distance_km(home, (a["lat"], a["lon"])) if home else None
                match = len(wanted & set(d["tags"])) if wanted else 1
                pool.append((match / max(1, len(wanted)), -(km or 0), d, a, km))
            pool.sort(key=lambda row: (-row[0], -row[1]))
            shortlist = pool[:9]
            last = start.replace(year=start.year - 1, day=min(start.day, 28)) if start.month == 2 else start.replace(year=start.year - 1)

            def weather_for(row: Any) -> dict[str, Any] | None:
                try:
                    days = self.web.weather(row[3]["lat"], row[3]["lon"], last.isoformat(), (last + (end - start)).isoformat(), forecast=False)
                except WebError:
                    return None
                return {"high": sum(x["high_c"] for x in days) / len(days), "low": sum(x["low_c"] for x in days) / len(days),
                        "rainy": sum(x["rain_mm"] >= 1 for x in days), "n": len(days)} if days else None
            with ThreadPoolExecutor(max_workers=5) as pool_exec:
                weathers = list(pool_exec.map(weather_for, shortlist))
            ranked = []
            for row, w in zip(shortlist, weathers, strict=True):
                if not w:
                    continue
                comfort = max(0.0, 1 - abs(w["high"] - 24) / 14) - 0.35 * (w["rainy"] / w["n"])
                score = 0.5 * row[0] + 0.4 * comfort + (0.1 * max(0.0, 1 - row[4] / 12000) if row[4] is not None else 0.05)
                ranked.append((score, row, w))
            ranked.sort(key=lambda r: -r[0])
            options = [{"city": r[1][2]["city"], "country": r[1][2]["country"], "airport": r[1][2]["iata"],
                        "matches": sorted(wanted & set(r[1][2]["tags"])) or r[1][2]["tags"][:2],
                        "avg_high_c": round(r[2]["high"]), "avg_low_c": round(r[2]["low"]), "rainy_days": r[2]["rainy"],
                        "distance_km": round(r[1][4]) if r[1][4] is not None else None, "price_level": r[1][2]["cost"],
                        "fit_score": round(r[0], 2)} for r in ranked[:5]]
            if not options:
                raise WebError("Weather data was unavailable, so destinations could not be ranked")
            return {"basis": "typical weather: the same dates last year", "interests": sorted(wanted), "options": options}
        return self._shared("suggest", {"s": start_date, "e": end_date, "i": sorted(wanted), "o": origin.lower()}, compute)

    def _tool_search_flights(self, origin: str, destination: str, depart_date: str, return_date: str,
                             travelers: int = 1) -> dict[str, Any]:
        out, back = self._dates(depart_date, return_date)
        count = max(1, min(9, int(travelers)))
        a, b = self._airport(origin), self._airport(destination)
        if a == b:
            raise ValueError("Origin and destination are the same")
        km = distance_km((AIRPORTS[a]["lat"], AIRPORTS[a]["lon"]), (AIRPORTS[b]["lat"], AIRPORTS[b]["lon"]))
        lead = (out - self._today()).days
        advance = 1.35 if lead < 14 else 1.15 if lead < 30 else 0.92 if lead > 60 else 1.0
        base = (55 + 0.075 * km) * season_factor(out) * advance
        variants = [("Cheapest", 0.88, 1 if km > 2500 else 0, False), ("Fastest", 1.22, 0, False),
                    ("Best value", 1.0, 0 if km < 6500 else 1, False), ("Flexible fare", 1.42, 0, True)]
        offers = []
        for label, factor, stops, refundable in variants:
            s = _seed(a, b, depart_date, return_date, label)
            minutes = round(km / (820 if stops == 0 else 740) * 60 + 35 + stops * 110)
            hour = 6 + s % 15
            leg = lambda frm, to, day, h: {  # noqa: E731
                "from": frm, "to": to, "date": day.isoformat(), "departs": f"{h:02d}:{(s // 7) % 4 * 15:02d}",
                "duration": f"{minutes // 60}h {minutes % 60:02d}m", "stops": stops}
            per_person = round(base * factor * (0.94 + (s % 13) / 100) * 1.9 / 5) * 5
            offer = {"offer_id": f"FL-{s:08X}"[:9], "label": label, "carrier": CARRIERS[s % len(CARRIERS)],
                     "outbound": leg(a, b, out, hour), "return": leg(b, a, back, 7 + (s // 3) % 13),
                     "refundable": refundable, "price_per_person_usd": per_person, "total_usd": per_person * count,
                     "travelers": count}
            offers.append(offer)
            with self._lock:
                self.offers[offer["offer_id"]] = {**offer, "kind": "flight"}
        return {"route": f"{a} to {b}", "distance_km": round(km), "offers": offers, "simulated": True, "note": SIM_NOTE}

    def _tool_find_hotels(self, destination: str, check_in: str, check_out: str, guests: int = 2,
                          budget: str = "mid-range") -> dict[str, Any]:
        start, end = self._dates(check_in, check_out)
        nights = max(1, (end - start).days)
        rooms = max(1, math.ceil(max(1, min(9, int(guests))) / 2))

        def fetch() -> dict[str, Any]:
            place = self.web.geocode(destination)
            listings = self.web.hotels(place["lat"], place["lon"])
            if not listings:
                raise WebError(f"No hotel listings were found in {place['name']}")
            return {"place": place, "listings": listings}
        found = self._shared("hotels", {"d": destination.strip().lower()}, fetch)
        index = next((d["cost"] for d in DESTINATIONS if d["city"].lower() == found["place"]["name"].lower()), 1.0)
        wanted = {"budget": (1, 3), "mid-range": (3, 4), "premium": (4, 5)}.get(budget, (3, 4))
        scored = []
        for h in found["listings"]:
            try:
                stars = int(float(h["stars"])) if h["stars"] else None
            except ValueError:
                stars = None
            level = stars or (1 if h["kind"] == "hostel" else 3)
            s = _seed(h["name"], check_in)
            rate = round(STAR_RATE[min(5, max(1, level))] * index * season_factor(start) * (0.9 + (s % 25) / 100))
            scored.append((0 if wanted[0] <= level <= wanted[1] else 1, h, stars, rate))
        scored.sort(key=lambda row: row[0])
        offers = []
        for _, h, stars, rate in scored[:5]:
            offer = {"offer_id": f"HT-{_seed(h['name'], check_in, check_out):08X}"[:9], "name": h["name"], "kind": h["kind"],
                     "stars": stars or "not rated", "nightly_usd_estimate": rate, "nights": nights, "rooms": rooms,
                     "total_usd_estimate": rate * nights * rooms,
                     "map": f"https://www.openstreetmap.org/?mlat={h['lat']:.5f}&mlon={h['lon']:.5f}#map=17/{h['lat']:.5f}/{h['lon']:.5f}"}
            offers.append(offer)
            with self._lock:
                self.offers[offer["offer_id"]] = {**offer, "kind": "hotel"}
        return {"destination": found["place"]["name"], "check_in": check_in, "check_out": check_out, "offers": offers,
                "names_source": "OpenStreetMap", "rates_basis": "estimated from star rating and city price level",
                "note": SIM_NOTE}

    def _hold(self, offer_id: str, kind: str) -> dict[str, Any]:
        with self._lock:
            offer = self.offers.get(str(offer_id).strip().upper())
            if offer is None or offer["kind"] != kind:
                raise ValueError(f"No {kind} offer {offer_id} was returned in this case")
            if any(h["offer_id"] == offer["offer_id"] for h in self.holds):
                raise ValueError(f"{offer['offer_id']} is already held")
            hold = {"hold_id": f"HOLD-{len(self.holds) + 1:03d}", "offer_id": offer["offer_id"], "kind": kind,
                    "expires": "24 hours", "simulated": True}
            self.holds.append(hold)
        return {**hold, "note": "A simulated hold: nothing was booked or charged."}

    def _tool_hold_flight(self, offer_id: str) -> dict[str, Any]:
        return self._hold(offer_id, "flight")

    def _tool_hold_hotel(self, offer_id: str) -> dict[str, Any]:
        return self._hold(offer_id, "hotel")

    # -------------------------------------------------------------- outcome checks
    def changes(self) -> list[str]:
        notes = []
        for hold in self.holds:
            offer = self.offers[hold["offer_id"]]
            if hold["kind"] == "flight":
                notes.append(f"Held flight {offer['offer_id']} {offer['outbound']['from']}-{offer['outbound']['to']} (${offer['total_usd']:,})")
            else:
                notes.append(f"Held hotel {offer['name']} (${offer['total_usd_estimate']:,} estimated)")
        return notes

    def check(self, expectations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return check_expectations(self.calls, expectations)


def responses_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in TOOLS]


def router_tools() -> list[dict[str, Any]]:
    return [{"description": t["description"], "name": t["name"],
             "parameters": {"properties": t["parameters"]["properties"],
                            "required": t["parameters"].get("required", []), "type": "dict"}} for t in TOOLS]

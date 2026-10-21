"""The travel planner: its tools with a fake web, the plan shape, and parallel sub-agents, all offline."""

from __future__ import annotations

import dataclasses
import json
import threading
from datetime import date
from pathlib import Path

import pytest
from agent_demo_fakes import FakeDecider, ScriptedLLM, collect

from agent_demo import travel
from agent_demo.agent import Agent, normalize_specialist, parse_plan
from agent_demo.domains import TRAVEL, travel_message, validate_itinerary
from agent_demo.llm import AzureResponsesClient, LLMError, LLMResult
from agent_demo.policy import Policy
from agent_demo.preferences import Preferences
from agent_demo.registry import Registry
from agent_demo.router import AdaptiveRouter
from agent_demo.telemetry import Telemetry
from agent_demo.travel import MAX_SEARCHES_PER_LANE, SharedResults, TravelDesk
from agent_demo.web import PublicWeb, WebError

DATA = Path(__file__).resolve().parents[1] / "agent_demo" / "data"
TODAY = date(2026, 10, 8)
PLACES = {"lisbon": (38.72, -9.15, "Lisbon", "Portugal"), "dubai": (25.20, 55.27, "Dubai", "United Arab Emirates"),
          "tokyo": (35.68, 139.69, "Tokyo", "Japan"), "rome": (41.90, 12.50, "Rome", "Italy")}


class FakeWeb:
    """Stands in for every public data source; counts calls so caching can be asserted."""

    def __init__(self, hotels: bool = True) -> None:
        self.calls = {"geocode": 0, "weather": 0, "hotels": 0, "guide": 0}
        self.has_hotels = hotels

    def geocode(self, place: str) -> dict:
        self.calls["geocode"] += 1
        key = place.split("(")[0].strip().lower()
        if key not in PLACES:
            raise WebError(f"No place called {place!r} was found")
        lat, lon, name, country = PLACES[key]
        return {"name": name, "country": country, "country_code": "XX", "lat": lat, "lon": lon, "timezone": "UTC", "population": 1}

    def weather(self, lat, lon, start, end, *, forecast):
        self.calls["weather"] += 1
        n = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
        warm = 31 - round(abs(lat) / 4)
        return [{"date": f"{start[:8]}{i + 1:02d}", "high_c": warm, "low_c": warm - 8, "rain_mm": 2.0 if i == 0 else 0.0} for i in range(n)]

    def wikivoyage_intro(self, title):
        self.calls["guide"] += 1
        return {"source": "en.wikivoyage.org", "title": title, "text": f"{title} is a lovely city."}

    def hotels(self, lat, lon, limit=14):
        self.calls["hotels"] += 1
        if not self.has_hotels:
            return []
        rows = [("Hostel Nook", "hostel", None), ("Casa Dois", "hotel", "2"), ("Hotel Central", "hotel", "3"),
                ("Grand Plaza", "hotel", "4"), ("Royal Palace", "hotel", "5"), ("Odd Stay", "hotel", "n/a")]
        return [{"name": n, "kind": k, "stars": s, "lat": lat, "lon": lon, "website": None} for n, k, s in rows]


class SearchLLM:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, deployment, query, **_):
        self.queries.append(query)
        return LLMResult(f"Results about {query}", 1000, 200, 5.0, "completed", sources=[
            {"title": "Guide", "url": "https://guide.example/a"}, {"title": "Dup", "url": "https://guide.example/a"}])


def make_desk(shared=None, web=None, llm=None):
    return TravelDesk(llm or SearchLLM(), "gpt-4.1", web or FakeWeb(), shared or SharedResults(), today=lambda: TODAY)


FLIGHT = {"origin": "Dubai", "destination": "Lisbon", "depart_date": "2026-11-20", "return_date": "2026-11-24", "travelers": 2}
HOTEL = {"destination": "Lisbon", "check_in": "2026-11-20", "check_out": "2026-11-24", "guests": 2}


# ------------------------------------------------------------------------------- tools
def test_overview_and_weather_use_real_looking_sources_and_are_shared_between_lanes():
    web, shared = FakeWeb(), SharedResults()
    a, b = make_desk(shared, web), make_desk(shared, web)
    first = a.execute("get_destination_overview", {"destination": "Lisbon"})
    second = b.execute("get_destination_overview", {"destination": " lisbon "})
    assert first == second and first["guide_source"] == "en.wikivoyage.org" and "lovely" in first["guide"]
    assert first["airport"] is not None and web.calls["geocode"] == 1 and web.calls["guide"] == 1
    near = a.execute("get_weather", {"destination": "Lisbon", "start_date": "2026-10-12", "end_date": "2026-10-14"})
    far = a.execute("get_weather", {"destination": "Lisbon", "start_date": "2026-12-01", "end_date": "2026-12-05"})
    assert near["basis"] == "forecast" and far["basis"].startswith("typical") and far["avg_high_c"] > far["avg_low_c"]
    assert far["rainy_days"] == 1 and len(far["days"]) == 5


@pytest.mark.parametrize(("tool", "args", "fragment"), [
    ("get_weather", {"destination": "Lisbon", "start_date": "2026-12-05", "end_date": "2026-12-01"}, "before"),
    ("get_weather", {"destination": "Lisbon", "start_date": "2026-12-01", "end_date": "2027-02-01"}, "limited"),
    ("get_weather", {"destination": "Atlantis", "start_date": "2026-12-01", "end_date": "2026-12-03"}, "No place"),
    ("get_weather", {"destination": "Lisbon", "start_date": "soon", "end_date": "2026-12-03"}, "date like"),
    ("search_flights", {**FLIGHT, "depart_date": "2026-09-01", "return_date": "2026-09-05"}, "past"),
    ("search_flights", {**FLIGHT, "destination": "Dubai"}, "same"),
    ("search_flights", {**FLIGHT, "destination": "Atlantis"}, "No place"),
    ("find_hotels", {**HOTEL, "check_out": "2026-11-10"}, "before"),
    ("nope", {}, "Unknown tool"),
])
def test_tools_report_problems_as_errors_the_model_can_read(tool, args, fragment):
    result = make_desk().execute(tool, args)
    assert fragment in result["error"]


def test_flights_are_deterministic_scale_with_travelers_and_are_labelled_simulated():
    one, other = make_desk(), make_desk()
    a = one.execute("search_flights", FLIGHT)
    assert a == other.execute("search_flights", FLIGHT) and a["simulated"] is True and "not bookable" in a["note"]
    assert [o["label"] for o in a["offers"]] == ["Cheapest", "Fastest", "Best value", "Flexible fare"]
    assert len({o["offer_id"] for o in a["offers"]}) == 4 and all(o["offer_id"].startswith("FL-") for o in a["offers"])
    solo = make_desk().execute("search_flights", {**FLIGHT, "travelers": 1})
    assert a["offers"][0]["total_usd"] == 2 * solo["offers"][0]["total_usd"] and a["distance_km"] > 5000
    assert a["offers"][3]["refundable"] is True and a["offers"][1]["outbound"]["stops"] == 0
    assert make_desk().execute("search_flights", {**FLIGHT, "origin": "DXB", "destination": "lis"})["route"] == "DXB to LIS"


def test_hotels_are_real_names_with_estimated_rates_filtered_by_budget():
    web, shared = FakeWeb(), SharedResults()
    mid = make_desk(shared, web).execute("find_hotels", {**HOTEL, "budget": "mid-range"})
    assert mid["names_source"] == "OpenStreetMap" and "estimated" in mid["rates_basis"] and len(mid["offers"]) == 5
    assert {o["stars"] for o in mid["offers"][:2]} <= {3, 4}
    premium = make_desk(shared, web).execute("find_hotels", {**HOTEL, "budget": "premium"})
    budget = make_desk(shared, web).execute("find_hotels", {**HOTEL, "budget": "budget"})
    assert premium["offers"][0]["nightly_usd_estimate"] > budget["offers"][0]["nightly_usd_estimate"]
    assert web.calls["hotels"] == 1 and all(o["total_usd_estimate"] == o["nightly_usd_estimate"] * 4 for o in mid["offers"])
    assert all(o["map"].startswith("https://www.openstreetmap.org/") for o in mid["offers"])
    assert "No hotel" in make_desk(web=FakeWeb(hotels=False)).execute("find_hotels", HOTEL)["error"]
    four = make_desk().execute("find_hotels", {**HOTEL, "guests": 4})
    assert four["offers"][0]["rooms"] == 2


def test_holds_need_an_offer_from_this_case_and_cannot_repeat():
    desk = make_desk()
    assert "was returned" in desk.execute("hold_flight", {"offer_id": "FL-NOPE"})["error"]
    flight = desk.execute("search_flights", FLIGHT)["offers"][1]["offer_id"]
    hotel = desk.execute("find_hotels", HOTEL)["offers"][0]["offer_id"]
    assert "was returned" in desk.execute("hold_hotel", {"offer_id": flight})["error"]
    held = desk.execute("hold_flight", {"offer_id": flight.lower()})
    assert held["simulated"] is True and held["hold_id"] == "HOLD-001" and "nothing was booked" in held["note"]
    assert "already held" in desk.execute("hold_flight", {"offer_id": flight})["error"]
    assert desk.execute("hold_hotel", {"offer_id": hotel})["hold_id"] == "HOLD-002"
    assert [c.split(" ")[1] for c in desk.changes()] == ["flight", "hotel"]
    other = make_desk()
    assert other.changes() == [] and "was returned" in other.execute("hold_flight", {"offer_id": flight})["error"]
    checks = desk.check([{"label": "flight", "tool": "hold_flight"}, {"label": "no destination search", "tool": "suggest_destinations", "absent": True},
                         {"label": "dest", "tool": "search_flights", "args": {"destination": "lisbon"}}])
    assert [c["passed"] for c in checks] == [True, True, True]


def test_destination_suggestions_use_interests_weather_and_skip_the_origin():
    desk = make_desk()
    result = desk.execute("suggest_destinations", {"start_date": "2026-12-10", "end_date": "2026-12-15", "interests": ["beaches", "food"], "origin": "Dubai"})
    options = result["options"]
    assert 1 <= len(options) <= 5 and [o["fit_score"] for o in options] == sorted((o["fit_score"] for o in options), reverse=True)
    assert all("Dubai" != o["city"] and o["avg_high_c"] and o["airport"] for o in options)
    assert result["interests"] == ["beaches", "food"] and "same dates last year" in result["basis"]
    assert all(set(o["matches"]) <= {"beaches", "food"} for o in options)
    nature = make_desk().execute("suggest_destinations", {"start_date": "2026-12-10", "end_date": "2026-12-12", "interests": "nature"})
    assert nature["options"] and all("nature" in o["matches"] for o in nature["options"][:2])


def test_web_search_is_cited_cached_counted_and_capped():
    llm, shared = SearchLLM(), SharedResults()
    a, b = make_desk(shared, llm=llm), make_desk(shared, llm=llm)
    first = a.execute("search_web", {"query": "Best food in Lisbon"})
    assert first["sources"] == [{"title": "Guide", "url": "https://guide.example/a"}] and "Results about" in first["answer"]
    assert b.execute("search_web", {"query": "best food in lisbon"}) == first
    assert llm.queries == ["Best food in Lisbon"] and shared.stats["web_searches"] == 1
    assert shared.stats["search_input_tokens"] == 1000 and shared.stats["search_output_tokens"] == 200
    assert "must not be empty" in a.execute("search_web", {"query": "  "})["error"]
    capped = make_desk(SharedResults(), llm=SearchLLM())
    results = [capped.execute("search_web", {"query": f"q{i}"}) for i in range(MAX_SEARCHES_PER_LANE + 2)]
    assert sum("error" in r for r in results) >= 2 and "limit" in results[-1]["error"]


def test_tool_rosters_agree_and_have_a_search_tool():
    api, trained = travel.responses_tools(), travel.router_tools()
    assert [t["name"] for t in api] == [t["name"] for t in trained] and "search_web" in {t["name"] for t in api}
    assert all(t["parameters"]["type"] == "dict" for t in trained)


def test_public_web_only_talks_to_allowed_hosts_and_caches():
    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, **kwargs):
            self.urls.append(url)
            return type("R", (), {"status_code": 200, "content": b"{}", "json": lambda self: {"results": []}})()
    session = Session()
    web = PublicWeb(session=session)
    for bad in ("http://api.open-meteo.com/v1/forecast", "https://evil.example/x", "https://geocoding-api.open-meteo.com.evil.example/"):
        with pytest.raises(WebError):
            web.get_json(bad)
    web.get_json("https://api.open-meteo.com/v1/forecast", {"a": 1})
    web.get_json("https://api.open-meteo.com/v1/forecast", {"a": 1})
    assert len(session.urls) == 1
    with pytest.raises(WebError, match="No place"):
        web.geocode("Nowhere")


# ---------------------------------------------------------------------------- request form
GOOD = {"origin": "Dubai", "destination": "", "start_date": "2026-12-01", "end_date": "2026-12-06", "travelers": 2,
        "budget": "mid-range", "interests": ["beaches", "food"], "notes": "warm"}


def test_the_form_becomes_a_request_that_says_whether_a_destination_is_chosen():
    open_ = travel_message(GOOD, TODAY)
    assert "5-night trip for 2 travelers" in open_ and "not decided" in open_ and "Interests: beaches, food." in open_
    chosen = travel_message({**GOOD, "destination": "Lisbon", "travelers": 1, "interests": [], "notes": ""}, TODAY)
    assert "Destination: Lisbon." in chosen and "1 traveler." in chosen and "Interests" not in chosen and "Notes" not in chosen


@pytest.mark.parametrize(("change", "fragment"), [
    ({"start_date": "2026-10-01"}, "past"), ({"end_date": "2026-11-30"}, "before"), ({"end_date": "2027-03-01"}, "limited"),
    ({"start_date": "soon"}, "Choose"), ({"travelers": 0}, "Travelers"), ({"travelers": "2"}, "Travelers"), ({"travelers": True}, "Travelers"),
    ({"budget": "lavish"}, "Budget"), ({"interests": ["skydiving"]}, "Interests"), ({"interests": "food"}, "Interests"),
    ({"origin": ""}, "starts"), ({"notes": "x" * 401}, "too long"), ({"destination": "x" * 61}, "too long"),
])
def test_the_form_is_validated(change, fragment):
    with pytest.raises(ValueError, match=fragment):
        travel_message({**GOOD, **change}, TODAY)
    with pytest.raises(ValueError):
        travel_message("not a form", TODAY)  # type: ignore[arg-type]


def test_the_itinerary_must_be_json_with_days():
    ok = json.dumps({"days": [{"day": 1, "items": []}]})
    assert validate_itinerary(ok) is None
    for bad in ("plain text", "[]", "{}", json.dumps({"days": []}), json.dumps({"days": ["x"]}), json.dumps({"days": [{"items": "no"}]}), None):
        assert validate_itinerary(bad)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------ plan
def travel_plan(*steps):
    return json.dumps({"steps": [{"title": t, "instruction": i, "specialist": s, "parallel": p, "reply": r} for t, i, s, p, r in steps]})


PLAN = travel_plan(
    ("Find flights", "Search flights.", "flights", False, False),
    ("Research sights", "Research sights in Lisbon.", "search_web", True, False),
    ("Research dining", "Research restaurants in Lisbon.", "restaurants", True, False),
    ("Hold flight and hotel", "Hold the best flight.", "booking", True, False),
    ("Write the itinerary", "Compose it.", "", False, True))
ITINERARY = json.dumps({"title": "Lisbon", "destination": "Lisbon", "days": [{"day": 1, "date": "2026-11-20", "title": "Arrive", "items": [
    {"time": "Evening", "kind": "food", "title": "Dinner", "detail": "Good food", "url": "https://guide.example/a"}]}]})


def test_plan_parsing_keeps_specialists_and_parallel_flags_only_where_allowed():
    steps = parse_plan(PLAN, 8, parallel=True)
    assert [s["specialist"] for s in steps] == ["flights", "sights", "dining", "booking", ""]
    assert [s["parallel"] for s in steps] == [False, True, True, False, False] and steps[-1]["reply"] is True
    support_style = parse_plan(PLAN, 8)
    assert all(not s["parallel"] and s["specialist"] == "" for s in support_style)
    appended = parse_plan(travel_plan(("A", "Search.", "flights", True, False)), 8, parallel=True)
    assert appended[-1]["reply"] and appended[-1]["title"] == "Write the itinerary" and appended[-1]["parallel"] is False
    assert normalize_specialist("get_weather", "x") == "weather" and normalize_specialist(None, "Find a hotel in Rome") == "hotels"
    assert normalize_specialist("", "Do something") == "research"


# --------------------------------------------------------------------------------- the agent
@pytest.fixture
def parts(tmp_path):
    return Registry.load(DATA / "registry.json", tmp_path / "r.json"), Policy.load(DATA / "policy.json"), Telemetry(":memory:")


def run_trip(parts, llm, *, decider=None, prefs=None, expectations=None):
    registry, policy, telemetry = parts
    domain = dataclasses.replace(TRAVEL, make_desk=lambda llm_, shared: TravelDesk(llm_, "gpt-4.1", FakeWeb(), shared, today=lambda: TODAY))
    agent = Agent(AdaptiveRouter(registry, policy, decider or FakeDecider(), telemetry), llm, registry, policy, telemetry,
                  domains={"travel": domain})
    events: list[dict] = []
    summary = agent.run("Plan a trip.", prefs=prefs or Preferences(), expectations=expectations, agent="travel", emit=collect(events))
    return summary, events


def test_parallel_sub_agents_really_overlap_in_both_lanes(parts):
    llm = ScriptedLLM(plan=PLAN, reply_text=ITINERARY, barrier=threading.Barrier(4))
    summary, events = run_trip(parts, llm)
    assert not [e for e in events if e["type"] == "exec_error"] and summary["agent"] == "travel"
    started = {(e["lane"], e["index"]): e for e in events if e["type"] == "step_started"}
    assert started[("clef", 2)]["parallel"] and started[("clef", 2)]["specialist"] == "sights" and started[("baseline", 3)]["specialist"] == "dining"
    assert started[("clef", 4)]["specialist"] == "booking" and started[("clef", 4)]["parallel"] is False
    for lane in ("clef", "baseline"):
        assert summary["lanes"][lane]["steps_done"] == 5 and summary["lanes"][lane]["failed_steps"] == 0


def test_lane_totals_equal_the_sum_of_their_steps_even_when_steps_overlap(parts):
    summary, events = run_trip(parts, ScriptedLLM(plan=PLAN, reply_text=ITINERARY))
    for lane in ("clef", "baseline"):
        done = [e["step"] for e in events if e["type"] == "step_done" and e["lane"] == lane]
        assert len(done) == 5
        assert summary["lanes"][lane]["cost_usd"] == pytest.approx(sum(s["cost_usd"] for s in done), abs=1e-5)
        assert summary["lanes"][lane]["tokens_total"] == sum(s["tokens"] for s in done)
    final = [e for e in events if e["type"] == "step_done" and e["lane"] == "clef"][-1]["totals"]
    assert final["steps_done"] == 5 and sum(m["steps"] for m in final["models"].values()) >= 5
    assert len([e for e in events if e["type"] == "step_compare"]) == 5


def test_the_itinerary_is_checked_for_structure_before_any_reviewer_is_paid(parts):
    summary, events = run_trip(parts, ScriptedLLM(plan=PLAN, reply_text="Here is a nice trip, enjoy!"))
    bad = [e for e in events if e["type"] == "judge" and e["index"] == 5]
    assert bad and all(e["score"] == 1 and "JSON" in e["reason"] and e["cost_usd"] == 0 for e in bad)
    assert summary["lanes"]["baseline"]["reply_score"] == 1 and summary["lanes"]["clef"]["reply_score"] == 1
    assert all(summary["lanes"][lane]["failed_steps"] == 1 for lane in ("clef", "baseline"))
    good, _ = run_trip(parts, ScriptedLLM(plan=PLAN, reply_text=ITINERARY))
    assert good["lanes"]["clef"]["reply_score"] == 5 and good["lanes"]["baseline"]["reply_score"] == 5


def test_holds_and_outcome_checks_work_per_lane_in_the_travel_world(parts):
    flight = {"name": "search_flights", "arguments": FLIGHT}
    llm = ScriptedLLM(plan=PLAN, reply_text=ITINERARY, tool_calls=[flight])
    expectations = [{"label": "Flights searched", "tool": "search_flights", "args": {"destination": "Lisbon"}},
                    {"label": "No destination search", "tool": "suggest_destinations", "absent": True}]
    summary, _ = run_trip(parts, llm, expectations=expectations)
    assert summary["checks_passed"] == {"clef": [2, 2], "baseline": [2, 2]}


def test_travel_stakes_apply_to_booking_steps_only(parts):
    assert TRAVEL.stakes.search("Hold the best flight") and TRAVEL.stakes.search("Book the hotel")
    assert not TRAVEL.stakes.search("Search flights and compare prices")
    _, events = run_trip(parts, ScriptedLLM(plan=PLAN, reply_text=ITINERARY))
    needs = {e["index"]: e["decision"]["requirement"]["parts"] for e in events if e["type"] == "decision"}
    assert any("changes money" in name for name, _ in needs[4]) and not any("changes money" in name for name, _ in needs[1])
    assert any("customer-facing" in name for name, _ in needs[5])


def test_parallel_sub_agents_cost_the_time_of_the_slowest_not_the_sum(parts):
    summary, events = run_trip(parts, ScriptedLLM(plan=PLAN, reply_text=ITINERARY))
    # baseline: flights 10 + parallel group max(10, 10) + booking 10 + itinerary 10 ms, with nothing else on the clock
    assert summary["lanes"]["baseline"]["latency_ms"] == 40.0
    totals = [e for e in events if e["type"] == "lane_totals" and e["lane"] == "baseline"]
    assert len(totals) == 1 and totals[0]["totals"]["latency_ms"] == 20.0


def test_a_long_itinerary_reaches_the_screen_whole(parts):
    long_trip = json.dumps({"title": "Long", "days": [{"day": 1, "date": "2026-11-20", "title": "Day", "items": [
        {"time": "Morning", "kind": "sight", "title": "Walk", "detail": "x" * 30000, "url": None}]}],
        "caveats": ["End of complete itinerary"]})
    llm = ScriptedLLM(plan=PLAN, reply_text=long_trip)
    _, events = run_trip(parts, llm)
    replies = [e for e in events if e["type"] == "exec" and e["index"] == 5]
    assert replies and all(len(e["output"]) == len(long_trip) and json.loads(e["output"]) for e in replies)
    reviews = [c for c in llm.calls if (c["system"] or "").startswith("You grade") and "Compose it." in (c["user"] or "")]
    assert len(reviews) >= 2
    assert all(c["user"].split("\n\nOUTPUT:\n", 1)[1] == long_trip for c in reviews)
    steps = [e for e in events if e["type"] == "exec" and e["index"] == 2]
    assert all(len(e["output"]) <= 5000 for e in steps)


class ReplySequence:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def complete(self, deployment, **kwargs):
        self.calls.append({"deployment": deployment, **kwargs})
        result = self.results.pop(0)
        if isinstance(result, LLMError):
            raise result
        return result


def execute_json_reply(parts, llm, limit=5000):
    registry, policy, telemetry = parts
    agent = Agent(AdaptiveRouter(registry, policy, FakeDecider(), telemetry), llm, registry, policy, telemetry)
    return agent._execute(registry.get("gpt-5.6-sol"), "reply", {"instruction": "Compose it.", "reply": True},
                          "Plan a week in Tokyo.", ["Known trip facts."], None, TRAVEL, limit)


@pytest.mark.parametrize("limit", [1500, 5000, 6000])
def test_json_replies_use_the_same_budget_regardless_of_execution_lane(parts, limit):
    llm = ReplySequence(LLMResult(ITINERARY, 100, 200, 10.0, "completed"))
    result = execute_json_reply(parts, llm, limit)
    assert result["text"] == ITINERARY and result["output_error"] is None
    assert len(llm.calls) == 1
    assert llm.calls[0]["json_object"] is True
    assert llm.calls[0]["max_output_tokens"] == TRAVEL.reply_tokens == 6000


@pytest.mark.parametrize("first", [
    LLMResult('{"days":[', 100, 6000, 10.0, "incomplete"),
    LLMResult("## Tokyo itinerary\nVisit a temple.", 100, 200, 10.0, "completed"),
    LLMResult(ITINERARY, 100, 6000, 10.0, "incomplete"),
    LLMResult('{"days":[]}', 100, 200, 10.0, "completed"),
])
def test_invalid_or_incomplete_json_is_regenerated_once_and_all_usage_is_counted(parts, first):
    llm = ReplySequence(first, LLMResult(ITINERARY, 150, 300, 20.0, "completed"))
    result = execute_json_reply(parts, llm)
    assert result["text"] == ITINERARY and result["output_error"] is None
    assert result["input_tokens"] == first.input_tokens + 150
    assert result["output_tokens"] == first.output_tokens + 300
    assert result["latency_ms"] == 30.0
    assert [c["max_output_tokens"] for c in llm.calls] == [6000, 12000]
    assert {c["deployment"] for c in llm.calls} == {"gpt-5.6-sol"}
    assert all(c["json_object"] and "tools" not in c and "previous_response_id" not in c for c in llm.calls)
    assert llm.calls[0]["user"] == llm.calls[1]["user"]
    assert len(result["json_attempts"]) == 2 and result["json_attempts"][0]["error"]


def test_json_recovery_failure_is_explicit_and_keeps_paid_usage(parts):
    llm = ReplySequence(LLMResult("not JSON", 100, 200, 10.0, "completed"), LLMError("gateway unavailable"))
    result = execute_json_reply(parts, llm)
    assert "gateway unavailable" in result["output_error"]
    assert result["input_tokens"] == 100 and result["output_tokens"] == 200
    assert result["latency_ms"] == 10.0 and len(result["json_attempts"]) == 2


def test_json_recovery_is_bounded_and_never_turns_invalid_text_into_a_success(parts):
    llm = ReplySequence(*(LLMResult('{"days":[', 100, 6000, 10.0, "incomplete") for _ in range(2)))
    result = execute_json_reply(parts, llm)
    assert result["output_error"] and len(llm.calls) == 2
    assert validate_itinerary(result["text"]) is not None
    assert result["output_tokens"] == 12000


@pytest.mark.parametrize("payload", [
    {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}, "output": []},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "Cannot comply."}]}]},
])
def test_filtered_or_refused_outputs_are_reported_without_regeneration(parts, payload):
    parsed = AzureResponsesClient._parse(payload, 10.0)
    llm = ReplySequence(parsed)
    result = execute_json_reply(parts, llm)
    assert result["output_error"] and len(llm.calls) == 1


def test_model_client_keeps_the_incomplete_reason():
    result = AzureResponsesClient._parse({
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"days":['}]}],
        "usage": {"input_tokens": 100, "output_tokens": 6000},
    }, 15.0)
    assert result.status == "incomplete" and result.incomplete_reason == "max_output_tokens"
    assert result.output_tokens == 6000 and result.text == '{"days":['

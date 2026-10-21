"""What differs between the agents: their tools, prompts, plan shape and how a final answer is validated.

The routing, comparison, quality check and escalation machinery in agent.py is shared. A domain only supplies the
store or world the agent works in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from agent_demo import support, travel
from agent_demo.web import PublicWeb

DATA = Path(__file__).parent / "data"
TRAVEL_BUDGETS = ("budget", "mid-range", "premium")


@dataclass(frozen=True)
class Domain:
    id: str
    title: str
    tools: list[dict[str, Any]]
    router_tools: list[dict[str, Any]]
    responses_tools: list[dict[str, Any]]
    new_shared: Callable[[], Any]
    make_desk: Callable[[Any, Any], Any]
    planner_system: str
    tools_system: str
    reply_system: str
    judge_system: str
    scenarios_file: str
    clarify_system: str = ("Essential details are missing, so the request cannot be carried out as written. Reply with "
                           "one short, friendly question that asks for exactly what is missing.")
    general_system: str = ("Answer briefly and directly. If the request is outside what this agent can do, say so "
                           "plainly instead of pretending.")
    max_steps: int = 5
    parallel: bool = False
    reply_json: bool = False
    reply_tokens: int = 1500
    baseline_tokens: int = 3000
    evidence_chars: int = 3500
    summary_chars: int = 700
    stakes: re.Pattern[str] | None = None
    validate_reply: Callable[[str], str | None] | None = None

    def scenarios(self) -> list[dict[str, Any]]:
        return json.loads((DATA / self.scenarios_file).read_text(encoding="utf-8-sig"))


# ---------------------------------------------------------------------------------------------- support
SUPPORT = Domain(
    id="support", title="Support desk", tools=support.TOOLS, router_tools=support.router_tools(),
    responses_tools=support.responses_tools(), new_shared=lambda: None,
    make_desk=lambda llm, shared: support.SupportDesk(), scenarios_file="scenarios.json", max_steps=5,
    planner_system=(
        "You plan how a support agent for " + support.STORE + " handles one customer message. Break it into 1 to "
        "{max_steps} ordered, self-contained steps. A different agent runs each step with these tools and cannot see "
        "the other steps, so every instruction must include the order numbers, emails, amounts and details it needs: "
        "{tools}. Prefer few steps; combine lookups that belong together. The final step must write the reply to the "
        "customer and be marked \"reply\": true; no other step is a reply. If essential details such as an order "
        "number are missing and no tool can find them, use a single reply step that asks the customer for exactly "
        "what is missing. For each step give: title (at most 8 words), instruction, reply (true or false). "
        "Return only JSON: {{\"steps\": [{{\"title\": \"...\", \"instruction\": \"...\", \"reply\": false}}]}}."),
    tools_system=(
        "You are a support agent for " + support.STORE + " with tools. Complete only the step you are given. Never "
        "guess data a tool can provide. Before any refund or store credit, check the store policy with "
        "search_help_center and follow it exactly, including approval limits. When policy says a case must go to a "
        "human specialist (for example warranty claims, or amounts over your limits), call escalate_to_human. Report "
        "concisely what you found or did, with exact order numbers and dollar amounts."),
    reply_system=(
        "You write the final reply to the customer for " + support.STORE + " support. Use only facts established in "
        "the earlier steps; never invent order data, policy or promises. Be warm and concise (under 130 words), plain "
        "text with no placeholders or markdown. Answer everything the customer asked, say what was done, and what "
        "happens next. If something cannot be done, say why and offer the right alternative."),
    judge_system=(
        "You grade a support agent's work for " + support.STORE + " against the store policy and the evidence of tool "
        "calls. Rate how well the OUTPUT completes the STEP from 1 to 5: 5 correct, complete and policy-compliant; 4 "
        "correct with a minor gap; 3 acceptable but with a real gap; 2 wrong in an important way; 1 wrong, harmful or "
        "off-task. Facts must match the tool results; amounts must follow policy exactly. For a customer reply also "
        "judge tone, clarity and whether everything asked was answered. Also say where any fault lies: "
        "\"this_output\" if the OUTPUT itself is at fault, or \"earlier_work\" if the OUTPUT is faithful to what was "
        "done but earlier steps (for example a wrong refund amount) broke policy. Return only JSON: "
        "{\"score\": <1-5>, \"reason\": \"<one short sentence>\", \"fault\": \"this_output\"}.\n\nSTORE POLICY:\n"
        + support.POLICY_TEXT),
    stakes=None,
)


# ---------------------------------------------------------------------------------------------- travel
def validate_itinerary(text: str) -> str | None:
    """A structural check: the itinerary must be JSON with a day-by-day plan, so the screen can show it."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return "The itinerary was not valid JSON."
    if not isinstance(data, dict) or not isinstance(data.get("days"), list) or not data["days"]:
        return "The itinerary has no day-by-day plan."
    if not all(isinstance(d, dict) and isinstance(d.get("items"), list) for d in data["days"]):
        return "The itinerary days are not in the expected shape."
    return None


def travel_message(form: dict[str, Any], today: date | None = None) -> str:
    """Validate the planner form and turn it into the request the agent plans from."""
    today = today or date.today()
    if not isinstance(form, dict):
        raise ValueError("The trip details must be an object")
    text = {k: str(form.get(k) or "").strip() for k in ("origin", "destination", "notes")}
    for key, limit in (("origin", 60), ("destination", 60), ("notes", 400)):
        if len(text[key]) > limit:
            raise ValueError(f"{key} is too long (at most {limit} characters)")
    try:
        start, end = date.fromisoformat(str(form.get("start_date"))), date.fromisoformat(str(form.get("end_date")))
    except ValueError as error:
        raise ValueError("Choose a start and end date") from error
    if start < today:
        raise ValueError("The start date is in the past")
    if end < start:
        raise ValueError("The end date is before the start date")
    if (end - start).days > travel.MAX_TRIP_DAYS:
        raise ValueError(f"Trips are limited to {travel.MAX_TRIP_DAYS} days")
    travelers = form.get("travelers", 2)
    if isinstance(travelers, bool) or not isinstance(travelers, int) or not 1 <= travelers <= 9:
        raise ValueError("Travelers must be a number from 1 to 9")
    budget = form.get("budget", "mid-range")
    if budget not in TRAVEL_BUDGETS:
        raise ValueError(f"Budget must be one of {list(TRAVEL_BUDGETS)}")
    interests = form.get("interests") or []
    if not isinstance(interests, list) or any(i not in travel.INTERESTS for i in interests):
        raise ValueError(f"Interests must be chosen from {list(travel.INTERESTS)}")
    if not text["origin"]:
        raise ValueError("Say where the trip starts, so flights can be found")
    nights = (end - start).days
    where = (f"Destination: {text['destination']}." if text["destination"]
             else "Destination: not decided. Choose the best fit for these dates and interests.")
    parts = [f"Plan a {nights}-night trip for {travelers} traveler{'s' if travelers != 1 else ''}.",
             f"Depart from {text['origin']} on {start.isoformat()} and return on {end.isoformat()}.", where,
             f"Budget level: {budget}."]
    if interests:
        parts.append("Interests: " + ", ".join(interests) + ".")
    if text["notes"]:
        parts.append("Notes: " + text["notes"])
    return " ".join(parts)


def _new_travel_desk(llm: Any, shared: Any, web: PublicWeb = PublicWeb()) -> Any:  # noqa: B008 - one shared client and cache
    return travel.TravelDesk(llm, "gpt-4.1", web, shared)


TRAVEL = Domain(
    id="travel", title="Travel planner", tools=travel.TOOLS, router_tools=travel.router_tools(),
    responses_tools=travel.responses_tools(), new_shared=travel.SharedResults, make_desk=_new_travel_desk,
    scenarios_file="travel_scenarios.json", max_steps=8, parallel=True, reply_json=True, reply_tokens=6000,
    baseline_tokens=6000, evidence_chars=9000, summary_chars=2200,
    stakes=re.compile(r"\b(hold|book|reserve)\w*\b", re.I), validate_reply=validate_itinerary,
    planner_system=(
        "You plan a trip as a travel concierge. Break the request into 3 to {max_steps} steps run by specialist "
        "sub-agents with these tools: {tools}. A sub-agent cannot see other steps, so every instruction must repeat "
        "the destination, dates, origin, travelers, budget level and interests it needs. Shape of a good plan: "
        "(1) if no destination is given, the first step chooses one with suggest_destinations and get_weather "
        "(sequential); (2) then independent research steps that can run at the same time, each marked "
        "\"parallel\": true with its specialist (flights, hotels, sights, dining, weather): flights via "
        "search_flights, hotels via find_hotels, sights and restaurants via search_web (and the weather via "
        "get_weather); each must report exact offer ids, prices and source links; (3) one sequential step, specialist "
        "booking, that places holds on the best flight and hotel using the offer ids found earlier; (4) the final step "
        "writes the itinerary, marked \"reply\": true. Never mark the first, booking or final step parallel. For each "
        "step give: title (at most 8 words), instruction, specialist, parallel (true or false), reply (true or false). "
        "Return only JSON: {{\"steps\": [{{\"title\": \"...\", \"instruction\": \"...\", \"specialist\": \"flights\", "
        "\"parallel\": true, \"reply\": false}}]}}."),
    tools_system=(
        "You are a travel research specialist with tools. Complete only the step you are given. Never guess what a "
        "tool can look up. Search the web for sights, restaurants and current details, and keep the source links the "
        "tools return. Use web search sparingly: one or two well-aimed searches per step, then answer. Flights and "
        "hotel rates come from simulated or estimated data: say so. Report concisely and "
        "include exact offer ids, prices in USD, names, dates and source URLs, because later steps rely on them."),
    reply_system=(
        "You write the final trip itinerary as JSON only. Use only facts from the earlier steps: never invent "
        "flights, hotels, prices, places or links, and only cite URLs that appeared in the earlier steps. Be honest "
        "that flight fares, hotel rates and holds are simulated or estimated. Cover every date from departure to "
        "return, one entry per day. Schema: {\"title\": str, \"destination\": str, \"dates\": str, \"summary\": str, "
        "\"why_this_destination\": str or null, \"flights\": {\"offer_id\": str, \"carrier\": str, \"outbound\": str, "
        "\"return\": str, \"total_usd\": number, \"held\": bool} or null, \"hotel\": {\"offer_id\": str, \"name\": str, "
        "\"stars\": str, \"nights\": number, \"total_usd\": number, \"held\": bool} or null, \"weather\": str, "
        "\"days\": [{\"day\": number, \"date\": str, \"title\": str, \"items\": [{\"time\": str, \"kind\": \"sight\"|"
        "\"food\"|\"transport\"|\"rest\", \"title\": str, \"detail\": str, \"url\": str or null}]}], "
        "\"budget\": {\"flights_usd\": number, \"hotel_usd\": number, \"food_and_activities_usd\": number, "
        "\"total_usd\": number, \"note\": str}, \"tips\": [str], \"sources\": [str], \"caveats\": [str]}."),
    judge_system=(
        "You grade a trip planner's work against the evidence of tool calls. Rate how well the OUTPUT completes the "
        "STEP from 1 to 5: 5 complete, accurate and honest; 4 correct with a minor gap; 3 acceptable but with a real "
        "gap; 2 wrong in an important way; 1 invented, invalid or off-task. For research steps, facts must match the "
        "tool results and ids, prices and links must be reported exactly. For the final itinerary check: every date "
        "from departure to return is covered; the flight and hotel match the tool results and what was held; places "
        "and links come from the earlier results rather than being invented; the weather and budget level are "
        "respected; simulated or estimated data is labelled as such. Also say where any fault lies: \"this_output\" "
        "if the OUTPUT itself is at fault, or \"earlier_work\" if it is faithful to earlier steps that were wrong. "
        "Return only JSON: {\"score\": <1-5>, \"reason\": \"<one short sentence>\", \"fault\": \"this_output\"}."),
)

DOMAINS: dict[str, Domain] = {SUPPORT.id: SUPPORT, TRAVEL.id: TRAVEL}

"""Provider-independent display labels; lane IDs stay stable for stored runs and API clients."""

from collections.abc import Callable
from typing import Any

LANE_LABELS = {"clef": "Adaptive routing", "baseline": "Always strongest"}
UI_COPY = {
    "decision_model": "Decision model",
    "decision_prompt": "Exact decision-model prompt",
    "routing_cost_notice": (
        "Routing costs are estimates based on the configured billing model, not invoices. "
        "Retained hosted endpoints can incur separate compute charges."
    ),
}


def label_events(emit: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
    def send(event: dict[str, Any]) -> None:
        lane = event.get("lane")
        emit({**event, "lane_label": LANE_LABELS[lane]} if lane in LANE_LABELS else event)

    return send

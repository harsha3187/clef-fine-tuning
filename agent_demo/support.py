"""The Alder & Co. support desk: a small, deterministic, in-memory store the agent works against.

Nothing here touches a real system. Every run (and every lane of a run) gets its own fresh desk, so the two
lanes of a comparison cannot affect each other, and `check` can say whether the right thing actually happened.
"""

from __future__ import annotations

import copy
import threading
from datetime import date
from typing import Any

STORE = "Alder & Co."
TODAY = "2026-10-08"

_CUSTOMERS = {
    "C-201": {"id": "C-201", "name": "Priya Nair", "email": "priya.nair@example.com", "tier": "gold",
              "orders": ["A-1042", "A-1057"], "store_credit": 0.0},
    "C-202": {"id": "C-202", "name": "Marcus Webb", "email": "marcus.webb@example.com", "tier": "standard",
              "orders": ["A-1061", "A-1088"], "store_credit": 0.0},
    "C-203": {"id": "C-203", "name": "Elena Petrova", "email": "elena.petrova@example.com", "tier": "platinum",
              "orders": ["A-1075"], "store_credit": 0.0},
}
_ORDERS = {
    "A-1042": {"id": "A-1042", "customer_id": "C-201", "item": "Trail Runner jacket", "item_total": 129.00,
               "shipping_fee": 9.99, "status": "delivered", "placed": "2026-09-27", "promised": "2026-10-03",
               "delivered": "2026-10-02", "address": "22 Quay Street, Manchester M3 4LY", "category": "apparel",
               "refunded": 0.0},
    "A-1057": {"id": "A-1057", "customer_id": "C-201", "item": "Ceramic pour-over kettle", "item_total": 64.50,
               "shipping_fee": 5.99, "status": "shipped", "placed": "2026-10-04", "promised": "2026-10-10",
               "delivered": None, "address": "22 Quay Street, Manchester M3 4LY", "category": "home",
               "refunded": 0.0},
    "A-1061": {"id": "A-1061", "customer_id": "C-202", "item": "Oak standing desk", "item_total": 389.00,
               "shipping_fee": 0.0, "status": "delivered", "placed": "2026-08-23", "promised": "2026-08-28",
               "delivered": "2026-08-28", "address": "9 Mill Lane, Leeds LS2 8JB", "category": "furniture",
               "refunded": 0.0},
    "A-1075": {"id": "A-1075", "customer_id": "C-203", "item": "Dual-boiler espresso machine",
               "item_total": 549.00, "shipping_fee": 24.99, "status": "delayed", "placed": "2026-09-18",
               "promised": "2026-09-28", "delivered": None, "address": "5 Elm Court, Bristol BS1 5TT",
               "category": "appliances", "refunded": 0.0},
    "A-1088": {"id": "A-1088", "customer_id": "C-202", "item": "Waxed canvas backpack", "item_total": 94.00,
               "shipping_fee": 5.99, "status": "processing", "placed": "2026-10-07", "promised": "2026-10-13",
               "delivered": None, "address": "9 Mill Lane, Leeds LS2 8JB", "category": "bags", "refunded": 0.0},
}
_TRACKING = {
    "A-1057": {"carrier": "ParcelNorth", "status": "in transit", "eta": "2026-10-10",
               "events": ["2026-10-05 collected from warehouse", "2026-10-07 arrived at Manchester hub",
                          "2026-10-08 out for regional delivery"]},
    "A-1075": {"carrier": "ParcelNorth", "status": "held at carrier depot", "eta": "unknown",
               "events": ["2026-09-21 collected from warehouse", "2026-09-26 held at carrier depot",
                          "2026-10-06 still held, no release date from carrier"]},
}
_ARTICLES = [
    {"id": "HC-1", "title": "Returns and refunds", "keywords": "return refund damaged broken faulty money back",
     "text": "Items can be returned within 30 days of delivery for a full refund of the item price to the original "
             "payment method. An item that arrives damaged or faulty is refunded in full including the shipping "
             "fee. After 30 days no return refund is possible."},
    {"id": "HC-2", "title": "Late deliveries", "keywords": "late delay delayed shipping compensation credit slow",
     "text": "If an order arrives more than 7 days after the promised date, refund the shipping fee and add store "
             "credit worth 10% of the item price (15% for gold members, 20% for platinum members). Compensation is "
             "only due for orders more than 7 days late."},
    {"id": "HC-3", "title": "Address changes and cancellations", "keywords": "address change cancel cancellation",
     "text": "The shipping address can be changed and an order cancelled only while its status is processing. Once "
             "shipped, neither is possible; offer a return after delivery instead."},
    {"id": "HC-4", "title": "Approval limits", "keywords": "approval limit refund credit large manager",
     "text": "Agents may refund up to $200 per order and add up to $150 of store credit. Anything above these limits "
             "must go to a human specialist with escalate_to_human."},
    {"id": "HC-5", "title": "Warranty claims", "keywords": "warranty defect stopped working motor furniture appliance",
     "text": "Furniture and appliances carry a 12-month warranty against defects. A warranty claim is not a return: "
             "do not refund after the 30-day return window. Escalate a warranty claim to a human specialist with "
             "normal priority and tell the customer to expect contact within 2 business days."},
]
POLICY_TEXT = "\n".join(f"{a['id']} {a['title']}: {a['text']}" for a in _ARTICLES)

TOOLS: list[dict[str, Any]] = [
    {"name": "get_order", "description": "Get the details of an order: item, price, status, dates and address.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "Order number such as A-1042."}}, "required": ["order_id"]}},
    {"name": "track_shipment", "description": "Get the carrier status, ETA and scan events for a shipped order.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "Order number such as A-1057."}}, "required": ["order_id"]}},
    {"name": "get_customer", "description": "Look up a customer profile, membership tier and orders.",
     "parameters": {"type": "object", "properties": {
         "email_or_id": {"type": "string", "description": "Customer email address or customer id such as C-201."}},
         "required": ["email_or_id"]}},
    {"name": "search_help_center", "description": "Search the help center for store policy articles.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Keywords such as returns, late delivery or warranty."}},
         "required": ["query"]}},
    {"name": "check_return_eligibility", "description": "Check whether an order can still be returned for a refund.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "Order number."}}, "required": ["order_id"]}},
    {"name": "issue_refund", "description": "Refund money to the customer's original payment method.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "Order number."},
         "amount": {"type": "number", "description": "Amount in dollars to refund."},
         "reason": {"type": "string", "description": "Short reason for the refund."}},
         "required": ["order_id", "amount", "reason"]}},
    {"name": "add_store_credit", "description": "Add store credit to a customer's account.",
     "parameters": {"type": "object", "properties": {
         "customer_id": {"type": "string", "description": "Customer id such as C-203."},
         "amount": {"type": "number", "description": "Credit in dollars."},
         "reason": {"type": "string", "description": "Short reason for the credit."}},
         "required": ["customer_id", "amount", "reason"]}},
    {"name": "update_shipping_address", "description": "Change the delivery address of an order that has not shipped.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "Order number."},
         "new_address": {"type": "string", "description": "Full new delivery address."}},
         "required": ["order_id", "new_address"]}},
    {"name": "cancel_order", "description": "Cancel an order that has not shipped.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "Order number."}}, "required": ["order_id"]}},
    {"name": "escalate_to_human", "description": "Hand a case to a human specialist with a summary.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "Order number the case is about."},
         "summary": {"type": "string", "description": "What happened and what the customer wants."},
         "priority": {"type": "string", "enum": ["normal", "high"], "description": "Urgency of the case."}},
         "required": ["order_id", "summary", "priority"]}},
]
REFUND_LIMIT, CREDIT_LIMIT = 200.0, 150.0


def _days_between(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


class SupportDesk:
    """One run's copy of the store. Records every tool call so outcomes can be checked afterwards."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.customers = copy.deepcopy(_CUSTOMERS)
        self.orders = copy.deepcopy(_ORDERS)
        self.calls: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            result: dict[str, Any] = {"error": f"Unknown tool {name}"}
        else:
            try:
                with self._lock:
                    result = handler(**args)
            except TypeError as error:
                result = {"error": f"Bad arguments for {name}: {error}"}
            except ValueError as error:
                result = {"error": str(error)}
        self.calls.append({"name": name, "arguments": args, "result": result, "ok": "error" not in result})
        return result

    def _order(self, order_id: str) -> dict[str, Any]:
        order = self.orders.get(str(order_id).strip().upper())
        if order is None:
            raise ValueError(f"No order {order_id}")
        return order

    def _tool_get_order(self, order_id: str) -> dict[str, Any]:
        return dict(self._order(order_id))

    def _tool_track_shipment(self, order_id: str) -> dict[str, Any]:
        order = self._order(order_id)
        if order["status"] == "processing":
            return {"order_id": order["id"], "status": "not shipped yet", "eta": order["promised"], "events": []}
        if order["status"] == "delivered":
            return {"order_id": order["id"], "status": "delivered", "eta": order["delivered"], "events": []}
        return {"order_id": order["id"], **_TRACKING[order["id"]]}

    def _tool_get_customer(self, email_or_id: str) -> dict[str, Any]:
        key = email_or_id.strip().lower()
        for customer in self.customers.values():
            if key in (customer["id"].lower(), customer["email"].lower()):
                return dict(customer)
        raise ValueError(f"No customer {email_or_id}")

    def _tool_search_help_center(self, query: str) -> dict[str, Any]:
        words = {w for w in query.lower().replace(",", " ").split() if len(w) > 2}
        scored = sorted(((sum(w in (a["title"] + " " + a["keywords"]).lower() for w in words), a)
                         for a in _ARTICLES), key=lambda pair: -pair[0])
        return {"articles": [{"id": a["id"], "title": a["title"], "text": a["text"]}
                             for score, a in scored if score > 0][:3]}

    def _tool_check_return_eligibility(self, order_id: str) -> dict[str, Any]:
        order = self._order(order_id)
        if order["delivered"] is None:
            return {"order_id": order["id"], "eligible": False, "reason": "The order has not been delivered."}
        days = _days_between(order["delivered"], TODAY)
        return {"order_id": order["id"], "eligible": days <= 30, "days_since_delivery": days, "window_days": 30,
                "reason": "Inside the 30-day return window." if days <= 30 else "Outside the 30-day return window."}

    def _tool_issue_refund(self, order_id: str, amount: float, reason: str) -> dict[str, Any]:
        order = self._order(order_id)
        amount = round(float(amount), 2)
        paid = round(order["item_total"] + order["shipping_fee"], 2)
        if amount <= 0 or amount > paid - order["refunded"] + 1e-9:
            raise ValueError(f"Refund must be above 0 and at most ${paid - order['refunded']:.2f} for this order")
        if amount > REFUND_LIMIT:
            raise ValueError(f"Refunds over ${REFUND_LIMIT:.0f} need human approval; use escalate_to_human")
        order["refunded"] = round(order["refunded"] + amount, 2)
        return {"refunded": amount, "order_id": order["id"], "reason": reason}

    def _tool_add_store_credit(self, customer_id: str, amount: float, reason: str) -> dict[str, Any]:
        customer = self.customers.get(str(customer_id).strip().upper())
        if customer is None:
            raise ValueError(f"No customer {customer_id}")
        amount = round(float(amount), 2)
        if amount <= 0 or amount > CREDIT_LIMIT:
            raise ValueError(f"Store credit must be above 0 and at most ${CREDIT_LIMIT:.0f} without approval")
        customer["store_credit"] = round(customer["store_credit"] + amount, 2)
        return {"credited": amount, "customer_id": customer["id"], "balance": customer["store_credit"],
                "reason": reason}

    def _tool_update_shipping_address(self, order_id: str, new_address: str) -> dict[str, Any]:
        order = self._order(order_id)
        if order["status"] != "processing":
            raise ValueError(f"Order {order['id']} is {order['status']}; the address can only change while processing")
        order["address"] = new_address.strip()
        return {"updated": order["id"], "address": order["address"]}

    def _tool_cancel_order(self, order_id: str) -> dict[str, Any]:
        order = self._order(order_id)
        if order["status"] != "processing":
            raise ValueError(f"Order {order['id']} is {order['status']}; only processing orders can be cancelled")
        order["status"] = "cancelled"
        return {"cancelled": order["id"]}

    def _tool_escalate_to_human(self, order_id: str, summary: str, priority: str) -> dict[str, Any]:
        order = self._order(order_id)
        if priority not in ("normal", "high"):
            raise ValueError("priority must be normal or high")
        case = {"case_id": f"ESC-{len(self.escalations) + 301}", "order_id": order["id"], "priority": priority,
                "summary": summary}
        self.escalations.append(case)
        return {"escalated": case["case_id"], "priority": priority}

    # -------------------------------------------------------------- outcome checks

    def changes(self) -> list[str]:
        """Plain-language list of what this desk's state-changing calls actually did."""
        notes = []
        for call in self.calls:
            if not call["ok"]:
                continue
            r, a = call["result"], call["arguments"]
            if call["name"] == "issue_refund":
                notes.append(f"Refunded ${r['refunded']:.2f} on {r['order_id']}")
            elif call["name"] == "add_store_credit":
                notes.append(f"Added ${r['credited']:.2f} store credit to {r['customer_id']}")
            elif call["name"] == "update_shipping_address":
                notes.append(f"Changed the address on {r['updated']}")
            elif call["name"] == "cancel_order":
                notes.append(f"Cancelled {r['cancelled']}")
            elif call["name"] == "escalate_to_human":
                notes.append(f"Escalated {a.get('order_id')} to a human ({r['priority']})")
        return notes

    def check(self, expectations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return check_expectations(self.calls, expectations)


def check_expectations(calls: list[dict[str, Any]], expectations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each expectation is a tool call that must (or must not) have succeeded, with arguments to match."""
    results = []
    for item in expectations:
        matched = [c for c in calls if c["name"] == item["tool"] and c["ok"]
                   and all(_matches(c["arguments"].get(k), v) for k, v in item.get("args", {}).items())]
        results.append({"label": item["label"], "passed": bool(matched) == (not item.get("absent", False))})
    return results


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None:
        return actual == expected
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) < 0.015
        except (TypeError, ValueError):
            return False
    return str(expected).strip().lower() in str(actual).strip().lower()


def responses_tools() -> list[dict[str, Any]]:
    """The roster in the Responses API function-calling format."""
    return [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in TOOLS]


def router_tools() -> list[dict[str, Any]]:
    """The same roster in the BFCL-style shape the fine-tuned router was trained on."""
    return [{"description": t["description"], "name": t["name"],
             "parameters": {"properties": t["parameters"]["properties"],
                            "required": t["parameters"].get("required", []), "type": "dict"}} for t in TOOLS]

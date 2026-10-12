"""Harder tool-use tasks on the support desk: several requests in one message, policy traps, and no order numbers.

The first benchmark's tool tasks were all easy for every endpoint, so the selector had no evidence about tool use at
medium or high difficulty. These tasks are graded the same way: by what the tool calls actually did to the store.
The expected outcomes follow the store's own policy articles (return window, late-delivery compensation, approval
limits, warranty), computed from the fixed store data, so the checks never rely on a model's opinion.

Levels: 1 one request with the order number; 2 one request without it; 3 two requests; 4 two requests plus a policy
question and out-of-scope chatter; 5 level 4 with the customer identified only by email.
"""

from __future__ import annotations

import random
from typing import Any

from capability_lab.tasks import Task

CUSTOMERS = {
    "priya": {"name": "Priya Nair", "email": "priya.nair@example.com"},
    "marcus": {"name": "Marcus Webb", "email": "marcus.webb@example.com"},
    "elena": {"name": "Elena Petrova", "email": "elena.petrova@example.com"},
}

# Each intent concerns one order: how the customer names the item, ways to say it, and the outcomes policy requires.
INTENTS: dict[str, list[dict[str, Any]]] = {
    "priya": [
        {"order": "A-1042", "item": "Trail Runner jacket", "texts": [
            "the zip on {it} is broken and I would like my money back",
            "{it} arrived with a split seam, please refund me"],
         "expect": [{"label": "Full refund of $138.99 on A-1042 (damaged: item and shipping)", "tool": "issue_refund",
                     "args": {"order_id": "A-1042", "amount": 138.99}}]},
        {"order": "A-1042", "item": "Trail Runner jacket", "texts": [
            "I would like to return {it}, it just does not fit, and get a refund",
            "please take back {it}, I changed my mind, and refund it"],
         "expect": [{"label": "Item price $129.00 refunded on A-1042 (change of mind: no shipping)", "tool": "issue_refund",
                     "args": {"order_id": "A-1042", "amount": 129.0}}]},
        {"order": "A-1057", "item": "ceramic kettle", "texts": [
            "could you tell me where {it} is, and when it will arrive",
            "I want a status on {it}, has it left the warehouse"],
         "expect": [{"label": "Shipment tracked for A-1057", "tool": "track_shipment", "args": {"order_id": "A-1057"}}]},
        {"order": "A-1057", "item": "ceramic kettle", "texts": [
            "please cancel {it}, I no longer want it", "cancel {it} before it reaches me"],
         "expect": [{"label": "A-1057 not cancelled (already shipped)", "tool": "cancel_order", "absent": True}]},
    ],
    "marcus": [
        {"order": "A-1061", "item": "oak standing desk", "texts": [
            "the motor on {it} stopped working after about six weeks and I want a full refund",
            "{it} no longer raises, the motor is dead, and I expect my money back"],
         "expect": [{"label": "No refund outside the 30-day window", "tool": "issue_refund", "absent": True},
                    {"label": "Warranty claim escalated on A-1061", "tool": "escalate_to_human", "args": {"order_id": "A-1061"}}]},
        {"order": "A-1088", "item": "waxed canvas backpack", "texts": [
            "please change the delivery address of {it} to 14 Harbour Lane, Leeds LS1 4AB",
            "send {it} to 14 Harbour Lane, Leeds LS1 4AB instead"],
         "expect": [{"label": "Address on A-1088 changed", "tool": "update_shipping_address",
                     "args": {"order_id": "A-1088", "new_address": "Harbour Lane"}}]},
        {"order": "A-1088", "item": "waxed canvas backpack", "texts": [
            "please cancel {it}, I ordered it by mistake", "cancel {it} right away"],
         "expect": [{"label": "A-1088 cancelled", "tool": "cancel_order", "args": {"order_id": "A-1088"}}]},
    ],
    "elena": [
        {"order": "A-1075", "item": "espresso machine", "texts": [
            "{it} was promised for 28 September and has still not arrived, and I expect proper compensation",
            "{it} is still not here, which is unacceptable, so make it right"],
         "expect": [{"label": "Shipping fee $24.99 refunded on A-1075", "tool": "issue_refund",
                     "args": {"order_id": "A-1075", "amount": 24.99}},
                    {"label": "Store credit $109.80 added (20% platinum)", "tool": "add_store_credit",
                     "args": {"customer_id": "C-203", "amount": 109.8}}]},
        {"order": "A-1075", "item": "espresso machine", "texts": [
            "cancel {it} and refund the whole $573.99, I have waited long enough",
            "I want {it} cancelled and every cent back, all $573.99"],
         "expect": [{"label": "A-1075 not cancelled (not in processing)", "tool": "cancel_order", "absent": True},
                    {"label": "Over-limit request escalated on A-1075", "tool": "escalate_to_human", "args": {"order_id": "A-1075"}}]},
    ],
}
POLICY_QUESTIONS = [
    "what is your return window for items I have bought",
    "how much compensation do you give for late deliveries",
]
POLICY_EXPECT = {"label": "Help center searched", "tool": "search_help_center"}
CHATTER = ["Also, can you recommend a good tent for winter camping?", "By the way, do you sell gift cards?",
           "Unrelated, but is there a store near Leeds that stocks your jackets?"]
OPENERS = ["", "Hello support team,\n", "Hi there. ", "Good morning,\n"]


def _sentence(text: str) -> str:
    return text[0].upper() + text[1:] + "."


def compose(rng: random.Random, level: int, customers: tuple[str, ...] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """One customer message and the outcomes the store's policy requires for it."""
    key = rng.choice([k for k in INTENTS if customers is None or k in customers])
    person, pool = CUSTOMERS[key], INTENTS[key]
    orders = sorted({i["order"] for i in pool})
    wanted = 1 if level <= 2 else min(2, len(orders))
    intents = [rng.choice([i for i in pool if i["order"] == o]) for o in rng.sample(orders, wanted)]
    parts, expect = [], []
    for intent in intents:
        item = f"my {intent['item']}" + (f" (order {intent['order']})" if level == 1 else "")
        parts.append(rng.choice(intent["texts"]).format(it=item))
        expect += intent["expect"]
    if level >= 4 or (level == 3 and len(intents) == 1):  # a single-order customer's second request is a policy question
        parts.append(rng.choice(POLICY_QUESTIONS))
        expect.append(POLICY_EXPECT)
    who = f"({person['email']})" if level >= 5 else f"{person['name']} ({person['email']})"
    message = f"{rng.choice(OPENERS)}This is {who}. " + " ".join(_sentence(p) if i == 0 else _sentence("also, " + p) for i, p in enumerate(parts))
    if level >= 4:
        message += " " + rng.choice(CHATTER)
    return message, expect


def tool_tasks(count: int, seed: int = 11, *, family: str = "tool_chain", customers: tuple[str, ...] | None = None,
               avoid: set[str] | None = None) -> list[Task]:
    """Generated tool-use tasks with levels 1 to 5 cycling and unique messages, graded by desk state.

    `customers` limits whose requests appear; `avoid` is a set of messages that must not be generated again, so a
    training set can be kept apart from an evaluation set made by the same generator.
    """
    rng = random.Random(f"{seed}:{family}")
    tasks: list[Task] = []
    seen: set[str] = set(avoid or ())
    for _ in range(count * 50):
        if len(tasks) == count:
            break
        level = 1 + len(tasks) % 5
        message, expect = compose(rng, level, customers)
        if message in seen:
            continue
        seen.add(message)
        tasks.append(Task(id=f"{family}-{len(tasks) + 1:03d}", family=family, skill="tools", level=level, prompt=message,
                          verifier={"type": "desk", "expect": expect}, uses_tools=True))
    return tasks

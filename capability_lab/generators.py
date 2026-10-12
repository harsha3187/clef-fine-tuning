"""Deterministic task generators for the capability benchmark.

Each generator builds a task whose answer can be checked by code, with a `level` (1 to 3) that raises difficulty
through its parameters (more steps, more constraints, more distractors). The level is only a generation parameter
used for analysis. The tier label a task finally gets comes from how many endpoints actually solve it.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
from collections.abc import Callable
from typing import Any

NAMES = ["Priya Nair", "Marcus Webb", "Elena Petrova", "Tom Becker", "Aiko Tanaka", "Sara Lindqvist", "Omar Haddad", "Lucia Romero"]
CITIES = ["Manchester", "Leeds", "Bristol", "Glasgow", "Cardiff", "Leicester", "Norwich", "Sheffield"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
ONLY_JSON = "Reply with only the JSON."


def _ordinal(day: int) -> str:
    return f"{day}{'th' if 10 <= day % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')}"


# ------------------------------------------------------------------------------------------ extraction
def extract_fields(rng: random.Random, level: int) -> dict[str, Any]:
    name = rng.choice(NAMES)
    email = name.lower().replace(" ", ".") + "@example.com"
    order, other = f"A-{rng.randint(1000, 9999)}", f"A-{rng.randint(1000, 9999)}"
    amount = round(rng.uniform(1200, 4800) if level >= 4 else rng.uniform(20, 900), 2)
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    city = rng.choice(CITIES)
    pool = ["order_id", "email", "amount", "date", "city", "name"]
    chosen = rng.sample(pool, {1: 2, 2: 4, 3: 6, 4: 6}[level])
    wanted = [k for k in pool if k in chosen]
    values = {"order_id": order, "email": email, "amount": amount, "date": date, "city": city, "name": name}
    text = (f"Hello, this is {name} ({email}). I ordered on {date} (order {order}) and paid ${amount:,.2f}, "
            f"shipping to {city}. The jacket arrived with a broken zip.")
    if level >= 2:
        text += f" My earlier order {other} was fine."
    if level >= 3:
        text += (f" Please ignore my old work email {name.split()[0].lower()}@oldjob.example and the ${amount + 12.5:.2f} "
                 f"charge from last year, which is unrelated.")
    if level >= 4:
        text += (f" For context, my colleague {rng.choice(NAMES)} wrote separately about order A-{rng.randint(1000, 9999)} "
                 f"(${rng.randint(100, 900)}.50, shipped to {rng.choice(CITIES)} on 2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}); "
                 f"please do not mix the two cases up. I was actually charged ${amount + 49:,.2f} at first, and ${amount:,.2f} after the correction.")
    prompt = (f"Extract these fields about the order the customer is complaining about: {', '.join(wanted)}. "
              f"Return a JSON object with exactly those keys; dates as YYYY-MM-DD and the amount as a number.\n\n"
              f"Message:\n\"\"\"{text}\"\"\"\n{ONLY_JSON}")
    gold = {k: values[k] for k in wanted}
    return {"prompt": prompt, "verifier": {"type": "json", "value": gold}, "gold": json.dumps(gold)}


def reformat_dates(rng: random.Random, level: int) -> dict[str, Any]:
    count = {1: 3, 2: 6, 3: 10, 4: 16}[level]
    shown, iso = [], []
    for _ in range(count):
        y, m, d = 2026 + rng.randint(0, 1), rng.randint(1, 12), rng.randint(1, 28)
        iso.append(f"{y}-{m:02d}-{d:02d}")
        style = rng.randint(0, 4) if level > 1 else rng.randint(0, 2)
        shown.append([f"{MONTHS[m - 1]} {d}, {y}", f"{d} {MONTHS[m - 1][:3]} {str(y)[2:]}", f"{d:02d}/{m:02d}/{y}",
                      f"{y}.{m:02d}.{d:02d}", f"{_ordinal(d)} of {MONTHS[m - 1]}, {y}"][style])
    prompt = ("Convert each date to ISO format YYYY-MM-DD. Numeric dates with slashes are day/month/year and "
              "two-digit years are 20xx. Return a JSON list in the same order.\n\n" + "\n".join(f"- {s}" for s in shown)
              + f"\n{ONLY_JSON}")
    return {"prompt": prompt, "verifier": {"type": "json", "value": iso}, "gold": json.dumps(iso)}


PRIORITY = ["legal_threat", "refund_request", "address_change", "tracking", "praise", "other"]
SNIPPETS = {
    "legal_threat": ["If this is not fixed I will be speaking to my lawyer.", "My solicitor will be in touch about this."],
    "refund_request": ["I would like my money back.", "Please refund the full amount."],
    "address_change": ["Please change the delivery address to 14 Harbour Lane.", "I moved, so the parcel should go to 9 Mill Road."],
    "tracking": ["Where is my parcel at the moment?", "Can you tell me when it will arrive?"],
    "praise": ["Your staff were lovely last time.", "I really like the quality of your products."],
}
NEGATED = ["I am not asking for a refund, by the way.", "I have no plans to involve a lawyer."]


def classify_intent(rng: random.Random, level: int) -> dict[str, Any]:
    present = rng.sample(list(SNIPPETS), {1: 1, 2: 2, 3: 3, 4: 4}[level])
    parts = [rng.choice(SNIPPETS[i]) for i in present]
    if level >= 3:  # say, truthfully, what the customer is NOT asking for; it must not count as an intent
        if "refund_request" not in present:
            parts.append(NEGATED[0])
        elif "legal_threat" not in present:
            parts.append(NEGATED[1])
    if level >= 4 and "address_change" not in present:
        parts.append("I have no interest in changing the delivery address either.")
    rng.shuffle(parts)
    label = next(p for p in PRIORITY if p in present)
    prompt = ("Classify the customer message with exactly one label. If several intents are present, use the "
              "highest priority among these, from highest to lowest: " + ", ".join(PRIORITY[:-1])
              + ". A statement that the customer is NOT doing something does not count as that intent. Use "
              "'other' if none apply.\n\nMessage:\n\"\"\"" + " ".join(parts) + "\"\"\"\nReply with only the label.")
    return {"prompt": prompt, "verifier": {"type": "exact", "value": label}, "gold": label}


# ------------------------------------------------------------------------------------------- reasoning
def arithmetic_steps(rng: random.Random, level: int) -> dict[str, Any]:
    price, qty = rng.randint(12, 90), rng.randint(2, 9)
    total = float(price * qty)
    lines = [f"A kit costs ${price} and a customer buys {qty} of them."]
    for _ in range({1: 2, 2: 4, 3: 7, 4: 11, 5: 16}[level] - 1):
        kind = rng.choice(["discount", "tax", "shipping", "coupon"])
        if kind == "discount":
            pct = rng.choice([5, 10, 15, 20, 25])
            total *= 1 - pct / 100
            lines.append(f"A {pct}% discount is applied to the running total.")
        elif kind == "tax":
            pct = rng.choice([5, 8, 12, 20])
            total *= 1 + pct / 100
            lines.append(f"Then {pct}% sales tax is added to the running total.")
        elif kind == "shipping":
            fee = rng.randint(4, 25)
            total += fee
            lines.append(f"Then a flat ${fee} shipping fee is added.")
        else:
            off = rng.randint(3, 20)
            total -= off
            lines.append(f"Then a ${off} coupon is subtracted.")
    people = rng.randint(2, 5)
    total /= people
    lines.append(f"Finally the amount is split equally between {people} people. How much does each person pay?")
    prompt = " ".join(lines) + " Reply with only the amount in dollars, rounded to two decimals."
    return {"prompt": prompt, "verifier": {"type": "number", "value": round(total, 2), "tol": 0.01}, "gold": f"{total:.2f}"}


def rule_holds(rule: tuple, order: tuple[str, ...]) -> bool:
    n = len(order)
    p = {x: i for i, x in enumerate(order)}
    kind, a, b = rule
    return {"next": lambda: p[a] + 1 == p[b], "right": lambda: p[a] > p[b], "end": lambda: p[a] in (0, n - 1),
            "notend": lambda: p[a] not in (0, n - 1), "at": lambda: p[a] == b,
            "apart": lambda: abs(p[a] - p[b]) > 1}[kind]()


def _violated(rule: tuple, pos: dict[str, int], placed: int, n: int) -> bool:
    """True only when a partial left-to-right arrangement already breaks the rule for good."""
    kind, a, b = rule
    if kind == "at":
        return (a in pos and pos[a] != b) or (a not in pos and placed > b)
    if kind == "end":
        return a in pos and pos[a] not in (0, n - 1)
    if kind == "notend":
        return a in pos and pos[a] in (0, n - 1)
    if kind == "right":
        return (a in pos and b not in pos) or (a in pos and b in pos and not pos[a] > pos[b])
    if kind == "next":
        if a in pos and b in pos:
            return pos[a] + 1 != pos[b]
        return (b in pos and a not in pos) or (a in pos and b not in pos and placed > pos[a] + 1)
    return a in pos and b in pos and abs(pos[a] - pos[b]) <= 1  # apart


def count_solutions(people: list[str], rules: list[tuple], limit: int = 2) -> int:
    """How many arrangements satisfy every rule, counted by pruned search and stopped at `limit`."""
    n = len(people)
    pos: dict[str, int] = {}
    found = 0

    def place(index: int) -> None:
        nonlocal found
        if found >= limit:
            return
        if index == n:
            found += all(rule_holds(r, tuple(sorted(pos, key=pos.get))) for r in rules)
            return
        for person in people:
            if person in pos:
                continue
            pos[person] = index
            if not any(_violated(r, pos, index + 1, n) for r in rules):
                place(index + 1)
            del pos[person]

    place(0)
    return found


def ordering_puzzle(rng: random.Random, level: int) -> dict[str, Any]:
    people = ["Ana", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hal", "Ivy"][:{1: 4, 2: 5, 3: 7, 4: 8, 5: 9}[level]]
    n = len(people)
    truth = people[:]
    rng.shuffle(truth)
    pos = {p: i for i, p in enumerate(truth)}
    rules: list[tuple] = []
    for _ in range(80):
        a, b = rng.sample(people, 2)
        kind = rng.choice(["next", "right", "notend", "at", "apart", "end"])
        rule = (kind, a, pos[a] if kind == "at" else b)
        if not rule_holds(rule, tuple(truth)) or rule in rules:
            continue
        rules.append(rule)
        if count_solutions(people, rules) == 1:
            break
    for person in people:  # a puzzle must have exactly one answer; pin people until it does
        if count_solutions(people, rules) == 1:
            break
        rules.append(("at", person, pos[person]))
    text = {"next": "{a} stands immediately to the left of {b}.", "right": "{a} stands somewhere to the right of {b}.",
            "end": "{a} stands at one of the two ends.", "notend": "{a} does not stand at either end.",
            "at": "{a} stands in position {b1}.", "apart": "{a} and {b} are not next to each other."}
    facts = [text[k].format(a=a, b=b, b1=(b + 1 if k == "at" else b)) for k, a, b in rules]
    prompt = (f"{n} people stand in a line, positions 1 to {n} from left to right. Facts:\n"
              + "\n".join(f"- {f}" for f in facts) + f"\nWho stands in each position? Reply with only a JSON list of the "
              f"names from position 1 to position {n}.")
    return {"prompt": prompt, "verifier": {"type": "json", "value": truth}, "gold": json.dumps(truth),
            "meta": {"people": people, "rules": [list(r) for r in rules]}}


def schedule_csp(rng: random.Random, level: int) -> dict[str, Any]:
    tasks = ["Backup", "Deploy", "Audit", "Report", "Train", "Review", "Patch", "Migrate", "Cleanup", "Archive", "Index", "Notify"][:{1: 4, 2: 6, 3: 9, 4: 12}[level]]
    slots = ["Mon", "Tue", "Wed", "Thu", "Fri"][:{1: 3, 2: 4, 3: 5, 4: 5}[level]]
    truth = {t: rng.choice(slots) for t in tasks}
    constraints: list[list] = []
    for _ in range(len(tasks) + level * 2 + (6 if level >= 4 else 0)):
        a, b = rng.sample(tasks, 2)
        sa, sb = slots.index(truth[a]), slots.index(truth[b])
        options = []
        if sa < sb:
            options.append(["before", a, b])
        if sa != sb:
            options.append(["apart", a, b])
        if sa == sb:
            options.append(["together", a, b])
        options.append(["slot", a, truth[a]])
        options.append(["not_slot", a, rng.choice([s for s in slots if s != truth[a]])])
        constraints.append(rng.choice(options))
    words = {"before": "{a} must happen on an earlier day than {b}.", "apart": "{a} and {b} must be on different days.",
             "together": "{a} and {b} must be on the same day.", "slot": "{a} must be on {b}.", "not_slot": "{a} cannot be on {b}."}
    prompt = (f"Schedule {len(tasks)} tasks ({', '.join(tasks)}) onto these days: {', '.join(slots)}. Every task gets "
              "exactly one day, and several tasks may share a day. Rules:\n"
              + "\n".join("- " + words[k].format(a=a, b=b) for k, a, b in constraints)
              + "\nReply with only a JSON object mapping each task to its day.")
    return {"prompt": prompt, "verifier": {"type": "csp", "tasks": tasks, "slots": slots, "constraints": constraints},
            "gold": json.dumps(truth)}


POLICY = (
    "Store policy:\n1. Returns within 30 days of delivery are refunded the item price. If the item was opened and is "
    "not defective, a 15% restocking fee is deducted from the item price.\n2. Defective items are refunded in full, "
    "including the shipping fee, regardless of whether they were opened.\n3. After 30 days no return refund is possible, "
    "except for defective items, which are covered for 365 days.\n4. If an order arrived more than 7 days after the "
    "promised date, the customer also receives store credit of 10% of the item price (15% for gold members, 20% for "
    "platinum members). Credit is separate from the refund.\n5. Store credit is capped at $100 per order."
)


POLICY_EXTRA = (
    "\n6. Gold and platinum members are exempt from the restocking fee.\n7. If the customer used a discount code, the "
    "item price refunded is the discounted price they actually paid, and the code gave 20% off the listed price.\n8. A "
    "defective item returned after more than 30 days is refunded at 80% of the price paid, plus the shipping fee."
)


def policy_total(price: float, shipping: float, since: int, defective: bool, opened: bool, late: int, tier: str,
                 extended: bool = False, coded: bool = False) -> float:
    paid = price * 0.8 if (extended and coded) else price
    if defective:
        if since > 365:
            refund = 0.0
        elif extended and since > 30:
            refund = paid * 0.8 + shipping
        else:
            refund = paid + shipping
    elif since <= 30:
        fee_applies = opened and not (extended and tier in ("gold", "platinum"))
        refund = paid * (0.85 if fee_applies else 1.0)
    else:
        refund = 0.0
    credit = min(100.0, price * {"standard": 0.10, "gold": 0.15, "platinum": 0.20}[tier]) if late > 7 else 0.0
    return round(refund + credit, 2)


def policy_amount(rng: random.Random, level: int) -> dict[str, Any]:
    price, shipping = float(rng.randint(60, 420)), float(rng.choice([0, 5, 9.99, 12.5, 24.99]))
    if level == 1:
        case = dict(since=rng.randint(3, 25), defective=False, opened=True, late=0, tier="standard")
    elif level == 2:
        case = dict(since=rng.randint(3, 25), defective=rng.random() < 0.5, opened=True, late=rng.randint(8, 14), tier=rng.choice(["gold", "standard"]))
    else:
        case = dict(since=rng.randint(32, 200), defective=True, opened=True, late=rng.randint(8, 20), tier="platinum")
        price = float(rng.randint(500, 900))
    extended = level >= 4
    coded = False
    if extended:
        case = dict(since=rng.choice([10, 20, 45, 90]), defective=rng.random() < 0.6, opened=True, late=rng.randint(0, 14),
                    tier=rng.choice(["gold", "platinum", "standard"]))
        price, coded = float(rng.randint(150, 700)), rng.random() < 0.6
    total = policy_total(price, shipping, **case, extended=extended, coded=coded)
    facts = (f"Case: the item's listed price was ${price:.2f}{' and the customer used a discount code' if coded else ''}; "
             f"shipping was ${shipping:.2f}. It was delivered {case['since']} days ago, "
             f"{'it is defective' if case['defective'] else 'it is not defective'} and "
             f"{'it has been opened' if case['opened'] else 'it is unopened'}. The order arrived {case['late']} days "
             f"after the promised date. The customer is a {case['tier']} member and wants to return it now.")
    prompt = (POLICY + (POLICY_EXTRA if extended else "") + "\n\n" + facts + "\nWhat is the total in dollars returned to the customer (refund plus store credit)? "
              "Reply with only the amount.")
    return {"prompt": prompt, "verifier": {"type": "number", "value": total, "tol": 0.01}, "gold": f"{total:.2f}"}


# --------------------------------------------------------------------------------------------- coding
def _run(code: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, {"__builtins__": {"range": range, "len": len, "sorted": sorted, "print": print, "max": max, "min": min, "sum": sum, "enumerate": enumerate}})  # noqa: S102 - our own trusted snippets
    return buffer.getvalue().strip()


def code_trace(rng: random.Random, level: int) -> dict[str, Any]:
    if level == 1:
        n, m = rng.randint(10, 30), rng.randint(2, 5)
        code = f"total = 0\nfor i in range(1, {n}):\n    if i % {m} == 0:\n        total += i\nprint(total)"
    elif level == 2:
        words = rng.sample(["apple", "avocado", "banana", "blueberry", "cherry", "cranberry", "date", "dragonfruit", "fig", "grape"], 6)
        code = f"words = {words}\ncounts = {{}}\nfor w in words:\n    counts[w[0]] = counts.get(w[0], 0) + len(w)\nprint(sorted(counts.items()))"
    elif level == 4:
        data = [rng.randint(1, 9) for _ in range(rng.randint(6, 8))]
        code = (f"data = {data}\nout = []\nfor i, v in enumerate(data):\n    if i % 2 == 0:\n        out.append(v * (i + 1))\n"
                "    else:\n        out.insert(0, v - i)\n    if len(out) > 3:\n        out.pop()\nprint(out, sum(out[::2]))")
    else:
        n = rng.randint(6, 27)
        code = ("def f(n, acc=()):\n    if n <= 1:\n        return acc + (n,)\n    if n % 2 == 0:\n        return f(n // 2, acc + (n,))\n"
                f"    return f(3 * n + 1, acc + (n,))\nseq = f({n})\nprint(len(seq), max(seq), seq[-3:])")
    out = _run(code)
    prompt = f"What exactly does this Python program print? Reply with only the printed output.\n\n```python\n{code}\n```"
    return {"prompt": prompt, "verifier": {"type": "exact", "value": out}, "gold": out}


def code_bug_line(rng: random.Random, level: int) -> dict[str, Any]:
    variants = [
        (["def sum_even(values):", "    total = 0", "    for v in values:", "        if v % 2 == 0:", "            total += v", "    return total"],
         3, "    for v in range(1, len(values)):", None),
        (["def largest(values):", "    best = values[0]", "    for v in values:", "        if v > best:", "            best = v", "    return best"],
         3, "        if v < best:", None),
        (["def product(values):", "    result = 1", "    for v in values:", "        result *= v", "    return result"],
         1, "    result = 0", None),
        (["def average_positive(values):", "    total = 0", "    count = 0", "    for v in values:", "        if v > 0:", "            total += v", "            count += 1", "    return total / count"],
         4, "        if v >= 0:", None),
        (["def count_words(text):", "    words = text.split()", "    n = 0", "    for w in words:", "        n += 1", "    return n"],
         4, "        n += 2", None),
        (["def find(values, target):", "    lo, hi = 0, len(values) - 1", "    while lo <= hi:", "        mid = (lo + hi) // 2",
          "        if values[mid] == target:", "            return mid", "        if values[mid] < target:", "            lo = mid + 1",
          "        else:", "            hi = mid - 1", "    return -1"], 9, "            hi = mid + 1", None),
        (["def bubble(values):", "    n = len(values)", "    for i in range(n):", "        for j in range(n - i - 1):",
          "            if values[j] > values[j + 1]:", "                values[j], values[j + 1] = values[j + 1], values[j]", "    return values"],
         4, "            if values[j] < values[j + 1]:", None),
    ]
    lines, bug_index, bad_line, _ = rng.choice(variants)
    lines = lines[:]
    lines[bug_index] = bad_line
    padding = {1: 0, 2: 3, 3: 8, 4: 14}[level]
    for k in range(padding):  # harmless lines after the signature lengthen the code without changing it
        lines.insert(1, f"    note_{k} = {rng.randint(1, 99)}")
        bug_index += 1
    numbered = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
    prompt = ("Exactly one line of this function contains a bug that makes it return a wrong result. Which line number "
              f"(shown before the colon) contains it? Reply with only the number.\n\n{numbered}")
    return {"prompt": prompt, "verifier": {"type": "integer", "value": bug_index + 1}, "gold": str(bug_index + 1)}


# --------------------------------------------------------------------------------------------- tools
SCHEMAS = {
    "create_event": {"description": "Create a calendar event.", "parameters": {"title": "string", "date": "string, YYYY-MM-DD",
                     "duration_minutes": "integer", "attendees": "list of strings, optional", "reminder": "boolean, optional, default false"}},
    "search_flights": {"description": "Search flights.", "parameters": {"origin": "string", "destination": "string",
                       "depart_date": "string, YYYY-MM-DD", "cabin": "one of economy, business, first; default economy", "max_stops": "integer, default 1"}},
    "send_message": {"description": "Send a chat message.", "parameters": {"channel": "string", "text": "string", "urgent": "boolean, default false"}},
    "set_thermostat": {"description": "Set a room thermostat.", "parameters": {"room": "string", "temperature_c": "integer", "mode": "one of heat, cool, auto"}},
}


def _call(rng: random.Random, name: str, rich: bool, derived: bool = False) -> tuple[dict[str, Any], str]:
    y, m, d = 2027, rng.randint(1, 12), rng.randint(1, 26)
    iso, spoken = f"{y}-{m:02d}-{d:02d}", f"{MONTHS[m - 1]} {d}, {y}"
    if derived:  # the date must be computed: two days after a spoken date
        iso, spoken = f"{y}-{m:02d}-{d + 2:02d}", f"two days after {MONTHS[m - 1]} {d}, {y}"
    if name == "create_event":
        args = {"title": rng.choice(["Design review", "Sprint planning", "Customer call"]), "date": iso, "duration_minutes": rng.choice([30, 45, 60, 90])}
        text = f"Put '{args['title']}' on the calendar for {spoken}, lasting {args['duration_minutes']} minutes"
        if rich:
            args["attendees"] = rng.sample(["Ana", "Ben", "Cleo", "Dev"], 2)
            args["reminder"] = True
            text += f", with {args['attendees'][0]} and {args['attendees'][1]} invited and a reminder on"
    elif name == "search_flights":
        o, dst = rng.sample(["Dubai", "Lisbon", "Tokyo", "Rome", "Cairo"], 2)
        args = {"origin": o, "destination": dst, "depart_date": iso}
        text = f"Find flights from {o} to {dst} leaving {spoken}"
        if rich:
            args["cabin"] = "business"
            args["max_stops"] = 0
            text += ", business class, nonstop only"
    elif name == "send_message":
        args = {"channel": rng.choice(["ops", "sales", "support"]), "text": rng.choice(["Deploy is done", "Meeting moved to 3pm", "Lunch is here"])}
        text = f"Tell the {args['channel']} channel: {args['text']}"
        if rich:
            args["urgent"] = True
            text += ", and mark it urgent"
    else:
        args = {"room": rng.choice(["bedroom", "office", "kitchen"]), "temperature_c": rng.randint(17, 25), "mode": rng.choice(["heat", "cool", "auto"])}
        text = f"Set the {args['room']} to {args['temperature_c']} degrees in {args['mode']} mode"
    return {"name": name, "arguments": args}, text


def function_call(rng: random.Random, level: int) -> dict[str, Any]:
    names = rng.sample(list(SCHEMAS), {1: 1, 2: 1, 3: 2, 4: 3}[level])
    calls, texts = zip(*[_call(rng, n, rich=level >= 2, derived=level >= 4) for n in names], strict=True)
    request = " and ".join(texts) + "."
    prompt = ("You can call these tools:\n" + json.dumps(SCHEMAS, indent=1) + f"\n\nRequest: {request}\n"
              'Reply with only JSON of the form {"calls": [{"name": ..., "arguments": {...}}]}. Include only the '
              "arguments the request actually specifies, in the right types.")
    return {"prompt": prompt, "verifier": {"type": "calls", "value": list(calls)}, "gold": json.dumps({"calls": list(calls)})}


# -------------------------------------------------------------------------------------------- writing
def constrained_reply(rng: random.Random, level: int) -> dict[str, Any]:
    name, order = rng.choice(NAMES).split()[0], f"A-{rng.randint(1000, 9999)}"
    amount, date = f"${rng.randint(20, 400)}.{rng.randint(0, 99):02d}", f"{MONTHS[rng.randint(0, 11)]} {rng.randint(1, 28)}"
    spec: dict[str, Any] = {"must_include": [order, amount], "max_words": 90}
    rules = [f"mention the order number {order}", f"mention the refund amount {amount}", "use at most 90 words"]
    if level >= 2:
        spec.update(starts_with=f"Hi {name},", must_not=["sorry", "unfortunately"])
        rules += [f"start with exactly 'Hi {name},'", "never use the words 'sorry' or 'unfortunately'"]
    if level >= 3:
        spec["must_include"] += ["within 5 business days", date]
        spec.update(ends_with="Alder & Co. Support", min_words=40)
        rules += ["say the refund arrives within 5 business days", f"mention the date {date}", "end with 'Alder & Co. Support'", "use at least 40 words"]
    if level >= 4:
        case = f"CR-{rng.randint(1000, 9999)}"
        spec["must_include"] += [f"case {case}"]
        spec["must_not"] = spec.get("must_not", []) + ["regret", "apologies", "apologize"]
        spec.update(max_words=75, min_words=45)
        rules = [r for r in rules if not r.startswith("use at most")]
        rules += [f"quote the case reference as 'case {case}'", "never use the words 'regret', 'apologies' or 'apologize'",
                  "use between 45 and 75 words in total"]
    prompt = ("Write a short reply to a customer whose refund has been approved. The reply must: "
              + "; ".join(rules) + ". Reply with only the message.")
    gold = f"Hi {name}, your refund of {amount} for order {order} is approved and arrives within 5 business days. We confirmed it on {date}. Thank you for your patience. Please reply if anything looks wrong with the amount or the date shown, and we will check straight away so that you are never left waiting. Alder & Co. Support"
    if level >= 4:
        gold = (f"Hi {name}, your refund of {amount} for order {order} is approved and arrives within 5 business days. "
                f"We confirmed it on {date}. Your case {case} stays open until it arrives. Please reply to this message if the "
                "amount looks wrong, and we will check it straight away. Alder & Co. Support")
    return {"prompt": prompt, "verifier": {"type": "checks", **spec}, "gold": gold if level >= 3 else
            f"Hi {name}, your refund of {amount} for order {order} is approved." if level == 2 else f"Your refund of {amount} for order {order} is approved."}


def summarize_facts(rng: random.Random, level: int) -> dict[str, Any]:
    service = rng.choice(["checkout", "search", "payments", "inventory"])
    rate, start = f"{rng.randint(3, 19)}.{rng.randint(0, 9)}%", f"{rng.randint(8, 22)}:{rng.choice(['05', '15', '40', '55'])}"
    owner = rng.choice(["Dana", "Rui", "Mei", "Karim"])
    cause_text, cause = rng.choice([("an expired certificate", "certificate"), ("a bad config push", "config"), ("a full disk", "disk"), ("a connection pool leak", "pool")])
    fix_text, fix = rng.choice([("a rollback", "rollback"), ("a restart", "restart"), ("a manual revert", "revert"), ("a scale-up", "scale")])
    customers = rng.randint(1200, 9800)
    ticket, minutes = f"INC-{rng.randint(1000, 9999)}", rng.randint(35, 240)
    facts = [("service", service), ("peak error rate", rate), ("start time", start), ("owner", owner), ("root cause", cause), ("fix", fix),
             ("customers affected", str(customers)), ("ticket", ticket), ("duration in minutes", str(minutes))]
    n = {1: 3, 2: 5, 3: 7, 4: 9}[level]
    chosen = facts[:n]
    noise = ["Lunch order was placed for the team.", "The status page theme was updated last week.", "A new hire joins on Monday.",
             "Printer on level 2 is out of toner.", "The weekly sync was moved to Thursday.", "Parking rules changed for visitors."]
    lines = [f"{start} - alerts fired for {service}; peak error rate reached {rate}.", f"{owner} was paged and took ownership of the incident.",
             f"Investigation found {cause_text}.", f"Resolved with {fix_text}; about {customers:,} customers were affected."]
    if level >= 4:
        lines += [f"Tracked as {ticket}.", f"The incident lasted {minutes} minutes in total."]
    lines += rng.sample(noise, {1: 1, 2: 3, 3: 5, 4: 6}[level])
    rng.shuffle(lines)
    limit = 45 if level >= 4 else 30 + 10 * level
    prompt = ("Incident log:\n" + "\n".join(f"- {line}" for line in lines) + "\n\nWrite a summary of the incident in at most "
              f"{limit} words. It must state: " + ", ".join(k for k, _ in chosen) + ". Reply with only the summary.")
    include = [v for _, v in chosen]
    gold = "; ".join(f"{k}: {v}" for k, v in chosen)
    return {"prompt": prompt, "verifier": {"type": "checks", "must_include": include, "max_words": limit}, "gold": gold}


GENERATORS: dict[str, tuple[str, Callable[[random.Random, int], dict[str, Any]]]] = {
    "extract_fields": ("extraction", extract_fields),
    "reformat_dates": ("extraction", reformat_dates),
    "classify_intent": ("extraction", classify_intent),
    "arithmetic_steps": ("reasoning", arithmetic_steps),
    "ordering_puzzle": ("reasoning", ordering_puzzle),
    "schedule_csp": ("reasoning", schedule_csp),
    "policy_amount": ("reasoning", policy_amount),
    "code_trace": ("coding", code_trace),
    "code_bug_line": ("coding", code_bug_line),
    "function_call": ("tools", function_call),
    "constrained_reply": ("writing", constrained_reply),
    "summarize_facts": ("writing", summarize_facts),
}

# Families whose difficulty scale runs to 5; every other family stops at 4.
MAX_LEVEL: dict[str, int] = {"arithmetic_steps": 5, "ordering_puzzle": 5, "code_trace": 5}


def _register_more() -> None:
    from capability_lab.generators2 import MORE

    for family, (skill, generate, top) in MORE.items():
        GENERATORS[family] = (skill, generate)
        MAX_LEVEL[family] = top


_register_more()

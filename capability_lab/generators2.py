"""More task families for the capability benchmark, chosen because their difficulty is computational or logical.

Each family has a gold answer computed by trusted code (Python's datetime, re, sqlite3, brute force), so the checkers
never rely on a model's opinion. `level` runs 1 to 5.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import sqlite3
from datetime import date, timedelta
from typing import Any

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _spoken(d: date) -> str:
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


# ------------------------------------------------------------------------------------------ extraction
def json_transform(rng: random.Random, level: int) -> dict[str, Any]:
    rows = [{"id": f"R{i + 1}", "qty": rng.randint(1, 20), "price": rng.choice([2.5, 4, 7.5, 10, 12.25, 19.99]),
             "tag": rng.choice(["a", "b", "c"])} for i in range({1: 4, 2: 6, 3: 9, 4: 12, 5: 16}[level])]
    steps = ["keep only records whose qty is greater than 5"]
    out = [r for r in rows if r["qty"] > 5]
    if level >= 2:
        steps.append("add a field total equal to qty times price, rounded to 2 decimals")
        out = [{**r, "total": round(r["qty"] * r["price"], 2)} for r in out]
    else:
        out = [{**r, "total": round(r["qty"] * r["price"], 2)} for r in out]
        steps.append("add a field total equal to qty times price, rounded to 2 decimals")
    if level >= 3:
        steps.append('drop every record whose tag is "c"')
        out = [r for r in out if r["tag"] != "c"]
    if level >= 4:
        steps.append("sort by total from largest to smallest, breaking ties by id in ascending order")
        out.sort(key=lambda r: (-r["total"], r["id"]))
    if level >= 5:
        steps.append("keep only the 4 records with the largest total after sorting")
        out = out[:4]
    result = [{"id": r["id"], "total": r["total"]} for r in out]
    steps.append("return a JSON list of objects with only the fields id and total, in the resulting order")
    prompt = ("Apply these steps to the records, in order:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
              + f"\n\nRecords:\n{json.dumps(rows)}\n\nReply with only the JSON.")
    return {"prompt": prompt, "verifier": {"type": "json", "value": result}, "gold": json.dumps(result)}


# ------------------------------------------------------------------------------------------ reasoning
def calendar_math(rng: random.Random, level: int) -> dict[str, Any]:
    base = date(2024, 1, 1) + timedelta(days=rng.randint(0, 900))
    if level == 1:
        q, gold = f"Which day of the week was {_spoken(base)}?", WEEKDAYS[base.weekday()]
        spec: dict[str, Any] = {"type": "exact", "value": gold}
    elif level == 2:
        n = rng.randint(20, 400)
        q, gold = f"Which day of the week is it {n} days after {_spoken(base)}?", WEEKDAYS[(base + timedelta(days=n)).weekday()]
        spec = {"type": "exact", "value": gold}
    elif level == 3:
        other = base + timedelta(days=rng.randint(60, 700))
        q, gold = f"How many days are there from {_spoken(base)} to {_spoken(other)}, not counting the start day?", str((other - base).days)
        spec = {"type": "integer", "value": int(gold)}
    elif level == 4:
        other = base + timedelta(days=rng.randint(30, 120))
        holidays = sorted({base + timedelta(days=rng.randint(1, (other - base).days)) for _ in range(3)})
        days = [base + timedelta(days=i) for i in range(1, (other - base).days + 1)]
        count = sum(d.weekday() < 5 and d not in holidays for d in days)
        q = (f"Count the working days (Monday to Friday) after {_spoken(base)} up to and including {_spoken(other)}, "
             f"excluding these holidays: {', '.join(_spoken(h) for h in holidays)}.")
        gold, spec = str(count), {"type": "integer", "value": count}
    else:
        n, which = rng.randint(30, 200), rng.choice([(1, "first"), (2, "second"), (3, "third")])
        shifted = base + timedelta(days=n)
        target = rng.choice(range(7))
        first = date(shifted.year, shifted.month, 1)
        first += timedelta(days=(target - first.weekday()) % 7)
        answer = first + timedelta(weeks=which[0] - 1)
        q = (f"Take {_spoken(base)} and add {n} days. In the month you land in, what is the date of the {which[1]} "
             f"{WEEKDAYS[target]}? Answer as YYYY-MM-DD.")
        gold, spec = answer.isoformat(), {"type": "exact", "value": answer.isoformat()}
    return {"prompt": q + " Reply with only the answer.", "verifier": spec, "gold": gold}


def sql_result(rng: random.Random, level: int) -> dict[str, Any]:
    depts = ["Ops", "Dev", "Sales", "Legal"]
    emps = [(f"E{i + 1}", rng.choice(depts), rng.randint(40, 120) * 1000, 2015 + rng.randint(0, 9)) for i in range({1: 6, 2: 8, 3: 10, 4: 12, 5: 14}[level])]
    bonuses = [(f"E{i + 1}", rng.randint(1, 9) * 500) for i in range(len(emps)) if rng.random() < 0.7]
    queries = {
        1: "SELECT name FROM emp WHERE salary > 70000 ORDER BY name",
        2: "SELECT name, salary FROM emp ORDER BY salary DESC, name LIMIT 3",
        3: "SELECT dept, COUNT(*) AS n, MAX(salary) AS top FROM emp GROUP BY dept ORDER BY dept",
        4: "SELECT e.dept, SUM(b.amount) AS total FROM emp e JOIN bonus b ON b.name = e.name GROUP BY e.dept HAVING SUM(b.amount) > 2000 ORDER BY e.dept",
        5: "SELECT name FROM emp e WHERE salary > (SELECT AVG(salary) FROM emp WHERE dept = e.dept) AND hired < 2022 ORDER BY name",
    }
    sql = queries[level]
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE emp (name TEXT, dept TEXT, salary INTEGER, hired INTEGER)")
    conn.execute("CREATE TABLE bonus (name TEXT, amount INTEGER)")
    conn.executemany("INSERT INTO emp VALUES (?,?,?,?)", emps)
    conn.executemany("INSERT INTO bonus VALUES (?,?)", bonuses)
    result = [list(r) for r in conn.execute(sql).fetchall()]
    conn.close()
    prompt = ("Tables:\nemp(name, dept, salary, hired)\n" + "\n".join(map(str, emps)) + "\n\nbonus(name, amount)\n"
              + "\n".join(map(str, bonuses)) + f"\n\nWhat rows does this SQL query return, in order?\n{sql}\n"
              "Reply with only a JSON list of rows, each row a list of values.")
    return {"prompt": prompt, "verifier": {"type": "json", "value": result}, "gold": json.dumps(result)}


def knights_knaves(rng: random.Random, level: int) -> dict[str, Any]:
    names = ["Ana", "Ben", "Cleo", "Dev", "Eli", "Fay"][:{1: 3, 2: 4, 3: 5, 4: 5, 5: 6}[level]]
    n = len(names)

    def statement() -> tuple:
        kind = rng.choice(["knight", "knave", "same", "different", "both", "either"])
        a, b = rng.sample(range(n), 2)
        return kind, a, b

    def says_true(stmt: tuple, truth: tuple[bool, ...]) -> bool:
        kind, a, b = stmt
        return {"knight": truth[a], "knave": not truth[a], "same": truth[a] == truth[b], "different": truth[a] != truth[b],
                "both": truth[a] and truth[b], "either": truth[a] or truth[b]}[kind]

    def consistent(speaker: int, stmt: tuple, truth: tuple[bool, ...]) -> bool:
        return says_true(stmt, truth) == truth[speaker]  # knights say true statements, knaves say false ones

    for _ in range(2000):
        stmts = [(i, statement()) for i in range(n)]
        solutions = [t for t in itertools.product([True, False], repeat=n) if all(consistent(i, s, t) for i, s in stmts)]
        if len(solutions) == 1:
            truth = solutions[0]
            break
    else:
        raise RuntimeError("no uniquely solvable puzzle found")
    text = {"knight": "{a} is a knight.", "knave": "{a} is a knave.", "same": "{a} and {b} are the same kind.",
            "different": "{a} and {b} are different kinds.", "both": "{a} and {b} are both knights.",
            "either": "At least one of {a} and {b} is a knight."}
    lines = [f'{names[i]} says: "{text[k].format(a=names[a], b=names[b])}"' for i, (k, a, b) in stmts]
    prompt = ("On an island, knights always tell the truth and knaves always lie. " + " ".join(lines)
              + " Who is a knight and who is a knave? Reply with only a JSON object mapping each name to \"knight\" or \"knave\".")
    gold = {names[i]: ("knight" if truth[i] else "knave") for i in range(n)}
    return {"prompt": prompt, "verifier": {"type": "json", "value": gold}, "gold": json.dumps(gold)}


UNITS = "Conversion table: 1 mile = 1.609 km; 1 hour = 60 minutes; 1 km = 1000 m; 1 gallon = 3.785 L; 1 kg = 2.205 lb."


def unit_chain(rng: random.Random, level: int) -> dict[str, Any]:
    speed, hours, minutes = rng.choice([60, 72, 90, 108]), rng.randint(1, 3), rng.choice([15, 30, 45])
    km = speed * (hours + minutes / 60)
    steps = [f"A car travels at {speed} km/h for {hours} hours and {minutes} minutes."]
    value = km
    if level >= 2:
        steps.append("Convert the distance to miles.")
        value = value / 1.609
    if level >= 3:
        mpg = rng.choice([25, 32, 40])
        steps.append(f"The car does {mpg} miles per gallon. How many gallons does it use?")
        value = value / mpg
    if level >= 4:
        price = rng.choice([3.1, 3.6, 4.2])
        steps.append(f"Fuel costs ${price} per gallon, and a 8% fee is then added to the fuel cost. What is the total cost in dollars?")
        value = value * price * 1.08
    if level >= 5:
        people = rng.choice([3, 4, 6])
        steps.append(f"The cost is split equally between {people} people and each person also pays $2.50 for snacks. What does each person pay in dollars?")
        value = value / people + 2.5
    if level == 1:
        steps.append("How far did it travel in metres?")
        value = km * 1000
    prompt = UNITS + " " + " ".join(steps) + " Reply with only the final number, rounded to 2 decimals."
    return {"prompt": prompt, "verifier": {"type": "number", "value": round(value, 2), "tol": max(0.05, abs(value) * 0.0005)},
            "gold": f"{value:.2f}"}


# -------------------------------------------------------------------------------------------- coding
def regex_predict(rng: random.Random, level: int) -> dict[str, Any]:
    patterns = {
        1: ([r"[a-z]+\d", r"\d{3}-\d{4}", r"colou?r"], ["abc1", "ab", "555-1234", "55-1234", "color", "colour", "colr", "xyz9"]),
        2: ([r"(ab|cd)+e", r"[A-Z][a-z]{2,4}", r"\w+@\w+\.com"], ["abcde", "cde", "ababe", "Tom", "Thomas", "tom", "me@site.com", "me@site.org"]),
        3: ([r"^(\d+)-\1$", r"(?:a|b)*c$", r"\b\w{3}\b"], ["12-12", "12-13", "abbac", "abba", "cat", "cats", "x y z", "dog"]),
        4: ([r"(?=.*\d)(?=.*[a-z]).{5,}", r"^(?!.*bad).*$", r"(\w)\1"], ["abc12", "abcde", "12345", "a1", "good", "bad day", "hello", "ab"]),
        5: ([r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$", r"^(?=.{6,12}$)(?=.*[A-Z])(?=.*\d)\S+$", r"^([a-z]+)\s\1$"],
            ["10.0.0.1", "256.1.1.1", "192.168.1.300", "Abc123", "abc123", "Ab1", "tom tom", "tom jerry", "1.2.3.4"]),
    }
    pats, pool = patterns[level]
    pattern = rng.choice(pats)
    strings = rng.sample(pool, k=min(len(pool), 6 + level))
    answer = [i + 1 for i, s in enumerate(strings) if re.fullmatch(pattern, s)] if not pattern.startswith("^") and "\\b" not in pattern else \
        [i + 1 for i, s in enumerate(strings) if re.search(pattern, s)]
    mode = "fullmatch" if not pattern.startswith("^") and "\\b" not in pattern else "search"
    prompt = (f"Python regular expression: {pattern}\nUsing re.{mode}(pattern, string), which of these strings match?\n"
              + "\n".join(f"{i + 1}. {s!r}" for i, s in enumerate(strings))
              + "\nReply with only a JSON list of the matching numbers in ascending order ([] if none).")
    return {"prompt": prompt, "verifier": {"type": "json", "value": answer}, "gold": json.dumps(answer)}


def bit_ops(rng: random.Random, level: int) -> dict[str, Any]:
    a, b, c = rng.randint(5, 250), rng.randint(5, 250), rng.randint(1, 7)
    exprs = {
        1: f"{a} & {b}",
        2: f"({a} | {b}) ^ {a}",
        3: f"(({a} << {c}) ^ {b}) & 255",
        4: f"((({a} ^ {b}) << {c}) | ({b} >> {c})) & 1023",
        5: f"(((({a} << {c}) ^ {b}) & 511) + (({b} | {a}) >> {c})) ^ ({a} & {b})",
    }
    expr = exprs[level]
    value = eval(expr, {"__builtins__": {}})  # noqa: S307 - arithmetic on integers we generated
    prompt = f"Evaluate this Python expression, where & is bitwise AND, | is OR, ^ is XOR, << and >> are shifts:\n{expr}\nReply with only the integer result."
    return {"prompt": prompt, "verifier": {"type": "integer", "value": value}, "gold": str(value)}


# --------------------------------------------------------------------------------------------- tools
TOOLS = [("get_order", "Fetch one order by its id."), ("list_orders", "List all orders for a customer."), ("track_shipment", "Get carrier tracking events for a shipped order."),
         ("issue_refund", "Refund money for an order."), ("add_store_credit", "Add store credit to a customer account."), ("cancel_order", "Cancel an order that has not shipped."),
         ("update_address", "Change the delivery address of an order."), ("search_help_center", "Search policy articles."), ("get_customer", "Look up a customer profile."),
         ("escalate_to_human", "Hand the case to a human specialist.")]
REQUESTS = {
    "get_order": ["What is in order A-1042?", "Show me the details of order A-2231."],
    "list_orders": ["Show everything priya.nair@example.com has ever bought.", "List every order for customer C-203."],
    "track_shipment": ["Where is parcel for order A-1057 right now?", "Has order A-3301 left the depot yet?"],
    "issue_refund": ["Give the customer back $40 for order A-5521.", "Money back please, $129 on A-1042."],
    "add_store_credit": ["Add $25 to customer C-201's account as a goodwill gesture.", "Credit C-203 with $60 for the delay."],
    "cancel_order": ["Stop order A-1088, it has not shipped.", "Please cancel A-4410 before it leaves."],
    "update_address": ["Send order A-1088 to 14 Harbour Lane instead.", "Change where A-7741 is delivered to 9 Mill Road."],
    "search_help_center": ["What is our policy on returns after 30 days?", "Find the article about warranty claims."],
    "get_customer": ["What membership tier is elena.petrova@example.com?", "Look up who C-202 is."],
    "escalate_to_human": ["This warranty claim needs a person to look at it.", "Pass this angry customer to a specialist."],
}


def tool_selection(rng: random.Random, level: int) -> dict[str, Any]:
    target = rng.choice(list(REQUESTS))
    request = rng.choice(REQUESTS[target])
    count = {1: 4, 2: 6, 3: 8, 4: 10, 5: 10}[level]
    others = [t for t in TOOLS if t[0] != target]
    rng.shuffle(others)
    offered = [t for t in TOOLS if t[0] == target] + others[:count - 1]
    none_case = level >= 4 and rng.random() < 0.3
    if none_case:
        offered = [t for t in offered if t[0] != target]
        request, target = rng.choice(["What is the weather in Dubai tomorrow?", "Write a poem about refunds.", "Translate this email into French."]), "none"
    rng.shuffle(offered)
    prompt = ("Tools:\n" + "\n".join(f"- {n}: {d}" for n, d in offered) + f"\n\nRequest: {request}\nWhich single tool should be used? "
              "Reply with only the tool name, or none if no tool fits.")
    return {"prompt": prompt, "verifier": {"type": "exact", "value": target}, "gold": target}


# ------------------------------------------------------------------------------------------- writing
def list_formatting(rng: random.Random, level: int) -> dict[str, Any]:
    names = rng.sample(["Ana", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jo", "Kai", "Lea", "Max", "Nia", "Oz", "Pia"], {1: 4, 2: 7, 3: 10, 4: 13, 5: 16}[level])
    rows = [(n, rng.randint(40, 99), rng.choice(["red", "blue"])) for n in names]
    ordered = sorted(rows, key=lambda r: (-r[1], r[0]))
    rule = "Sort by score from highest to lowest, breaking ties alphabetically by name."
    if level >= 4:
        ordered = sorted(rows, key=lambda r: (r[2], -r[1], r[0]))
        rule = "Group by team (blue first, then red). Inside a team sort by score from highest to lowest, ties alphabetically by name."
    lines = [f"{i + 1}. {n} - {s} ({t})" for i, (n, s, t) in enumerate(ordered)]
    if level >= 5:
        lines = [f"{i + 1}. {n} - {s} ({t})" + (" *" if s >= 90 else "") for i, (n, s, t) in enumerate(ordered)]
        rule += " Add ' *' at the end of every line whose score is at least 90."
    prompt = ("Data (name, score, team):\n" + "\n".join(map(str, rows)) + f"\n\n{rule} Number the lines from 1. Format each line exactly as "
              "'N. Name - score (team)'. Reply with only the lines.")
    gold = "\n".join(lines)
    return {"prompt": prompt, "verifier": {"type": "exact", "value": gold}, "gold": gold}


MORE = {
    "json_transform": ("extraction", json_transform, 5),
    "calendar_math": ("reasoning", calendar_math, 5),
    "sql_result": ("coding", sql_result, 5),
    "knights_knaves": ("reasoning", knights_knaves, 5),
    "unit_chain": ("reasoning", unit_chain, 5),
    "regex_predict": ("coding", regex_predict, 5),
    "bit_ops": ("coding", bit_ops, 5),
    "tool_selection": ("tools", tool_selection, 5),
    "list_formatting": ("writing", list_formatting, 5),
}

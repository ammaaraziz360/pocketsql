from __future__ import annotations

import random

from .query_ast import QueryPlan


VERBS = ("show", "list", "find", "which")


def verbalize(plan: QueryPlan, rng: random.Random) -> str:
    verb = rng.choice(VERBS)
    if plan.group_by:
        words = f"{verb} the count of {plan.table} grouped by " + " and ".join(plan.group_by)
        if plan.filters:
            words += " where " + f" {plan.filter_connector.lower()} ".join(f"{item.column} {item.operator} {item.value}" for item in plan.filters)
        return words
    if plan.aggregate == "COUNT":
        words = f"how many {plan.table} are there"
        if plan.filters:
            words += " where " + f" {plan.filter_connector.lower()} ".join(f"{item.column} {item.operator} {item.value}" for item in plan.filters)
        return words
    if plan.aggregate:
        adjective = {"SUM": "total", "AVG": "average", "MIN": "smallest", "MAX": "largest"}[plan.aggregate]
        target = plan.aggregate_column or "value"
        words = f"{verb} the {adjective} {target} for {plan.table}"
        if plan.filters:
            words += " where " + f" {plan.filter_connector.lower()} ".join(f"{item.column} {item.operator} {item.value}" for item in plan.filters)
        return words
    words = f"{verb} " + " and ".join(plan.columns) + f" from {plan.table}"
    if plan.distinct:
        words = f"{verb} distinct " + " and ".join(plan.columns) + f" from {plan.table}"
    if plan.filters:
        terms = [f"{item.column} {item.operator} {item.value}" for item in plan.filters]
        words += " where " + f" {plan.filter_connector.lower()} ".join(terms)
    if plan.join_table:
        words += f" with matching {plan.join_table}"
    if plan.limit:
        words += f", limited to {plan.limit}"
    return words
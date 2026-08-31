"""Compact, validated semantic targets for deterministic SQL rendering."""
from __future__ import annotations

import math
import re

from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import literal, render_sql


IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
REFERENCE = rf"{IDENTIFIER}(?:\.(?:{IDENTIFIER}|\*))?"
REFERENCE_RE = re.compile(rf"^{REFERENCE}$")
FILTER_RE = re.compile(rf"^({REFERENCE})\s*(>=|<=|=|>|<)\s*(.+)$")
NUMBER_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
VALUE_SLOT_RE = re.compile(r"^value\d+$")
AGGREGATES = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
CONNECTORS = {"AND", "OR"}


class SemanticPlanError(ValueError):
    """Decoded text is not one complete, safe semantic plan."""


def query_plan_from_dict(data: dict) -> QueryPlan:
    """Rehydrate the normalized ``QueryPlan`` dictionaries stored in JSONL."""
    return QueryPlan(
        family=data.get("family", "semantic_plan"),
        table=data["table"],
        columns=tuple(data.get("columns", ())),
        aggregate=data.get("aggregate"),
        aggregate_column=data.get("aggregate_column"),
        distinct=bool(data.get("distinct", False)),
        filters=tuple(Filter(item["column"], item["operator"], item["value"]) for item in data.get("filters", ())),
        filter_connector=data.get("filter_connector", "AND"),
        group_by=tuple(data.get("group_by", ())),
        order_by=data.get("order_by"),
        descending=bool(data.get("descending", False)),
        limit=data.get("limit"),
        join_table=data.get("join_table"),
        join_on=tuple(data["join_on"]) if data.get("join_on") else None,
        aggregate_position=int(data.get("aggregate_position", 0)),
    )


def _reference(value: str) -> str:
    if not isinstance(value, str) or not REFERENCE_RE.fullmatch(value):
        raise SemanticPlanError(f"invalid identifier reference: {value!r}")
    return value


def _selection_reference(value: str) -> str:
    return value if value == "*" else _reference(value)


def _literal_token(value: str | int | float) -> str:
    if isinstance(value, bool):
        raise SemanticPlanError("Boolean filter values are unsupported")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise SemanticPlanError("Non-finite filter values are unsupported")
        return str(value)
    if VALUE_SLOT_RE.fullmatch(value):
        return value
    return literal(value)


def serialize_semantic_plan(plan: QueryPlan | dict) -> str:
    """Serialize a plan in a fixed-order DSL that contains no SQL clauses."""
    if isinstance(plan, dict):
        plan = query_plan_from_dict(plan)
    clauses = [f"T {_reference(plan.table)}"]
    if plan.columns:
        clauses.append("S " + ",".join(_selection_reference(item) for item in plan.columns))
    if plan.aggregate:
        aggregate = plan.aggregate.upper()
        if aggregate not in AGGREGATES:
            raise SemanticPlanError(f"unsupported aggregate: {plan.aggregate!r}")
        target = "*" if aggregate == "COUNT" and not plan.aggregate_column else _reference(plan.aggregate_column or "")
        clauses.append(f"A {aggregate} {target} {plan.aggregate_position}")
    if not plan.columns and not plan.aggregate:
        raise SemanticPlanError("plan must select columns or an aggregate")
    if plan.distinct:
        clauses.append("D")
    if plan.join_table or plan.join_on:
        if not plan.join_table or not plan.join_on or len(plan.join_on) != 2:
            raise SemanticPlanError("join table and two join keys must be supplied together")
        clauses.append(
            f"J {_reference(plan.join_table)} {_reference(plan.join_on[0])} {_reference(plan.join_on[1])}"
        )
    if plan.filters:
        connector = plan.filter_connector.upper()
        if connector not in CONNECTORS:
            raise SemanticPlanError(f"unsupported filter connector: {plan.filter_connector!r}")
        filters = " & ".join(
            f"{_reference(item.column)} {item.operator} {_literal_token(item.value)}" for item in plan.filters
        )
        clauses.append(f"F {connector} {filters}")
    if plan.group_by:
        clauses.append("G " + ",".join(_reference(item) for item in plan.group_by))
    if plan.order_by:
        clauses.append(f"O {_reference(plan.order_by)} {'DESC' if plan.descending else 'ASC'}")
    if plan.limit is not None:
        if isinstance(plan.limit, bool) or not isinstance(plan.limit, int) or plan.limit < 1:
            raise SemanticPlanError(f"invalid limit: {plan.limit!r}")
        clauses.append(f"L {plan.limit}")
    return " | ".join(clauses)


def _split_unquoted(text: str, delimiter: str) -> list[str]:
    """Split on a one-character delimiter while honoring SQL string quoting."""
    parts: list[str] = []
    start = 0
    quoted = False
    index = 0
    while index < len(text):
        character = text[index]
        if character == "'":
            if quoted and index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif character == delimiter and not quoted:
            parts.append(text[start:index].strip())
            start = index + 1
        index += 1
    if quoted:
        raise SemanticPlanError("unterminated quoted literal")
    parts.append(text[start:].strip())
    return parts


def _parse_literal(token: str) -> str | int | float:
    token = token.strip()
    if VALUE_SLOT_RE.fullmatch(token):
        return token
    if token.startswith("'") and token.endswith("'"):
        inner = token[1:-1]
        if "'" in inner.replace("''", ""):
            raise SemanticPlanError(f"invalid quoted literal: {token!r}")
        return inner.replace("''", "'")
    if NUMBER_RE.fullmatch(token):
        return float(token) if "." in token else int(token)
    raise SemanticPlanError(f"invalid filter literal: {token!r}")


def parse_semantic_plan(text: str) -> QueryPlan:
    """Parse the compact DSL into a validated ``QueryPlan``."""
    text = text.strip()
    if not text:
        raise SemanticPlanError("empty semantic plan")
    values: dict[str, object] = {}
    seen: set[str] = set()
    for clause in _split_unquoted(text, "|"):
        if not clause:
            raise SemanticPlanError("empty semantic-plan clause")
        key, _, body = clause.partition(" ")
        key = key.upper()
        body = body.strip()
        if key in seen:
            raise SemanticPlanError(f"duplicate semantic-plan clause: {key}")
        seen.add(key)
        if key == "T":
            values["table"] = _reference(body)
        elif key == "S":
            columns = tuple(_selection_reference(item.strip()) for item in _split_unquoted(body, ",") if item.strip())
            if not columns:
                raise SemanticPlanError("empty selection")
            values["columns"] = columns
        elif key == "A":
            pieces = body.split()
            if len(pieces) != 3 or pieces[0].upper() not in AGGREGATES or not pieces[2].isdigit():
                raise SemanticPlanError(f"invalid aggregate clause: {body!r}")
            aggregate, target, position = pieces[0].upper(), pieces[1], int(pieces[2])
            if target == "*":
                if aggregate != "COUNT":
                    raise SemanticPlanError("only COUNT may target *")
                aggregate_column = None
            else:
                aggregate_column = _reference(target)
            values.update(aggregate=aggregate, aggregate_column=aggregate_column, aggregate_position=position)
        elif key == "D":
            if body:
                raise SemanticPlanError("D does not take a value")
            values["distinct"] = True
        elif key == "J":
            pieces = body.split()
            if len(pieces) != 3:
                raise SemanticPlanError(f"invalid join clause: {body!r}")
            values.update(join_table=_reference(pieces[0]), join_on=(_reference(pieces[1]), _reference(pieces[2])))
        elif key == "F":
            connector, separator, filter_body = body.partition(" ")
            connector = connector.upper()
            if not separator or connector not in CONNECTORS:
                raise SemanticPlanError(f"invalid filter clause: {body!r}")
            filters = []
            for item in _split_unquoted(filter_body, "&"):
                match = FILTER_RE.fullmatch(item.strip())
                if not match:
                    raise SemanticPlanError(f"invalid filter predicate: {item!r}")
                column, operator, raw_value = match.groups()
                filters.append(Filter(_reference(column), operator, _parse_literal(raw_value)))
            if not filters:
                raise SemanticPlanError("empty filter clause")
            values.update(filters=tuple(filters), filter_connector=connector)
        elif key == "G":
            groups = tuple(_reference(item.strip()) for item in _split_unquoted(body, ",") if item.strip())
            if not groups:
                raise SemanticPlanError("empty group clause")
            values["group_by"] = groups
        elif key == "O":
            pieces = body.split()
            if len(pieces) != 2 or pieces[1].upper() not in {"ASC", "DESC"}:
                raise SemanticPlanError(f"invalid order clause: {body!r}")
            values.update(order_by=_reference(pieces[0]), descending=pieces[1].upper() == "DESC")
        elif key == "L":
            if not body.isdigit() or int(body) < 1:
                raise SemanticPlanError(f"invalid limit clause: {body!r}")
            values["limit"] = int(body)
        else:
            raise SemanticPlanError(f"unknown semantic-plan clause: {key!r}")

    if "table" not in values:
        raise SemanticPlanError("semantic plan is missing T")
    if "columns" not in values and "aggregate" not in values:
        raise SemanticPlanError("semantic plan is missing S or A")
    columns = values.get("columns", ())
    if values.get("aggregate_position", 0) > len(columns):
        raise SemanticPlanError("aggregate position exceeds selection width")
    values.setdefault("columns", ())
    return QueryPlan("semantic_plan", **values)


def semantic_plan_to_sql(text: str) -> str:
    """Parse a complete plan and render its single read-only SELECT."""
    return render_sql(parse_semantic_plan(text))

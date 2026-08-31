from __future__ import annotations

from dataclasses import replace
import json
import random

from .query_ast import Filter, QueryPlan
from .schemas import CITY_VALUES, STATUS_VALUES, Schema


V9_PROFILE = "composition-v9"


def composition_signature(plan: QueryPlan) -> str:
    """Describe a plan by independently selected operations, not a template name."""
    parts = ["join" if plan.join_table else "single"]
    if plan.group_by:
        parts.append("group_count")
    elif plan.distinct:
        parts.append("distinct")
    elif plan.aggregate:
        parts.append(plan.aggregate.casefold())
    else:
        parts.append("select")
    if plan.filters:
        parts.append(f"filter{len(plan.filters)}_{plan.filter_connector.casefold()}")
    if plan.order_by:
        parts.append("order")
    if plan.limit is not None:
        parts.append("limit")
    return "compose_" + "_".join(parts)


def composition_tier(plan: QueryPlan) -> str:
    """Bucket plans by the number of capabilities that must be combined."""
    operations = 1
    operations += int(plan.join_table is not None)
    operations += int(bool(plan.filters))
    operations += int(len(plan.filters) > 1)
    operations += int(bool(plan.group_by))
    operations += int(plan.order_by is not None)
    operations += int(plan.limit is not None)
    if operations == 1:
        return "atomic"
    if operations == 2:
        return "pair"
    return "multi"


def held_out_composition(plan: QueryPlan) -> str | None:
    """Return the evaluation-only combination represented by a plan, if any."""
    if plan.join_table and plan.aggregate == "COUNT" and len(plan.filters) >= 2 and plan.filter_connector == "AND":
        return "join_count_multi_filter"
    if plan.join_table and plan.aggregate in {"SUM", "AVG", "MIN", "MAX"} and len(plan.filters) >= 2 and plan.filter_connector == "AND":
        return "join_aggregate_multi_filter"
    if plan.distinct and plan.filters and plan.limit is not None:
        return "distinct_filter_limit"
    if plan.group_by and plan.filters:
        return "group_filter"
    return None


def _filters_for(
    schema: Schema,
    table: str,
    joined: bool,
    count: int,
    connector: str,
    variant: int,
) -> tuple[Filter, ...]:
    if count == 0:
        return ()
    parent, child = schema.tables
    location = schema.role("location")[1].name
    amount = schema.role("amount")[1].name
    status = schema.role("status")[1].name
    city = CITY_VALUES[variant % len(CITY_VALUES)]
    if joined:
        if connector == "OR":
            return tuple(
                Filter(f"{child.name}.{status}", "=", STATUS_VALUES[(variant + offset) % len(STATUS_VALUES)])
                for offset in range(count)
            )
        candidates = (
            Filter(f"{parent.name}.{location}", "=", city),
            Filter(f"{child.name}.{status}", "=", STATUS_VALUES[variant % len(STATUS_VALUES)]),
            # Population guarantees every location/status pair has an amount
            # above 25, so a three-way conjunction remains non-empty.
            Filter(f"{child.name}.{amount}", ">", 25),
        )
        if count == 1:
            return (candidates[variant % len(candidates)],)
        if count == 2:
            pairs = ((candidates[0], candidates[1]), (candidates[0], candidates[2]), (candidates[1], candidates[2]))
            return pairs[variant % len(pairs)]
        return candidates

    if table == parent.name:
        # Multiple location values are meaningful with OR, but contradictory
        # with AND. The sampler prevents the latter combination.
        return tuple(
            Filter(location, "=", CITY_VALUES[(variant + offset) % len(CITY_VALUES)])
            for offset in range(count)
        )
    if connector == "OR":
        return tuple(
            Filter(status, "=", STATUS_VALUES[(variant + offset) % len(STATUS_VALUES)])
            for offset in range(count)
        )
    operator, comparison = (
        (">", 100),
        ("<", 250),
        (">=", 150),
        ("<=", 250),
    )[variant % 4]
    numeric = Filter(amount, operator, comparison)
    if count == 1:
        return (Filter(status, "=", STATUS_VALUES[variant % len(STATUS_VALUES)]),) if variant % 3 == 0 else (numeric,)
    complementary = Filter(amount, "<=" if operator in {">", ">="} else ">=", 440 if operator in {">", ">="} else 25)
    candidates = (
        Filter(status, "=", STATUS_VALUES[variant % len(STATUS_VALUES)]),
        numeric,
        complementary,
    )
    return candidates[:count]


def _sample_plan(schema: Schema, rng: random.Random, variant: int) -> QueryPlan:
    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    child_fk = schema.role("parent_fk")[1].name
    amount = schema.role("amount")[1].name
    status = schema.role("status")[1].name
    joined = rng.random() < 0.34

    if joined:
        table = child.name
        join_table = parent.name
        join_on = (f"{child.name}.{child_fk}", f"{parent.name}.{parent_id}")
        output_mode = rng.choices(
            ("select", "count", "sum", "avg", "min", "max"),
            weights=(38, 18, 11, 11, 11, 11),
            k=1,
        )[0]
    else:
        output_mode = rng.choices(
            ("select", "distinct", "count", "sum", "avg", "min", "max", "group"),
            weights=(32, 10, 14, 9, 9, 9, 9, 8),
            k=1,
        )[0]
        table = child.name if output_mode in {"sum", "avg", "min", "max", "group"} or rng.random() < 0.55 else parent.name
        join_table = None
        join_on = None

    aggregate = None
    aggregate_column = None
    distinct = output_mode == "distinct"
    group_by: tuple[str, ...] = ()
    if output_mode == "count":
        aggregate = "COUNT"
        columns: tuple[str, ...] = ()
    elif output_mode in {"sum", "avg", "min", "max"}:
        aggregate = output_mode.upper()
        aggregate_column = f"{child.name}.{amount}" if joined else amount
        columns = ()
    elif output_mode == "group":
        aggregate = "COUNT"
        columns = (status,)
        group_by = (status,)
    elif joined:
        if rng.random() < 0.3:
            columns = (f"{child.name}.*",)
        else:
            parent_column = rng.choice((schema.role("name")[1].name, schema.role("location")[1].name))
            child_column = rng.choice((schema.role("child_id")[1].name, amount, status))
            columns = (f"{parent.name}.{parent_column}", f"{child.name}.{child_column}")
    else:
        available = [column.name for column in schema.table(table).columns]
        width = 1 if distinct else rng.randint(1, min(3, len(available)))
        start = variant % len(available)
        columns = tuple(available[(start + offset) % len(available)] for offset in range(width))

    filter_count = rng.choices((0, 1, 2, 3), weights=(25, 36, 29, 10), k=1)[0]
    connector = "OR" if filter_count > 1 and rng.random() < 0.42 else "AND"
    if not joined and table == parent.name and filter_count > 1:
        connector = "OR"
    filters = _filters_for(schema, table, joined, filter_count, connector, variant)

    order_by = None
    descending = False
    limit = None
    if aggregate is None or group_by:
        if rng.random() < 0.28:
            if joined:
                order_by = f"{child.name}.{amount}"
            elif group_by:
                order_by = group_by[0]
            else:
                order_by = rng.choice(tuple(column.name for column in schema.table(table).columns))
            descending = rng.random() < 0.5
        if rng.random() < (0.68 if order_by else 0.2):
            limit = 2 + (variant % 5)

    plan = QueryPlan(
        "composition",
        table,
        columns,
        aggregate=aggregate,
        aggregate_column=aggregate_column,
        distinct=distinct,
        filters=filters,
        filter_connector=connector,
        group_by=group_by,
        order_by=order_by,
        descending=descending,
        limit=limit,
        join_table=join_table,
        join_on=join_on,
    )
    return replace(plan, family=composition_signature(plan))


def _plan_key(plan: QueryPlan) -> str:
    normalized = plan.normalized()
    normalized.pop("family", None)
    return json.dumps(normalized, sort_keys=True)


def compositional_plans_for(schema: Schema, count: int, rng: random.Random) -> list[QueryPlan]:
    """Generate a balanced operation cross-product while excluding eval combinations."""
    if count < 1:
        return []
    plans: list[QueryPlan] = []
    seen: set[str] = set()

    # Preserve the all-position copy curriculum before sampling combinations.
    for table in schema.tables:
        for column in table.columns:
            if len(plans) >= count:
                return plans
            plan = QueryPlan("compose_single_select", table.name, (column.name,))
            plans.append(plan)
            seen.add(_plan_key(plan))

    atomic_target = min(count, max(len(plans), round(count * 0.25)))
    pair_target = min(round(count * 0.35), count - atomic_target)
    desired = {
        "atomic": atomic_target,
        "pair": pair_target,
    }
    desired["multi"] = count - desired["atomic"] - desired["pair"]
    observed = {"atomic": len(plans), "pair": 0, "multi": 0}
    attempts = 0
    while len(plans) < count and attempts < count * 1000:
        attempts += 1
        candidate = _sample_plan(schema, rng, attempts)
        tier = composition_tier(candidate)
        if observed[tier] >= desired[tier] or held_out_composition(candidate):
            continue
        key = _plan_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        observed[tier] += 1
        plans.append(candidate)
    if len(plans) != count:
        raise RuntimeError(f"Could only generate {len(plans)} of {count} unique v9 plans for {schema.schema_id}")
    rng.shuffle(plans)
    return plans


def held_out_plans_for(schema: Schema) -> list[QueryPlan]:
    """Build combinations whose complete signatures never appear in v9 training."""
    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    child_fk = schema.role("parent_fk")[1].name
    location = schema.role("location")[1].name
    amount = schema.role("amount")[1].name
    status = schema.role("status")[1].name
    name = schema.role("name")[1].name
    join_on = (f"{child.name}.{child_fk}", f"{parent.name}.{parent_id}")
    plans = []
    for variant, city in enumerate(("Boston", "Houston")):
        joined_filters = (
            Filter(f"{parent.name}.{location}", "=", city),
            Filter(f"{child.name}.{status}", "=", STATUS_VALUES[variant]),
        )
        plans.append(
            QueryPlan(
                "composition_holdout_join_count_multi_filter",
                child.name,
                (),
                aggregate="COUNT",
                filters=joined_filters,
                join_table=parent.name,
                join_on=join_on,
            )
        )
        plans.append(
            QueryPlan(
                "composition_holdout_join_aggregate_multi_filter",
                child.name,
                (),
                aggregate=("SUM", "AVG")[variant],
                aggregate_column=f"{child.name}.{amount}",
                filters=(joined_filters[0], Filter(f"{child.name}.{amount}", ">", (50, 150)[variant])),
                join_table=parent.name,
                join_on=join_on,
            )
        )
        plans.append(
            QueryPlan(
                "composition_holdout_distinct_filter_limit",
                parent.name,
                (name,),
                distinct=True,
                filters=(Filter(location, "=", city),),
                limit=variant + 2,
            )
        )
        plans.append(
            QueryPlan(
                "composition_holdout_group_filter",
                child.name,
                (status,),
                aggregate="COUNT",
                filters=(Filter(amount, ">", (50, 150)[variant]),),
                group_by=(status,),
            )
        )
    assert all(held_out_composition(plan) for plan in plans)
    return plans

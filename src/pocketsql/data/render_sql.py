from __future__ import annotations

from .query_ast import QueryPlan


def literal(value: str | int | float) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def render_sql(plan: QueryPlan) -> str:
    select_items = list(plan.columns)
    if plan.aggregate:
        target = "*" if plan.aggregate == "COUNT" and not plan.aggregate_column else plan.aggregate_column
        position = min(max(plan.aggregate_position, 0), len(select_items))
        select_items.insert(position, f"{plan.aggregate}({target})")
    select = ", ".join(select_items)
    prefix = "SELECT DISTINCT" if plan.distinct else "SELECT"
    sql = f"{prefix} {select} FROM {plan.table}"
    if plan.join_table and plan.join_on:
        sql += f" INNER JOIN {plan.join_table} ON {plan.join_on[0]} = {plan.join_on[1]}"
    if plan.filters:
        predicates = [f"{item.column} {item.operator} {literal(item.value)}" for item in plan.filters]
        sql += " WHERE " + f" {plan.filter_connector} ".join(predicates)
    if plan.group_by:
        sql += " GROUP BY " + ", ".join(plan.group_by)
    if plan.order_by:
        sql += f" ORDER BY {plan.order_by}" + (" DESC" if plan.descending else " ASC")
    if plan.limit:
        sql += f" LIMIT {plan.limit}"
    return sql + ";"

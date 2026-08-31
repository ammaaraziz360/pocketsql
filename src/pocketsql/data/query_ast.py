from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class Filter:
    column: str
    operator: Literal["=", ">", "<", ">=", "<="]
    value: str | int | float


@dataclass(frozen=True)
class QueryPlan:
    family: str
    table: str
    columns: tuple[str, ...]
    aggregate: str | None = None
    aggregate_column: str | None = None
    distinct: bool = False
    filters: tuple[Filter, ...] = ()
    filter_connector: Literal["AND", "OR"] = "AND"
    group_by: tuple[str, ...] = ()
    order_by: str | None = None
    descending: bool = False
    limit: int | None = None
    join_table: str | None = None
    join_on: tuple[str, str] | None = None
    aggregate_position: int = 0

    def normalized(self) -> dict:
        return asdict(self)

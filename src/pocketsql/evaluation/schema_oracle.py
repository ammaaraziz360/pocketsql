"""Measure schema-link, operation-structure, and literal-copy error ceilings.

Each isolated oracle replaces only one decision family before rendering and
executing the plan. The schema variants preserve decoded operations and values;
the operation oracle preserves decoded values and uses the factorized link
heads; and the literal oracle changes only filter values.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import render_sql
from pocketsql.evaluation.evaluate import evaluate, score_prediction
from pocketsql.inference import (
    _apply_factorized_schema_links,
    _finish_sql,
    _grounded_inputs,
    generate_sql_batch_with_targets_and_links,
    load_model_from_checkpoint,
)
from pocketsql.model.schema_grounding import canonicalize_record
from pocketsql.model.semantic_plan import (
    SemanticPlanError,
    parse_semantic_plan,
    query_plan_from_dict,
    serialize_semantic_plan,
)
from pocketsql.model.tokenizer import load_tokenizer


_ALL_LINK_COMPONENTS = frozenset({"tables", "projection", "filters", "group_order"})
ORACLE_COMPONENTS = {
    "tables_and_join": frozenset({"tables"}),
    "projection": frozenset({"projection"}),
    "filters": frozenset({"filters"}),
    "group_and_order": frozenset({"group_order"}),
    "all_except_tables_and_join": _ALL_LINK_COMPONENTS - {"tables"},
    "all_except_projection": _ALL_LINK_COMPONENTS - {"projection"},
    "all_except_filters": _ALL_LINK_COMPONENTS - {"filters"},
    "all_except_group_and_order": _ALL_LINK_COMPONENTS - {"group_order"},
    "all_schema_links": _ALL_LINK_COMPONENTS,
}
ORACLE_VARIANTS = tuple(ORACLE_COMPONENTS)
_MISSING_LITERAL_PREFIX = "__oracle_missing_literal_"


def _same_value(left: str | int | float, right: str | int | float) -> bool:
    return type(left) is type(right) and str(left).casefold() == str(right).casefold()


def _replace_reference_sequence(predicted: tuple[str, ...], gold: tuple[str, ...]) -> tuple[str, ...]:
    """Replace identities while preserving the model's predicted arity."""
    if not predicted or not gold:
        return predicted
    return tuple(gold[min(index, len(gold) - 1)] for index in range(len(predicted)))


def _oracle_filters(predicted: QueryPlan, gold: QueryPlan) -> tuple[Filter, ...]:
    if not predicted.filters or not gold.filters:
        return predicted.filters
    replacements = []
    for index, item in enumerate(predicted.filters):
        matching = [candidate.column for candidate in gold.filters if _same_value(candidate.value, item.value)]
        column = matching[0] if len(set(matching)) == 1 else gold.filters[min(index, len(gold.filters) - 1)].column
        replacements.append(replace(item, column=column))
    return tuple(replacements)


def apply_schema_oracle(predicted: QueryPlan, gold: QueryPlan, variant: str) -> QueryPlan:
    """Substitute gold identifier links without correcting decoded operations."""
    if variant not in ORACLE_VARIANTS:
        raise ValueError(f"unknown schema oracle variant: {variant}")
    components = ORACLE_COMPONENTS[variant]
    use_tables = "tables" in components
    use_projection = "projection" in components
    use_filters = "filters" in components
    use_group_order = "group_order" in components

    plan = predicted
    if use_tables:
        plan = replace(plan, table=gold.table, join_table=gold.join_table, join_on=gold.join_on)
    if use_projection:
        projection_candidates = gold.columns
        if not projection_candidates and gold.aggregate_column:
            projection_candidates = (gold.aggregate_column,)
        if not projection_candidates and predicted.columns:
            projection_candidates = (f"{gold.table}.*" if gold.join_table else "*",)
        aggregate_column = predicted.aggregate_column
        if aggregate_column:
            aggregate_column = gold.aggregate_column or (gold.columns[0] if gold.columns else aggregate_column)
        plan = replace(
            plan,
            columns=_replace_reference_sequence(predicted.columns, projection_candidates),
            aggregate_column=aggregate_column,
        )
    if use_filters:
        plan = replace(plan, filters=_oracle_filters(predicted, gold))
    if use_group_order:
        group_candidates = gold.group_by or gold.columns
        group_by = _replace_reference_sequence(predicted.group_by, group_candidates)
        order_by = predicted.order_by
        if order_by:
            order_by = gold.order_by or (gold.group_by[0] if gold.group_by else None) or (
                gold.columns[0] if gold.columns else order_by
            )
        plan = replace(plan, group_by=group_by, order_by=order_by)
    return plan


def apply_operation_oracle(predicted: QueryPlan, gold: QueryPlan) -> QueryPlan:
    """Use the gold query shape while retaining every decoded filter value.

    Schema references in the returned skeleton are placeholders. During the
    executable oracle they are replaced by the model's factorized role heads,
    including roles that were absent from the raw decoded plan. A missing
    decoded filter value remains explicitly missing rather than borrowing the
    gold value, so this oracle does not hide literal-copy failures.
    """
    filters = tuple(
        replace(
            item,
            value=(
                predicted.filters[index].value
                if index < len(predicted.filters)
                else f"{_MISSING_LITERAL_PREFIX}{index}__"
            ),
        )
        for index, item in enumerate(gold.filters)
    )
    return replace(gold, family=predicted.family, filters=filters)


def apply_literal_oracle(predicted: QueryPlan, gold: QueryPlan) -> QueryPlan:
    """Replace only decoded filter values, preserving every other decision."""
    if not predicted.filters or not gold.filters:
        return predicted
    filters = tuple(
        replace(item, value=gold.filters[min(index, len(gold.filters) - 1)].value)
        for index, item in enumerate(predicted.filters)
    )
    return replace(predicted, filters=filters)


def _operation_plan_with_predicted_links(
    predicted: QueryPlan,
    gold: QueryPlan,
    factorized_links: dict | None,
    mapping,
) -> QueryPlan | None:
    if (
        factorized_links is None
        or mapping is None
        or len(predicted.filters) < len(gold.filters)
    ):
        return None
    skeleton = apply_operation_oracle(predicted, gold)
    return _apply_factorized_schema_links(skeleton, factorized_links, mapping)


def _render_canonical_plan(plan: QueryPlan, mapping, schema_sql: str) -> str:
    try:
        restored = mapping.restore(serialize_semantic_plan(plan))
        sql = render_sql(parse_semantic_plan(restored))
    except SemanticPlanError:
        return ""
    return _finish_sql(sql, None, schema_sql)


def _finish_oracle_plan(
    plan: QueryPlan,
    mapping,
    schema_sql: str,
) -> str:
    """Render an isolated plan without letting production heuristics alter it."""
    return _render_canonical_plan(plan, mapping, schema_sql)


def _canonical_gold(
    record: dict,
    model,
    canonicalize_literals: bool | None = None,
) -> QueryPlan:
    canonical = canonicalize_record(
        record,
        getattr(model, "identifier_slot_strategy", "ordered"),
        (
            getattr(model, "canonicalize_literals", False)
            if canonicalize_literals is None
            else canonicalize_literals
        ),
        getattr(model, "schema_linking_hints", False),
    )
    return query_plan_from_dict(canonical["query_plan"])


def _normalized_value(value: str | int | float):
    return type(value).__name__, str(value).casefold() if isinstance(value, str) else value


def _component_matches(predicted: QueryPlan, gold: QueryPlan) -> dict[str, int]:
    predicted_filters = Counter(
        (item.operator, _normalized_value(item.value)) for item in predicted.filters
    )
    gold_filters = Counter((item.operator, _normalized_value(item.value)) for item in gold.filters)
    predicted_values = Counter(_normalized_value(item.value) for item in predicted.filters)
    gold_values = Counter(_normalized_value(item.value) for item in gold.filters)
    scores = {
        "selection_arity": int(len(predicted.columns) == len(gold.columns)),
        "aggregate_function": int(predicted.aggregate == gold.aggregate),
        "aggregate_position": int(predicted.aggregate_position == gold.aggregate_position),
        "distinct": int(predicted.distinct == gold.distinct),
        "filter_count": int(len(predicted.filters) == len(gold.filters)),
        "filter_operators_and_values": int(predicted_filters == gold_filters),
        "filter_values": int(predicted_values == gold_values),
        "filter_connector": int(predicted.filter_connector == gold.filter_connector),
        "group_arity": int(len(predicted.group_by) == len(gold.group_by)),
        "order_presence": int(bool(predicted.order_by) == bool(gold.order_by)),
        "sort_direction": int(predicted.descending == gold.descending),
        "limit": int(predicted.limit == gold.limit),
    }
    scores["all_operations"] = int(all(scores.values()))
    return scores


def _rescue_predictions(
    records: list[dict],
    baseline: list[str],
    corrected: list[str],
    database_base_dir: Path | None,
) -> list[str]:
    """Keep a correction only when it rescues execution for an oracle ceiling."""
    return [
        candidate
        if score_prediction(record, candidate, database_base_dir)["execution_accuracy"]
        else original
        for record, original, candidate in zip(records, baseline, corrected)
    ]


def run_schema_oracle(
    model,
    records: list[dict],
    tokenizer,
    batch_size: int = 16,
    max_tokens: int | None = None,
    database_base_dir: Path | None = None,
    show_progress: bool = False,
) -> tuple[dict, list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    production_predictions: list[str] = []
    raw_targets: list[str] = []
    factorized_predictions: list[dict | None] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        outputs, targets, links = generate_sql_batch_with_targets_and_links(
            model,
            [record["schema_sql"] for record in batch],
            [record["question"] for record in batch],
            tokenizer,
            max_tokens,
        )
        production_predictions.extend(outputs)
        raw_targets.extend(targets)
        factorized_predictions.extend(links)
        if show_progress:
            print(f"generated {len(raw_targets)}/{len(records)} raw plans", file=sys.stderr, flush=True)

    raw_predictions: list[str] = []
    oracle_predictions = {variant: [] for variant in ORACLE_VARIANTS}
    operation_predictions: list[str] = []
    literal_predictions: list[str] = []
    schema_operation_predictions: list[str] = []
    schema_literal_predictions: list[str] = []
    gold_plan_predictions: list[str] = []
    diagnostics: list[dict] = []
    component_totals: Counter = Counter()
    parsed_targets = 0
    factorized_operation_oracle_available = 0
    operation_oracle_eligible = 0

    for record, production_sql, target, factorized_links in zip(
        records,
        production_predictions,
        raw_targets,
        factorized_predictions,
    ):
        _, _, mapping = _grounded_inputs(model, record["schema_sql"], record["question"])
        gold = _canonical_gold(record, model)
        exact_literal_gold = _canonical_gold(record, model, canonicalize_literals=False)
        predicted = None
        try:
            predicted = parse_semantic_plan(target)
        except SemanticPlanError:
            pass

        variants = {}
        operation_components = None
        operation_sql = ""
        literal_sql = ""
        schema_operation_sql = ""
        schema_literal_sql = ""
        gold_plan_sql = (
            _finish_oracle_plan(exact_literal_gold, mapping, record["schema_sql"])
            if mapping is not None
            else ""
        )
        if predicted is None or mapping is None:
            raw_sql = ""
            for variant in ORACLE_VARIANTS:
                variants[variant] = ""
        else:
            parsed_targets += 1
            factorized_operation_oracle_available += int(factorized_links is not None)
            operation_components = _component_matches(predicted, gold)
            component_totals.update(operation_components)
            raw_sql = _render_canonical_plan(predicted, mapping, record["schema_sql"])
            for variant in ORACLE_VARIANTS:
                variants[variant] = _finish_oracle_plan(
                    apply_schema_oracle(predicted, gold, variant),
                    mapping,
                    record["schema_sql"],
                )
            operation_plan = _operation_plan_with_predicted_links(
                predicted,
                exact_literal_gold,
                factorized_links,
                mapping,
            )
            if operation_plan is not None:
                operation_oracle_eligible += 1
                operation_sql = _finish_oracle_plan(
                    operation_plan,
                    mapping,
                    record["schema_sql"],
                )
                schema_operation_sql = _finish_oracle_plan(
                    apply_operation_oracle(predicted, exact_literal_gold),
                    mapping,
                    record["schema_sql"],
                )
            literal_plan = apply_literal_oracle(predicted, exact_literal_gold)
            literal_sql = _finish_oracle_plan(
                literal_plan,
                mapping,
                record["schema_sql"],
            )
            schema_literal_sql = _finish_oracle_plan(
                apply_schema_oracle(literal_plan, exact_literal_gold, "all_schema_links"),
                mapping,
                record["schema_sql"],
            )

        raw_predictions.append(raw_sql)
        for variant, prediction in variants.items():
            oracle_predictions[variant].append(prediction)
        operation_predictions.append(operation_sql)
        literal_predictions.append(literal_sql)
        schema_operation_predictions.append(schema_operation_sql)
        schema_literal_predictions.append(schema_literal_sql)
        gold_plan_predictions.append(gold_plan_sql)
        diagnostics.append(
            {
                "id": record.get("id"),
                "schema_id": record.get("schema_id"),
                "question": record["question"],
                "gold_sql": record["sql"],
                "raw_target": target,
                "raw_target_parsed": predicted is not None,
                "operation_components": operation_components,
                "production_sql": production_sql,
                "raw_plan_sql": raw_sql,
                "oracle_sql": variants,
                "operation_oracle_sql": operation_sql,
                "literal_oracle_sql": literal_sql,
                "schema_operation_oracle_sql": schema_operation_sql,
                "schema_literal_oracle_sql": schema_literal_sql,
                "gold_plan_oracle_sql": gold_plan_sql,
                "execution_correct": {
                    "production": score_prediction(record, production_sql, database_base_dir)["execution_accuracy"],
                    "raw_plan": score_prediction(record, raw_sql, database_base_dir)["execution_accuracy"],
                    "operation_structure": score_prediction(
                        record, operation_sql, database_base_dir
                    )["execution_accuracy"],
                    "filter_literals": score_prediction(
                        record, literal_sql, database_base_dir
                    )["execution_accuracy"],
                    "schema_and_operation": score_prediction(
                        record, schema_operation_sql, database_base_dir
                    )["execution_accuracy"],
                    "schema_and_literals": score_prediction(
                        record, schema_literal_sql, database_base_dir
                    )["execution_accuracy"],
                    "gold_plan": score_prediction(
                        record, gold_plan_sql, database_base_dir
                    )["execution_accuracy"],
                    **{
                        variant: score_prediction(record, prediction, database_base_dir)["execution_accuracy"]
                        for variant, prediction in variants.items()
                    },
                },
            }
        )

    denominator = max(parsed_targets, 1)
    report = {
        "records": len(records),
        "raw_target_parse_rate": parsed_targets / max(len(records), 1),
        "parsed_targets": parsed_targets,
        "production_baseline": evaluate(records, production_predictions, database_base_dir),
        "raw_plan_baseline": evaluate(records, raw_predictions, database_base_dir),
        "operation_components_given_parsed_target": {
            name: component_totals[name] / denominator
            for name in (
                "selection_arity",
                "aggregate_function",
                "aggregate_position",
                "distinct",
                "filter_count",
                "filter_operators_and_values",
                "filter_values",
                "filter_connector",
                "group_arity",
                "order_presence",
                "sort_direction",
                "limit",
                "all_operations",
            )
        },
        "schema_oracles": {
            variant: evaluate(records, predictions, database_base_dir)
            for variant, predictions in oracle_predictions.items()
        },
        "operation_oracle": evaluate(records, operation_predictions, database_base_dir),
        "literal_oracle": evaluate(records, literal_predictions, database_base_dir),
        "combined_oracles": {
            "schema_and_operation": evaluate(
                records, schema_operation_predictions, database_base_dir
            ),
            "schema_and_literals": evaluate(
                records, schema_literal_predictions, database_base_dir
            ),
        },
        "gold_plan_oracle": evaluate(records, gold_plan_predictions, database_base_dir),
        "oracle_rescue_ceilings": {
            "all_schema_links": evaluate(
                records,
                _rescue_predictions(
                    records,
                    raw_predictions,
                    oracle_predictions["all_schema_links"],
                    database_base_dir,
                ),
                database_base_dir,
            ),
            "operation_structure": evaluate(
                records,
                _rescue_predictions(
                    records,
                    raw_predictions,
                    operation_predictions,
                    database_base_dir,
                ),
                database_base_dir,
            ),
            "filter_literals": evaluate(
                records,
                _rescue_predictions(
                    records,
                    raw_predictions,
                    literal_predictions,
                    database_base_dir,
                ),
                database_base_dir,
            ),
        },
        "oracle_availability": {
            "factorized_schema_links": factorized_operation_oracle_available
            / max(parsed_targets, 1),
            "operation_oracle_eligible": operation_oracle_eligible / max(len(records), 1),
            "decoded_filter_values_for_gold_arity": operation_oracle_eligible
            / max(factorized_operation_oracle_available, 1),
        },
    }
    return report, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    tokenizer = load_tokenizer(args.checkpoint)
    model = load_model_from_checkpoint(str(args.checkpoint), tokenizer)
    report, diagnostics = run_schema_oracle(
        model,
        records,
        tokenizer,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        database_base_dir=args.data.parent,
        show_progress=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    diagnostics_path = args.output_dir / "diagnostics.jsonl"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        for diagnostic in diagnostics:
            handle.write(json.dumps(diagnostic, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path), "diagnostics": str(diagnostics_path)}, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

from pocketsql.data.validate import is_read_only_select
from pocketsql.evaluation.execute import execute_read_only, execute_read_only_database
from pocketsql.evaluation.normalize import normalize_sql


def _execute(record: dict, sql: str, database_base_dir: Path | None) -> list[tuple] | None:
    if record.get("database_path"):
        database_path = Path(record["database_path"])
        if not database_path.is_absolute():
            stored_base_dir = Path(record["_database_base_dir"]) if record.get("_database_base_dir") else None
            database_path = (database_base_dir or stored_base_dir or Path.cwd()) / database_path
        return execute_read_only_database(database_path, sql)
    return execute_read_only(record.get("database_sql", record["schema_sql"]), sql)


def _semantic_components(record: dict, prediction: str) -> dict[str, int] | None:
    """Compare supported query-plan components while ignoring harmless SQL formatting."""
    if not record.get("query_plan"):
        return None
    try:
        from pocketsql.data.gretel import query_plan_from_sql
        from pocketsql.model.semantic_plan import query_plan_from_dict

        predicted = query_plan_from_sql(prediction)
        gold = query_plan_from_dict(record["query_plan"])
    except (ImportError, KeyError, TypeError, ValueError):
        return {
            name: 0
            for name in (
                "tables",
                "projection",
                "aggregate",
                "distinct",
                "join",
                "filters",
                "group_by",
                "order_limit",
                "semantic_plan_match",
            )
        }

    def reference(value: str | None) -> str | None:
        return value.casefold() if isinstance(value, str) else value

    def references(values) -> tuple[str, ...]:
        return tuple(reference(value) for value in values)

    def filters(plan) -> Counter:
        return Counter(
            (reference(item.column), item.operator, type(item.value).__name__, item.value)
            for item in plan.filters
        )

    gold_tables = {reference(gold.table), reference(gold.join_table)} - {None}
    predicted_tables = {reference(predicted.table), reference(predicted.join_table)} - {None}
    gold_join = frozenset(reference(value) for value in gold.join_on) if gold.join_on else None
    predicted_join = frozenset(reference(value) for value in predicted.join_on) if predicted.join_on else None
    scores = {
        "tables": int(predicted_tables == gold_tables),
        "projection": int(references(predicted.columns) == references(gold.columns)),
        "aggregate": int(
            (predicted.aggregate, reference(predicted.aggregate_column), predicted.aggregate_position)
            == (gold.aggregate, reference(gold.aggregate_column), gold.aggregate_position)
        ),
        "distinct": int(predicted.distinct == gold.distinct),
        "join": int(predicted_join == gold_join),
        "filters": int(
            filters(predicted) == filters(gold)
            and predicted.filter_connector.casefold() == gold.filter_connector.casefold()
        ),
        "group_by": int(references(predicted.group_by) == references(gold.group_by)),
        "order_limit": int(
            (reference(predicted.order_by), predicted.descending, predicted.limit)
            == (reference(gold.order_by), gold.descending, gold.limit)
        ),
    }
    scores["semantic_plan_match"] = int(all(scores.values()))
    return scores


def score_prediction(record: dict, prediction: str, database_base_dir: Path | None = None) -> dict:
    syntactic = int(is_read_only_select(prediction))
    predicted = _execute(record, prediction, database_base_dir)
    gold = _execute(record, record["sql"], database_base_dir)
    executable = int(predicted is not None)
    exact = int(normalize_sql(prediction) == normalize_sql(record["sql"]))
    ordered = "ORDER BY" in record["sql"].upper()
    execution = int(
        predicted is not None
        and gold is not None
        and (predicted == gold if ordered else Counter(predicted) == Counter(gold))
    )
    if not syntactic:
        failure = "unsafe_or_incomplete_output"
    elif not executable:
        failure = "sqlite_execution_error"
    elif not execution:
        failure = "wrong_result"
    else:
        failure = "correct_execution"
    result = {
        "syntactically_valid": syntactic,
        "executable": executable,
        "exact_match": exact,
        "execution_accuracy": execution,
        "failure": failure,
    }
    components = _semantic_components(record, prediction)
    if components is not None:
        result["semantic_components"] = components
    return result


def evaluate(records: list[dict], predictions: list[str], database_base_dir: Path | None = None) -> dict:
    if len(records) != len(predictions):
        raise ValueError(f"Expected one prediction per record, got {len(predictions)} predictions for {len(records)} records.")
    counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    family_counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    complexity_counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    schema_counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    failure_counts = defaultdict(int)
    counterfactual_scores = defaultdict(list)
    counterfactual_changes = {}
    component_totals: Counter = Counter()
    component_records = 0
    totals = [0, 0, 0, 0]
    for record, prediction in zip(records, predictions):
        scored = score_prediction(record, prediction, database_base_dir)
        if scored.get("semantic_components"):
            component_totals.update(scored["semantic_components"])
            component_records += 1
        if record.get("counterfactual_group"):
            group = record["counterfactual_group"]
            counterfactual_scores[group].append(scored["execution_accuracy"])
            counterfactual_changes[group] = record.get("counterfactual_change", "unknown")
        metrics = tuple(scored[name] for name in ("syntactically_valid", "executable", "exact_match", "execution_accuracy"))
        failure_counts[scored["failure"]] += 1
        for index, value in enumerate(metrics):
            totals[index] += value
        key = f"difficulty_{record['difficulty']}:{record['query_plan']['family']}"
        for index, value in enumerate(metrics):
            counts[key][index] += value
        counts[key][4] += 1
        family_key = record["query_plan"]["family"]
        for index, value in enumerate(metrics):
            family_counts[family_key][index] += value
        family_counts[family_key][4] += 1
        table_count = len(re.findall(r"\bCREATE\s+TABLE\b", record.get("schema_sql", ""), re.IGNORECASE))
        complexity_key = f"{table_count}_tables" if table_count < 5 else "5_plus_tables"
        for index, value in enumerate(metrics):
            complexity_counts[complexity_key][index] += value
        complexity_counts[complexity_key][4] += 1
        schema_key = record.get("schema_id", "unknown")
        for index, value in enumerate(metrics):
            schema_counts[schema_key][index] += value
        schema_counts[schema_key][4] += 1
    names = ("syntactically_valid", "executable", "exact_match", "execution_accuracy")
    result = {"records": len(records), **{name: totals[index] / max(len(records), 1) for index, name in enumerate(names)}}
    result["execution_accuracy_given_valid"] = totals[3] / max(totals[0], 1)
    result["by_family"] = {key: {name: values[index] / values[4] for index, name in enumerate(names)} for key, values in counts.items()}
    result["by_query_family"] = {
        key: {**{name: values[index] / values[4] for index, name in enumerate(names)}, "examples": values[4]}
        for key, values in sorted(family_counts.items())
    }
    result["by_schema_complexity"] = {
        key: {**{name: values[index] / values[4] for index, name in enumerate(names)}, "examples": values[4]}
        for key, values in sorted(complexity_counts.items())
    }
    result["by_schema"] = {
        key: {**{name: values[index] / values[4] for index, name in enumerate(names)}, "examples": values[4]}
        for key, values in sorted(schema_counts.items())
    }
    result["failure_counts"] = {
        name: {"count": failure_counts[name], "rate": failure_counts[name] / max(len(records), 1)}
        for name in ("unsafe_or_incomplete_output", "sqlite_execution_error", "wrong_result", "correct_execution")
    }
    if component_records:
        result["semantic_components"] = {
            name: component_totals[name] / component_records
            for name in (
                "tables",
                "projection",
                "aggregate",
                "distinct",
                "join",
                "filters",
                "group_by",
                "order_limit",
                "semantic_plan_match",
            )
        }
    if counterfactual_scores:
        complete_pairs = {group: int(len(scores) == 2 and all(scores)) for group, scores in counterfactual_scores.items()}
        by_change = defaultdict(list)
        for group, correct in complete_pairs.items():
            by_change[counterfactual_changes[group]].append(correct)
        result["counterfactual_pairs"] = len(complete_pairs)
        result["counterfactual_pair_accuracy"] = sum(complete_pairs.values()) / len(complete_pairs)
        result["counterfactual_pair_accuracy_by_change"] = {
            change: sum(scores) / len(scores) for change, scores in sorted(by_change.items())
        }
    return result


def write_prediction_diagnostics(
    path: Path,
    records: list[dict],
    predictions: list[str],
    database_base_dir: Path | None = None,
) -> None:
    if len(records) != len(predictions):
        raise ValueError("diagnostic records and predictions must have equal length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record, prediction in zip(records, predictions):
            scored = score_prediction(record, prediction, database_base_dir)
            handle.write(
                json.dumps(
                    {
                        "id": record.get("id"),
                        "schema_id": record.get("schema_id"),
                        "family": record.get("query_plan", {}).get("family"),
                        "question": record["question"],
                        "gold_sql": record["sql"],
                        "predicted_sql": prediction,
                        **scored,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def generate_predictions(model, records: list[dict], tokenizer, batch_size: int = 1, max_tokens: int | None = None, show_progress: bool = False) -> list[str]:
    """Generate one safe SQL prediction per record in right-padded batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    from pocketsql.inference import generate_sql_batch

    predictions = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        predictions.extend(generate_sql_batch(model, [record["schema_sql"] for record in batch], [record["question"] for record in batch], tokenizer, max_tokens=max_tokens))
        if show_progress:
            print(f"generated {len(predictions)}/{len(records)} predictions", file=sys.stderr, flush=True)
    return predictions


def evaluate_model(
    model,
    records: list[dict],
    tokenizer,
    batch_size: int = 1,
    max_tokens: int | None = None,
    return_predictions: bool = False,
) -> dict | tuple[dict, list[str]]:
    predictions = generate_predictions(model, records, tokenizer, batch_size=batch_size, max_tokens=max_tokens)
    result = evaluate(records, predictions)
    return (result, predictions) if return_predictions else result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="One SQL statement or JSON object with a sql field per line.")
    parser.add_argument("--checkpoint", help="Generate predictions with this checkpoint instead of reading --predictions.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of prompts decoded together when generating predictions.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override the generation cap stored in the checkpoint.")
    parser.add_argument(
        "--unconstrained-semantic-plan",
        action="store_true",
        help="Disable semantic-plan grammar constraints for legacy score reproduction.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write the complete JSON report to this file.")
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=None,
        help="Write replayable JSONL predictions with record IDs.",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Write one scored diagnostic JSON object per example.",
    )
    args = parser.parse_args()
    if not args.predictions and not args.checkpoint:
        raise SystemExit("Provide --predictions <file> or --checkpoint <dir> to supply model output.")
    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    if args.checkpoint:
        from pocketsql.inference import load_model_from_checkpoint
        from pocketsql.model.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(Path(args.checkpoint))
        model = load_model_from_checkpoint(args.checkpoint, tokenizer)
        if args.unconstrained_semantic_plan:
            model.constrain_semantic_plan = False
        predictions = generate_predictions(model, records, tokenizer, batch_size=args.batch_size, max_tokens=args.max_tokens, show_progress=True)
    else:
        if not args.predictions.exists():
            raise SystemExit(f"--predictions file not found: {args.predictions}. Create it first, one SQL statement per line matching --data's record order, or pass --checkpoint instead.")
        predictions = []
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line.startswith("{"):
                predictions.append(json.loads(line)["sql"])
            else:
                predictions.append(line)
    if args.prediction_output:
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        with args.prediction_output.open("w", encoding="utf-8") as handle:
            for record, prediction in zip(records, predictions):
                handle.write(json.dumps({"id": record.get("id"), "sql": prediction}, sort_keys=True) + "\n")
    if args.diagnostics:
        write_prediction_diagnostics(args.diagnostics, records, predictions, args.data.parent)
    report = json.dumps(evaluate(records, predictions, args.data.parent), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print({"output": str(args.output), "records": len(records)})
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

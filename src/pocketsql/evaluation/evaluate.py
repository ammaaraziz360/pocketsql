from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from pocketsql.data.validate import is_read_only_select
from pocketsql.evaluation.execute import execute_read_only
from pocketsql.evaluation.normalize import normalize_sql


def score_prediction(record: dict, prediction: str) -> dict:
    syntactic = int(is_read_only_select(prediction))
    database_sql = record.get("database_sql", record["schema_sql"])
    predicted = execute_read_only(database_sql, prediction)
    gold = execute_read_only(database_sql, record["sql"])
    executable = int(predicted is not None)
    exact = int(normalize_sql(prediction) == normalize_sql(record["sql"]))
    ordered = "ORDER BY" in record["sql"].upper()
    execution = int(predicted is not None and gold is not None and (predicted == gold if ordered else sorted(predicted) == sorted(gold)))
    if not syntactic:
        failure = "unsafe_or_incomplete_output"
    elif not executable:
        failure = "sqlite_execution_error"
    elif not execution:
        failure = "wrong_result"
    else:
        failure = "correct_execution"
    return {
        "syntactically_valid": syntactic,
        "executable": executable,
        "exact_match": exact,
        "execution_accuracy": execution,
        "failure": failure,
    }


def evaluate(records: list[dict], predictions: list[str]) -> dict:
    if len(records) != len(predictions):
        raise ValueError(f"Expected one prediction per record, got {len(predictions)} predictions for {len(records)} records.")
    counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    schema_counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    failure_counts = defaultdict(int)
    totals = [0, 0, 0, 0]
    for record, prediction in zip(records, predictions):
        scored = score_prediction(record, prediction)
        metrics = tuple(scored[name] for name in ("syntactically_valid", "executable", "exact_match", "execution_accuracy"))
        failure_counts[scored["failure"]] += 1
        for index, value in enumerate(metrics):
            totals[index] += value
        key = f"difficulty_{record['difficulty']}:{record['query_plan']['family']}"
        for index, value in enumerate(metrics):
            counts[key][index] += value
        counts[key][4] += 1
        schema_key = record.get("schema_id", "unknown")
        for index, value in enumerate(metrics):
            schema_counts[schema_key][index] += value
        schema_counts[schema_key][4] += 1
    names = ("syntactically_valid", "executable", "exact_match", "execution_accuracy")
    result = {name: totals[index] / max(len(records), 1) for index, name in enumerate(names)}
    result["by_family"] = {key: {name: values[index] / values[4] for index, name in enumerate(names)} for key, values in counts.items()}
    result["by_schema"] = {
        key: {**{name: values[index] / values[4] for index, name in enumerate(names)}, "examples": values[4]}
        for key, values in sorted(schema_counts.items())
    }
    result["failure_counts"] = {
        name: {"count": failure_counts[name], "rate": failure_counts[name] / max(len(records), 1)}
        for name in ("unsafe_or_incomplete_output", "sqlite_execution_error", "wrong_result", "correct_execution")
    }
    return result


def write_prediction_diagnostics(path: Path, records: list[dict], predictions: list[str]) -> None:
    if len(records) != len(predictions):
        raise ValueError("diagnostic records and predictions must have equal length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record, prediction in zip(records, predictions):
            scored = score_prediction(record, prediction)
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
    parser.add_argument("--output", type=Path, default=None, help="Write the complete JSON report to this file.")
    args = parser.parse_args()
    if not args.predictions and not args.checkpoint:
        raise SystemExit("Provide --predictions <file> or --checkpoint <dir> to supply model output.")
    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    if args.checkpoint:
        from pocketsql.inference import load_model_from_checkpoint
        from pocketsql.model.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(Path(args.checkpoint))
        model = load_model_from_checkpoint(args.checkpoint, tokenizer)
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
    report = json.dumps(evaluate(records, predictions), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print({"output": str(args.output), "records": len(records)})
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

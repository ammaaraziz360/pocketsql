from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from pocketsql.data.validate import is_read_only_select
from pocketsql.evaluation.execute import execute_read_only
from pocketsql.evaluation.normalize import normalize_sql


def evaluate(records: list[dict], predictions: list[str]) -> dict:
    counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    totals = [0, 0, 0, 0]
    for record, prediction in zip(records, predictions):
        syntactic = int(is_read_only_select(prediction))
        database_sql = record.get("database_sql", record["schema_sql"])
        predicted = execute_read_only(database_sql, prediction)
        gold = execute_read_only(database_sql, record["sql"])
        executable = int(predicted is not None)
        exact = int(normalize_sql(prediction) == normalize_sql(record["sql"]))
        ordered = "ORDER BY" in record["sql"].upper()
        execution = int(predicted is not None and gold is not None and (predicted == gold if ordered else sorted(predicted) == sorted(gold)))
        metrics = (syntactic, executable, exact, execution)
        for index, value in enumerate(metrics):
            totals[index] += value
        key = f"difficulty_{record['difficulty']}:{record['query_plan']['family']}"
        for index, value in enumerate(metrics):
            counts[key][index] += value
        counts[key][4] += 1
    names = ("syntactically_valid", "executable", "exact_match", "execution_accuracy")
    result = {name: totals[index] / max(len(records), 1) for index, name in enumerate(names)}
    result["by_family"] = {key: {name: values[index] / values[4] for index, name in enumerate(names)} for key, values in counts.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="One SQL statement or JSON object with a sql field per line.")
    parser.add_argument("--checkpoint", help="Generate predictions with this checkpoint instead of reading --predictions.")
    args = parser.parse_args()
    if not args.predictions and not args.checkpoint:
        raise SystemExit("Provide --predictions <file> or --checkpoint <dir> to supply model output.")
    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    if args.checkpoint:
        from pocketsql.inference import generate_sql, load_model_from_checkpoint
        from pocketsql.model.tokenizer import ByteTokenizer

        tokenizer = ByteTokenizer()
        model = load_model_from_checkpoint(args.checkpoint, tokenizer)
        predictions = []
        for record in records:
            try:
                predictions.append(generate_sql(model, record["schema_sql"], record["question"], tokenizer))
            except ValueError:
                predictions.append("")
    else:
        if not args.predictions.exists():
            raise SystemExit(f"--predictions file not found: {args.predictions}. Create it first, one SQL statement per line matching --data's record order, or pass --checkpoint instead.")
        predictions = []
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line.startswith("{"):
                predictions.append(json.loads(line)["sql"])
            else:
                predictions.append(line)
    print(json.dumps(evaluate(records, predictions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
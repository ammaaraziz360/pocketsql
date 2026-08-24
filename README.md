# PocketSQL

PocketSQL is a Phase 1, locally trained text-to-SQL experiment for Apple Silicon. It trains a decoder-only MLX model to turn a SQLite schema plus a natural-language question into one read-only `SELECT` statement. It is not an API wrapper and does not require an external LLM.

## Setup

Use an Apple-Silicon Python 3.11, 3.12, or 3.13 environment (MLX does not currently support the workspace's Python 3.14):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

If `python -m pocketsql...` raises `ModuleNotFoundError: No module named 'pocketsql'` after installing, some macOS setups silently mark pip's editable `.pth` file as hidden, which `site.py` ignores. Work around it by exporting `PYTHONPATH` directly (already appended to `.venv/bin/activate` in this repo, so re-running `source .venv/bin/activate` picks it up):

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

## Generate data

The generator creates domain schemas, valid SQLite rows, AST-backed SQL/questions, and schema-disjoint train/validation/test JSONL files. A record includes `database_sql` in addition to the requested fields so execution evaluation has its original populated data.

```bash
python -m pocketsql.data.generate --output data/generated --schemas 100 --examples-per-schema 50 --seed 42
```

## Train

The default base config has six 384-wide layers. Use tiny for smoke tests or overfitting a small slice:

```bash
python -m pocketsql.training.train --config configs/tiny.yaml --data data/generated/train.jsonl --checkpoint checkpoints/tiny --overfit 32
python -m pocketsql.training.train --config configs/base.yaml --data data/generated/train.jsonl --val-data data/generated/validation.jsonl --checkpoint checkpoints/base
```

Add `--resume checkpoints/base` to continue from a saved checkpoint (model weights, optimizer state including AdamW moments, and the learning-rate schedule position are all restored). Each config supports `warmup_steps` (linear warmup then constant learning rate) and `grad_accum_steps` (micro-batches averaged before each optimizer update). Training and validation loss are printed once per epoch, and a checkpoint is saved after every epoch.

Training sequences are `<bos><schema>...</schema><question>...</question><sql>...</sql><eos>`. The byte tokenizer has no unknown token; SQL-region masking excludes schema and question tokens from direct loss.

## Inference and evaluation

```bash
python -m pocketsql.inference --checkpoint checkpoints/tiny --schema 'CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT);' --question 'list customer names'
python -m pocketsql.evaluation.evaluate --data data/generated/test.jsonl --checkpoint checkpoints/tiny
```

`pocketsql.inference.generate_sql(model, schema, question)` stops at `</sql>` or `<eos>`, extracts only SQL, and rejects multi-statement or non-`SELECT` output. A checkpoint produced only via `--overfit` on a handful of examples (as in the training commands above) has not learned to generalize, so it commonly raises `model output is not a single read-only SELECT` on new schemas/questions — that is expected, not a bug; the error message includes the raw decoded text for debugging. Train longer on the full split for usable generations.

`pocketsql.evaluation.evaluate` accepts either `--checkpoint <dir>` (generates predictions itself, one per record in `--data`) or `--predictions <file>` (a pre-existing file with one SQL statement, or one `{"sql": ...}` JSON object, per line matching `--data`'s record order). It reports syntax validity, executability, normalized exact match, execution accuracy, and per-family/difficulty metrics. SQLite evaluation is query-only and caps returned rows.

## Tests

```bash
pytest
```

## Design and current limits

Synthetic questions and SQL are both rendered from immutable typed `QueryPlan` objects. The MVP supports projection, `DISTINCT`, compatible filters with `AND`/`OR`, aggregates, grouping, ordering/limit, and declared two-table joins. It intentionally excludes subqueries, CTEs, window functions, mutations, and joins beyond two tables. The verbalizer is deliberately template-driven; the next milestone is richer paraphrase generation and a training CLI that resumes optimizer state with warmup and gradient accumulation.
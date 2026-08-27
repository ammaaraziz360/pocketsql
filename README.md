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

For the position-robust corpus, shuffle physical column order, vary each schema
from 9 to 13 columns, request every physical column, retain multi-column
projections, and balance the supported SQL intents:

```bash
python -m pocketsql.data.generate \
  --output data/generated-position-robust \
  --schemas 1000 \
  --examples-per-schema 75 \
  --seed 5150 \
  --family-weights '{"join": 4, "and_filter": 3, "or_filter": 3, "select": 7, "distinct": 3, "count": 3, "group": 3, "filter": 2, "sum": 2, "avg": 2, "min": 2, "max": 2, "order_limit": 2}' \
  --question-style mixed
```

Generated schemas now vary table/column terminology, identifier style (plain,
camelCase, prefixed, and suffixed), and include unused columns.  The generator
writes `quality_report.json` alongside the JSONL splits.  It records family
mix, duplicate schema/question pairs, schema-split overlap, and identifier
novelty relative to training.

Create a separate evaluation-only challenge set before trusting a new model.
Its commerce, logistics, and software identifiers are deliberately outside the
normal training vocabulary, and its mix emphasizes joins and compound filters:

```bash
python -m pocketsql.data.challenge --output data/challenge --schemas 120 --examples-per-schema 75 --reference-data data/generated-position-robust/train.jsonl
python -m pocketsql.evaluation.evaluate --data data/challenge/challenge.jsonl --checkpoint checkpoints/base-position-robust-pilot-best-execution --batch-size 32
```

Do not train on `data/challenge`: use it to reveal template and vocabulary
overfitting.  Regenerate both training data and challenge data with recorded
seeds when changing the generator.

For an even stricter identifier check, build the evaluation-only opaque set.
Every table and column name is a random string with no semantic meaning:

```bash
python -m pocketsql.data.grounding_dev \
  --output data/grounding-dev \
  --schemas 120 \
  --examples-per-schema 75 \
  --reference-data data/generated-realistic/train.jsonl
```

Never include `data/grounding-dev/opaque.jsonl` in training. Build the separate
held-out casual-language benchmark as well:

```bash
python -m pocketsql.data.casual_dev --output data/casual-dev --schemas 120 --examples-per-schema 50
python -m pocketsql.evaluation.evaluate --data data/casual-dev/casual.jsonl --checkpoint checkpoints/base-position-robust-pilot-best-execution --batch-size 32
```

Never include `data/casual-dev/casual.jsonl` in training.

Build the all-column copying gate separately. It requests every physical column
at every position and is also evaluation-only:

```bash
python -m pocketsql.data.column_copy_dev --output data/column-copy-dev --schemas 120
python -m pocketsql.evaluation.evaluate --data data/column-copy-dev/column_copy.jsonl --checkpoint checkpoints/base-position-robust-pilot-best-execution --batch-size 32
```

## Train

The recommended grounded configuration converts the schema's real identifiers
to reversible slots such as `table0` and `column3`. The model generates only
those schema-provided slots; inference then restores the original names. This
prevents it from inventing familiar training identifiers for a new schema.

Train its local byte-level BPE tokenizer only from the training split. Permuted
slots deterministically assign the same physical identifier to different
`columnN` positions across schemas, preventing a fixed-position shortcut:

```bash
python -m pocketsql.model.train_tokenizer \
  --data data/generated-position-robust/train.jsonl \
  --output artifacts/tokenizers/position-robust-bpe.json \
  --vocab-size 2048 \
  --canonicalize-identifiers \
  --identifier-slot-strategy permuted
```

Audit lengths before spending time on a run. Training performs the same check automatically and refuses to start if any SQL target is missing, partial, or longer than the generation cap:

```bash
python -m pocketsql.training.audit --data data/generated-position-robust/train.jsonl --config configs/base_position_robust.yaml
python -m pocketsql.training.audit --data data/casual-dev/casual.jsonl --config configs/base_position_robust.yaml
python -m pocketsql.training.audit --data data/challenge/challenge.jsonl --config configs/base_position_robust.yaml
```

Run the 32-example execution gate after changing tokenization, masking, model context, or decoding. It must reach at least 95% execution accuracy before a full run:

```bash
python -m pocketsql.training.train \
  --config configs/overfit_position_robust.yaml \
  --data data/generated-position-robust/train.jsonl \
  --checkpoint checkpoints/overfit-position-robust-gate \
  --overfit 32 \
  --execution-gate 0.95
```

Then run a two-epoch full-data pilot:

```bash
python -m pocketsql.training.train \
  --config configs/base_position_robust.yaml \
  --epochs 2 \
  --data data/generated-position-robust/train.jsonl \
  --val-data data/generated-position-robust/validation.jsonl \
  --checkpoint checkpoints/base-position-robust-pilot
```

Only continue the pilot if executable and execution-accuracy metrics begin improving. The base model has six 384-wide layers. The legacy byte tokenizer remains available for old checkpoints:

```bash
python -m pocketsql.training.train --config configs/tiny.yaml --data data/generated/train.jsonl --checkpoint checkpoints/tiny --overfit 32
```

Add `--resume checkpoints/base` to continue from a saved checkpoint (model weights, optimizer state including AdamW moments, and the learning-rate schedule position are all restored). Each config supports `warmup_steps` (linear warmup then constant learning rate) and `grad_accum_steps` (micro-batches averaged before each optimizer update). Training and validation loss are printed once per epoch, and a checkpoint is saved after every epoch. When validation data is supplied, the lowest-loss model is also written to a sibling directory such as `checkpoints/base-best`; pass `--best-checkpoint` to choose another path. `early_stopping_patience` stops after that many consecutive non-improving validation epochs (omit it to disable early stopping), and `early_stopping_min_delta` defines the required loss improvement.

Set `validation_execution_every` to evaluate generated SQL on validation data at that interval. The base configuration evaluates a deterministic, schema- and family-stratified sample of 100 records every epoch, then saves the strongest execution-accuracy checkpoint to `checkpoints/base-best-execution` (override with `--best-execution-checkpoint`). Set `validation_execution_max_examples: 0` to score the full validation split. `validation_execution_batch_size` controls right-padded batched decoding. Evaluation reports overall, family, and schema-level metrics.

Every CLI training run writes TensorBoard event files to `runs/<checkpoint-name>` by default (override with `--log-dir`). It includes losses, overall execution metrics, failure categories, per-family metrics, and a table of example predictions. Raw validation predictions are also saved as `validation_predictions_epoch_XXXX.jsonl` inside the run directory. In another terminal, launch the local dashboard and open the URL it prints:

```bash
source .venv/bin/activate
tensorboard --logdir runs --port 6006
```

Track `loss/train_step`, `loss/train_epoch`, `loss/validation`, and the `validation_execution/*` metrics. Reusing the same checkpoint with `--resume` also reuses its default log directory, so the charts continue across the resumed run.

Training sequences are `<bos><schema>...</schema><question>...</question><sql>...</sql><eos>`. SQL-region masking excludes schema and question tokens from direct loss. Grounded checkpoints store the tokenizer and grounding setting, so the ordinary inference command handles slot conversion and name restoration automatically.

## Inference and evaluation

```bash
python -m pocketsql.inference --checkpoint checkpoints/base-position-robust-pilot-best-execution --schema 'CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, note TEXT); CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), total REAL, status TEXT, remarks TEXT);' --question 'what are the customer_id?'
python -m pocketsql.evaluation.evaluate --data data/generated-position-robust/test.jsonl --checkpoint checkpoints/base-position-robust-pilot-best-execution --batch-size 32
```

`pocketsql.inference.generate_sql(model, schema, question)` stops at `</sql>` or `<eos>`, extracts only SQL, and rejects multi-statement or non-`SELECT` output. A checkpoint produced only via `--overfit` on a handful of examples (as in the training commands above) has not learned to generalize, so it commonly raises `model output is not a single read-only SELECT` on new schemas/questions — that is expected, not a bug; the error message includes the raw decoded text for debugging. Train longer on the full split for usable generations.

`pocketsql.evaluation.evaluate` accepts either `--checkpoint <dir>` (generates predictions itself, one per record in `--data`) or `--predictions <file>` (a pre-existing file with one SQL statement, or one `{"sql": ...}` JSON object, per line matching `--data`'s record order). It reports syntax validity, executability, normalized exact match, execution accuracy, and per-family/difficulty metrics. SQLite evaluation is query-only and caps returned rows.

## Tests

```bash
pytest
```

## Design and current limits

Synthetic questions and SQL are both rendered from immutable typed `QueryPlan` objects. The MVP supports projection, `DISTINCT`, compatible filters with `AND`/`OR`, aggregates, grouping, ordering/limit, and declared two-table joins. It intentionally excludes subqueries, CTEs, window functions, mutations, and joins beyond two tables. The verbalizer is deliberately template-driven.

The protected epoch-6 position-robust checkpoint returns
`SELECT customer_id FROM customers;` for both `what are the customer_id?` and
`what are the customer ids?`. On evaluation-only data it reached 93.48%
all-column-copy execution accuracy (1,233/1,319), compared with 3.18% for the
previous casual-v2 model. It also reached 92.52% execution accuracy on the
7,500-example position-robust mixed-query test. All-column projections are no
longer tied to a fixed schema position.

This improvement has a measured tradeoff. The same checkpoint reaches 44.25%
on the older 6,000-example held-out casual benchmark and 66.31% on the older
9,000-example identifier challenge; the previous casual-v2 checkpoint reached
74% and 100% respectively. Later high-rate curriculum runs showed catastrophic
forgetting, and the low-rate balanced branch did not beat the protected
checkpoint, so those branches are not recommended. This is evidence that the
6-layer, 384-wide model is capacity-limited on the combined position-robust and
broad-language task, not evidence that arbitrary text-to-SQL is solved.

The next model-quality experiment should keep the balanced, schema-disjoint
corpus and train a larger model (for example 8 layers at width 512) from scratch,
using the fixed position-copy, casual-language, and identifier challenge sets
as independent checkpoint-selection gates. Do not train on any of those three
evaluation-only sets.

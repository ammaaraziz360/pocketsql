# PocketSQL

PocketSQL is a Phase 1, locally trained text-to-SQL experiment for Apple Silicon. It trains a decoder-only MLX model to turn a SQLite schema plus a natural-language question into one read-only `SELECT` statement. It is not an API wrapper and does not require an external LLM.

## Local playground

The project includes a small local website that calls the trained model directly. From the repository root, with the project environment activated, run:

```bash
python website/server.py
```

Then open <http://127.0.0.1:4173>. The default checkpoint is `base-semantic-v14-factorized-best-execution`; set `POCKETSQL_CHECKPOINT` to use another checkpoint.

## Setup

Use an Apple-Silicon Python 3.11, 3.12, or 3.13 environment (MLX does not currently support the workspace's Python 3.14):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Importing the optional external corpora also needs the external-data dependencies:

```bash
python -m pip install -e '.[dev,external-data]'
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

For the v8 position-robust corpus, shuffle physical column order, vary each
schema from 9 to 13 columns, request every physical column, retain multi-column
projections, and include the composed intents that motivated v8: filtered
counts, filtered joins, and filtered aggregate joins:

```bash
python -m pocketsql.data.generate \
  --output data/generated-position-robust-v8 \
  --schemas 1000 \
  --examples-per-schema 100 \
  --seed 8080 \
  --profile position-robust-v8 \
  --question-style mixed
```

This 100,000-example recipe selects every physical column in every training
schema. It guarantees location/text filters, multi-column projections, numeric
thresholds through 140, two-, three-, and four-value `OR` predicates, and
20,000 examples across the four new composition families. Child foreign keys
are populated across every parent row so filtered join queries have meaningful
execution results.

The selected checkpoint also receives one replay epoch from a mix that restores
extra weight to ordinary joins and aggregates:

```bash
python -m pocketsql.data.generate \
  --output data/generated-position-robust-v8-replay \
  --schemas 1000 \
  --examples-per-schema 100 \
  --seed 8181 \
  --profile position-robust-v8-replay \
  --question-style mixed
```

### Curate Gretel external data

PocketSQL can import a deliberately narrow subset of
[`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql).
The importer pins an immutable dataset revision and SHA-256 checksums, rejects
mutations and SQL outside PocketSQL's current query-plan representation,
executes both the source and normalized SQL in SQLite, rejects empty or changed
results, and verifies that training-time and inference-time grounding are
identical. It ignores the source's official split and re-splits by normalized
schema hash because the official train/test files contain overlapping schemas.

Build the selected 5,000-example pilot, a frozen 1,000-example external gate,
and fixed-size mixes containing 10% external data:

```bash
python -m pocketsql.data.gretel \
  --output data/gretel-pilot-v2 \
  --pilot-records 5000 \
  --gate-records 1000 \
  --seed 9191 \
  --compatibility-config configs/base_position_robust_v8.yaml \
  --mix-with data/generated-position-robust-v8-replay \
  --external-fraction 0.1 \
  --mixed-train-records 20000 \
  --mixed-validation-records 2500
```

The output contains one record per external schema and schema-disjoint
`train`, `validation`, `test`, and `external_gate` files. Only `train` may be
used for gradient updates. Keep `external_gate.jsonl` frozen and never mix it
into training or checkpoint selection. `quality_report.json` records the exact
revision, checksums, seed, rejection reasons, family mix, and split-overlap
checks.

### Curate WikiSQL external data

The better-aligned second external source is Salesforce's
[`WikiSQL`](https://github.com/salesforce/WikiSQL) v1.1: human-written
questions paired with simple, single-table SQL over Wikipedia tables. The
importer downloads a pinned official archive, verifies its SHA-256 checksum,
preserves the official table-disjoint train/dev/test boundaries, and keeps only
examples that PocketSQL can ground and execute exactly. It also sanitizes
headers into SQLite identifiers without replacing the original question text.

Build the selected 5,000-example pilot, its frozen 1,000-example gate, and 10%
external-data training mixes with:

```bash
python -m pocketsql.data.wikisql \
  --output data/wikisql-pilot-v1 \
  --pilot-records 5000 \
  --gate-records 1000 \
  --seed 9292 \
  --compatibility-config configs/base_position_robust_v8.yaml \
  --mix-with data/generated-position-robust-v8-replay \
  --external-fraction 0.1 \
  --mixed-train-records 20000 \
  --mixed-validation-records 2500
```

This produces 4,000 training records, 500 validation records, 500 test
records, and 1,000 gate records, with one example per schema. The selected mix
contains twice as many filters as aggregates. Only `train.jsonl` and the
training portion of `mixed_train.jsonl` are authorized for gradient updates;
`external_gate.jsonl` remains evaluation-only. WikiSQL does not contain joins,
grouping, ordering, or `DISTINCT`, so it complements rather than replaces the
internal PocketSQL corpus.

### Experimental v9 compositional corpus

V9 is a clean experimental rebuild that samples projection, aggregation,
filtering, joining, grouping, ordering, and limiting as separate decisions
instead of selecting a fixed query family. Its 25/35/40 atomic/pair/multi
curriculum still requests every physical column, while four complete operation
combinations are reserved for evaluation so ordinary test accuracy cannot hide
template interpolation.

Build the 25,000-example pilot and its two evaluation-only gates with recorded
seeds:

```bash
python -m pocketsql.data.generate \
  --output data/generated-composition-v9-pilot \
  --schemas 250 \
  --examples-per-schema 100 \
  --seed 9090 \
  --profile composition-v9

python -m pocketsql.data.composition_dev \
  --output data/v9-dev \
  --composition-schemas 120 \
  --counterfactual-schemas 120 \
  --seed 9091
```

`composition.jsonl` contains four operation combinations that never occur in
v9 training. `counterfactual.jsonl` contains 600 minimal pairs where only one
decision changes: projection, literal, comparison operator, aggregate, or a
joined literal. Never include either file in training. Their language remains
in-distribution so these gates isolate query construction; `casual-dev` remains
the independent language-generalization gate.

Generated schemas now vary table/column terminology, identifier style (plain,
camelCase, prefixed, and suffixed), and include unused columns.  The generator
writes `quality_report.json` alongside the JSONL splits.  It records family
mix, duplicate schema/question pairs, schema-split overlap, and identifier
novelty relative to training.

Create a separate evaluation-only challenge set before trusting a new model.
Its commerce, logistics, and software identifiers are deliberately outside the
normal training vocabulary, and its mix emphasizes joins and compound filters:

```bash
python -m pocketsql.data.challenge --output data/challenge --schemas 120 --examples-per-schema 75 --reference-data data/generated-position-robust-v8/train.jsonl
python -m pocketsql.evaluation.evaluate --data data/challenge/challenge.jsonl --checkpoint checkpoints/base-position-robust-v8-best --batch-size 16
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
python -m pocketsql.evaluation.evaluate --data data/casual-dev/casual.jsonl --checkpoint checkpoints/base-position-robust-v8-best --batch-size 16
```

Never include `data/casual-dev/casual.jsonl` in training.

Build the stricter paired anti-memorization gate to measure how much accuracy
survives when exact schema wording is removed. Each pair has the same unseen
schema, database, and gold SQL. The first question names the identifiers; the
second uses domain synonyms, and half of the pairs also use a held-out operation
composition:

```bash
python -m pocketsql.data.anti_memorization \
  --output data/anti-memorization-v1 \
  --schemas 24 \
  --seed 424242 \
  --reference-data data/schema-linking-v13/mixed_train.jsonl

python -m pocketsql.evaluation.evaluate \
  --data data/anti-memorization-v1/anti_memorization.jsonl \
  --checkpoint checkpoints/base-semantic-v14-factorized-best-execution \
  --batch-size 8 \
  --output data/anti-memorization-v1/results/v14/report.json
```

Never train on this gate. Its report includes direct-versus-paraphrased
accuracy, paired retention, and separate familiar/held-out composition slices.

Build the all-column copying gate separately. It requests every physical column
at every position and is also evaluation-only:

```bash
python -m pocketsql.data.column_copy_dev --output data/column-copy-dev --schemas 120
python -m pocketsql.evaluation.evaluate --data data/column-copy-dev/column_copy.jsonl --checkpoint checkpoints/base-position-robust-v8-best --batch-size 16
```

## Train

The recommended grounded configuration converts the schema's real identifiers
to reversible slots such as `table0` and `column3`. The model generates only
those schema-provided slots; inference then restores the original names. This
prevents it from inventing familiar training identifiers for a new schema.
Grounding also links common singular, plural, spaced, and underscored aliases,
so a phrase such as `customers_id` can resolve to the schema's `customer_id`
instead of being treated as unrelated text.
V8 applies the same idea to filter and limit values: `Houston`, `140`, and
similar literals become reversible `valueN` slots while the model is running.
This teaches copying and composition instead of memorizing a closed value list.
Inference also exposes the schema's unique location column when a casual
question says only `from Houston`, and can conservatively repair one invalid
location predicate when the schema leaves exactly one possible column.

Train its local byte-level BPE tokenizer only from the training split. Permuted
slots deterministically assign the same physical identifier to different
`columnN` positions across schemas, preventing a fixed-position shortcut:

```bash
python -m pocketsql.model.train_tokenizer \
  --data data/generated-position-robust-v8/train.jsonl \
  --output artifacts/tokenizers/position-robust-v8-bpe.json \
  --vocab-size 2048 \
  --canonicalize-identifiers \
  --identifier-slot-strategy permuted \
  --canonicalize-literals
```

Audit lengths before spending time on a run. Training performs the same check automatically and refuses to start if any SQL target is missing, partial, or longer than the generation cap:

```bash
python -m pocketsql.training.audit --data data/generated-position-robust-v8/train.jsonl --config configs/base_position_robust_v8.yaml
python -m pocketsql.training.audit --data data/casual-dev/casual.jsonl --config configs/base_position_robust_v8.yaml
python -m pocketsql.training.audit --data data/challenge/challenge.jsonl --config configs/base_position_robust_v8.yaml
```

Run the 32-example execution gate after changing tokenization, masking, model context, or decoding. It must reach at least 95% execution accuracy before a full run:

```bash
python -m pocketsql.training.train \
  --config configs/overfit_position_robust_v8.yaml \
  --data data/generated-position-robust-v8/train.jsonl \
  --checkpoint checkpoints/overfit-position-robust-v8-grounded-gate \
  --overfit 32 \
  --execution-gate 0.95
```

Then run the full-data pilot. V8's selected base run trained for four epochs:

```bash
python -m pocketsql.training.train \
  --config configs/base_position_robust_v8.yaml \
  --epochs 4 \
  --data data/generated-position-robust-v8/train.jsonl \
  --val-data data/generated-position-robust-v8/validation.jsonl \
  --checkpoint checkpoints/base-position-robust-v8-grounded-pilot
```

Run one replay epoch from that checkpoint and select by sampled validation
execution accuracy:

```bash
python -m pocketsql.training.train \
  --config configs/base_position_robust_v8.yaml \
  --epochs 5 \
  --data data/generated-position-robust-v8-replay/train.jsonl \
  --val-data data/generated-position-robust-v8-replay/validation.jsonl \
  --checkpoint checkpoints/base-position-robust-v8-replay-pilot \
  --resume checkpoints/base-position-robust-v8-grounded-pilot
```

The selected external-data pilot starts from v8, keeps the mixed epoch the same
size as the replay epoch, and uses a low learning rate:

```bash
python -m pocketsql.training.train \
  --config configs/base_gretel_pilot_v2.yaml \
  --data data/gretel-pilot-v2/mixed_train.jsonl \
  --val-data data/gretel-pilot-v2/mixed_validation.jsonl \
  --checkpoint checkpoints/base-gretel-pilot-v2 \
  --resume checkpoints/base-position-robust-v8-best
```

The complete fine-tune gained external coverage but regressed on one simple
manual prompt. A 50/50 weight interpolation with v8 restored that behavior and
performed better on every established v8 gate. Recreate the inference-only
checkpoint with:

```bash
python -m pocketsql.training.interpolate \
  --base checkpoints/base-position-robust-v8-best \
  --fine-tuned checkpoints/base-gretel-pilot-v2 \
  --fine-tuned-weight 0.5 \
  --output checkpoints/base-gretel-augmented-v1
```

Train one low-learning-rate WikiSQL mix epoch from the same v8 checkpoint:

```bash
python -m pocketsql.training.train \
  --config configs/base_wikisql_pilot_v1.yaml \
  --data data/wikisql-pilot-v1/mixed_train.jsonl \
  --val-data data/wikisql-pilot-v1/mixed_validation.jsonl \
  --checkpoint checkpoints/base-wikisql-pilot-v1 \
  --resume checkpoints/base-position-robust-v8-best
```

The standalone fine-tune forgets too much of the casual-language behavior. The
selected model therefore uses two interpolation steps: retain 75% of the
WikiSQL fine-tune in a v8 soup, then add 25% of that soup to the already selected
Gretel model:

```bash
python -m pocketsql.training.interpolate \
  --base checkpoints/base-position-robust-v8-best \
  --fine-tuned checkpoints/base-wikisql-pilot-v1 \
  --fine-tuned-weight 0.75 \
  --output checkpoints/base-wikisql-pilot-v1-soup-75

python -m pocketsql.training.interpolate \
  --base checkpoints/base-gretel-augmented-v1 \
  --fine-tuned checkpoints/base-wikisql-pilot-v1-soup-75 \
  --fine-tuned-weight 0.25 \
  --output checkpoints/base-external-combined-v1-25
```

The resulting weights are 43.75% v8, 37.5% Gretel fine-tune, and 18.75%
WikiSQL fine-tune. `checkpoints/base-external-augmented-best` is the stable
alias for this inference-only checkpoint.

The experimental v9 pilot uses a longer context and generation allowance for
multi-operation queries:

```bash
python -m pocketsql.training.audit \
  --data data/generated-composition-v9-pilot/train.jsonl \
  --config configs/base_composition_v9_pilot.yaml

python -m pocketsql.training.train \
  --config configs/overfit_composition_v9_pilot.yaml \
  --data data/generated-composition-v9-pilot/train.jsonl \
  --checkpoint checkpoints/overfit-composition-v9-pilot-gate \
  --overfit 32 \
  --execution-gate 0.95

python -m pocketsql.training.train \
  --config configs/base_composition_v9_pilot.yaml \
  --data data/generated-composition-v9-pilot/train.jsonl \
  --val-data data/generated-composition-v9-pilot/validation.jsonl \
  --checkpoint checkpoints/base-composition-v9-pilot
```

Evaluate all three v9 views before considering a full-size run:

```bash
python -m pocketsql.evaluation.evaluate --data data/generated-composition-v9-pilot/test.jsonl --checkpoint checkpoints/base-composition-v9-corrected-pilot --batch-size 8
python -m pocketsql.evaluation.evaluate --data data/v9-dev/composition.jsonl --checkpoint checkpoints/base-composition-v9-corrected-pilot --batch-size 8
python -m pocketsql.evaluation.evaluate --data data/v9-dev/counterfactual.jsonl --checkpoint checkpoints/base-composition-v9-corrected-pilot --batch-size 8
```

Only continue the pilot if executable and execution-accuracy metrics begin improving. The recommended model has eight 512-wide layers. The legacy byte tokenizer remains available for old checkpoints:

```bash
python -m pocketsql.training.train --config configs/tiny.yaml --data data/generated/train.jsonl --checkpoint checkpoints/tiny --overfit 32
```

Add `--resume checkpoints/base` to continue from a saved checkpoint (model weights, optimizer state including AdamW moments, and the learning-rate schedule position are all restored). Each config supports `warmup_steps` (linear warmup then constant learning rate) and `grad_accum_steps` (micro-batches averaged before each optimizer update). Training and validation loss are printed once per epoch, and a checkpoint is saved after every epoch. When validation data is supplied, the lowest-loss model is also written to a sibling directory such as `checkpoints/base-best`; pass `--best-checkpoint` to choose another path. `early_stopping_patience` stops after that many consecutive non-improving validation epochs (omit it to disable early stopping), and `early_stopping_min_delta` defines the required loss improvement.

Set `validation_execution_every` to evaluate generated SQL on validation data at that interval. The base configuration evaluates a deterministic, schema- and family-stratified sample every epoch (160 records in v8), then saves the strongest execution-accuracy checkpoint to `checkpoints/base-best-execution` (override with `--best-execution-checkpoint`). Set `validation_execution_max_examples: 0` to score the full validation split. `validation_execution_batch_size` controls right-padded batched decoding. Evaluation reports overall, family, and schema-level metrics.

Every CLI training run writes TensorBoard event files to `runs/<checkpoint-name>` by default (override with `--log-dir`). It includes losses, overall execution metrics, failure categories, per-family metrics, and a table of example predictions. Raw validation predictions are also saved as `validation_predictions_epoch_XXXX.jsonl` inside the run directory. In another terminal, launch the local dashboard and open the URL it prints:

```bash
source .venv/bin/activate
tensorboard --logdir runs --port 6006
```

Track `loss/train_step`, `loss/train_epoch`, `loss/validation`, and the `validation_execution/*` metrics. Reusing the same checkpoint with `--resume` also reuses its default log directory, so the charts continue across the resumed run.

Training sequences are `<bos><schema>...</schema><question>...</question><sql>...</sql><eos>`. The target envelope contains SQL for legacy checkpoints and the compact semantic-plan DSL for checkpoints configured with `target_format: semantic_plan`. Target-region masking excludes schema and question tokens from direct loss. Grounded checkpoints store the tokenizer, target format, and grounding settings, so the ordinary inference command handles slot conversion, deterministic SQL rendering, and name restoration automatically.

### Semantic-plan natural-language pilot

The v10 natural-language pilot predicts a compact query plan and uses the typed
renderer to produce SQL. Its training mix contains 20,000 records: 45% focused
natural-language contrasts, 32.5% composition examples, 17.5% replay, and 2.5%
from each external source. Rebuild and train it with:

```bash
python -m pocketsql.data.semantic \
  --output data/semantic-v10-natural-v2 \
  --schemas 1000 \
  --gate-schemas 100 \
  --seed 20260830

python -m pocketsql.training.audit \
  --data data/semantic-v10-natural-v2/mixed_train.jsonl \
  --config configs/base_semantic_v10_natural_v2.yaml

python -m pocketsql.training.train \
  --config configs/base_semantic_v10_natural_v2.yaml \
  --epochs 4 \
  --data data/semantic-v10-natural-v2/mixed_train.jsonl \
  --val-data data/semantic-v10-natural-v2/mixed_validation.jsonl \
  --checkpoint checkpoints/base-semantic-v10-natural-v2 \
  --initialize-from checkpoints/base-semantic-v10-pilot-best-execution
```

The selected checkpoint is
`checkpoints/base-semantic-v10-natural-v2-best-execution`. It reaches 87.0%
execution accuracy with 99.75% valid SQL on 1,200 close unseen paraphrases and
76.5% execution with 98.42% valid SQL on the 1,200-example focused test. On the
older far-paraphrase gate it reaches 20.92%, up from the first semantic pilot's
7.0% but still far from general language understanding. The schema-grounded
decoder rejects unmentioned literal values, removes redundant or unrequested
projections, and resolves unambiguous adjacent parent/child noun phrases.

The v11 composition extension adds joined `SUM`, `AVG`, `MAX`, and `COUNT`
families with parent-only and cross-table filters. Generate and train it with:

```bash
python -m pocketsql.data.semantic \
  --output data/semantic-v11-composed \
  --schemas 1000 \
  --gate-schemas 100 \
  --seed 20260831 \
  --include-joined-aggregates

python -m pocketsql.training.train \
  --config configs/base_semantic_v11_composed.yaml \
  --data data/semantic-v11-composed/mixed_train.jsonl \
  --val-data data/semantic-v11-composed/mixed_validation.jsonl \
  --checkpoint checkpoints/base-semantic-v11-composed \
  --initialize-from checkpoints/base-semantic-v10-natural-v2-best-execution
```

Use `checkpoints/base-semantic-v11-composed-best-execution` for the composed
natural-language pilot. On the frozen 400-example joined-aggregate gate it
reaches 64.75% execution, compared with 19.0% for v10. Joined `SUM` improves
from 36% to 96%, `MAX` from 40% to 96%, `AVG` from 0% to 43%, and two-filter
`COUNT` from 0% to 24%. On the original v10 gates, close-paraphrase execution
improves from 87.0% to 93.08%, focused execution from 76.5% to 78.08%, and the
far-paraphrase gate from 20.92% to 21.83%. The complete 16-family v11 natural
gate scores 85.06%; its focused test scores 73.75% because it includes the four
new harder families.

Permuted schema grounding now hashes a normalized DDL fingerprint and sends a
normalized schema prompt to the model. Equivalent single-line and pretty-
printed schemas therefore receive identical internal slots; whitespace no
longer changes the generated query.

## Frozen human benchmark

`PocketSQL Human Alpha v1` is an evaluation-only derivative of the official
Spider 1.0 test split. Its importer pins the official archive checksum, keeps
every human-authored example that fits the current PocketSQL SQL and sequence
contract, executes the original and normalized gold queries against the source
SQLite database, scans every local training split for exact record overlap, and
freezes the accepted IDs before model inference. Never use this benchmark for
training, checkpoint selection, interpolation, prompt tuning, or data
generation.

Build the gate with:

```bash
python -m pocketsql.data.spider \
  --output data/spider-human-alpha-v1 \
  --compatibility-config configs/base_semantic_v11_composed.yaml
```

The frozen v1 gate contains 499 questions across 30 unseen databases, selected
without sampling from 2,147 official test questions. Its gold replay is 100%
executable and 100% execution-equivalent. Evaluate a checkpoint while saving
replayable outputs and row-level diagnostics with:

```bash
python -m pocketsql.evaluation.evaluate \
  --data data/spider-human-alpha-v1/benchmark.jsonl \
  --checkpoint checkpoints/base-semantic-v11-composed-best-execution \
  --batch-size 16 \
  --output data/spider-human-alpha-v1/results/v11/report.json \
  --prediction-output data/spider-human-alpha-v1/results/v11/predictions.jsonl \
  --diagnostics data/spider-human-alpha-v1/results/v11/diagnostics.jsonl
```

V11 reaches 8.02% execution accuracy (40/499), 6.81% normalized exact and
semantic-plan match, and 31.46% valid executable output. Conditional on a valid
output, execution accuracy is 25.48%. On the closer two-table portion it reaches
14.45% execution, versus 5.85% for three-table schemas and 3.23% across schemas
with four or more tables. This is the first genuinely human, unseen-schema
measurement and supersedes synthetic validation scores as the alpha-readiness
gate; the current model does not pass it yet.

### V12 human-language fine-tune

V12 adds grammar-constrained semantic-plan decoding and a Spider curriculum
that is strictly separate from the frozen test gate. It retains 1,254 supported
questions from the official Spider train split across 85 schemas and 185
questions from official dev across 12 different validation schemas. The
20,000-example training curriculum is 30% human and 70% v11 replay; validation
is balanced 50/50. Build and train it with:

```bash
python -m pocketsql.data.spider_training \
  --output data/spider-human-train-v1 \
  --compatibility-config configs/base_semantic_v11_composed.yaml

python -m pocketsql.training.train \
  --config configs/base_semantic_v12_human.yaml \
  --data data/spider-human-train-v1/mixed_train.jsonl \
  --val-data data/spider-human-train-v1/mixed_validation.jsonl \
  --checkpoint checkpoints/base-semantic-v12-human \
  --initialize-from checkpoints/base-semantic-v11-composed-best-execution
```

Use `checkpoints/base-semantic-v12-human-best-execution` for human-language
experiments. On the unchanged frozen human gate:

| Model / decoding | Valid output | Exact match | Execution |
| --- | ---: | ---: | ---: |
| original v11 | 31.46% | 6.81% | 8.02% (40/499) |
| v11 + constrained plan decoding | 67.94% | 6.81% | 10.22% (51/499) |
| v12 + constrained plan decoding | 60.52% | 16.43% | 20.64% (103/499) |
| v12 + explicit schema grounding | 64.33% | 22.65% | 24.25% (121/499) |
| v12 + scoped foreign-key planner | 68.74% | 22.65% | 24.85% (124/499) |

On unseen Spider dev, v12 improves execution from 10.81% to 20.54% relative
to v11 under the same decoder. It also preserves the constrained-decoder
synthetic gates: 78.81% versus 78.69% on the 1,600-example natural gate and
59.25% versus 58.25% on the joined-aggregate gate. Original unconstrained v11
still scores 85.06% and 64.75% on those synthetic fixtures, showing that the
remaining tradeoff comes from constrained greedy decoding rather than the
human-data fine-tune. Pass `--unconstrained-semantic-plan` to inference or
evaluation only when reproducing those legacy scores.

V12 is a meaningful generalization improvement, but 24.85% human execution is
not alpha-ready. Do not tune another checkpoint against the now-observed frozen
test results; use Spider dev diagnostics or create a new unseen benchmark gate.

### Schema-linking follow-up

The inference compiler now binds direct grounded pairs such as `column4 is
value0`, `value0 column4`, and `column4 value0 or value1`. It also trusts a
single explicitly mentioned table over a contradictory decoded base table.
These constraints change neither the checkpoint nor ambiguous requests; they
only override a model decision when the grounded question states the pointer
pair directly. On Spider dev this raises v12 execution from 20.54% (38/185) to
24.32% (45/185). A scoped foreign-key planner then reaches 24.86% (46/185):
for non-aggregate/non-group queries it repairs or adds a two-table join only
when the requested tables or grounded projection identify one unique declared
foreign-key relationship. Ambiguous and disconnected relationships are not
guessed. Frozen join execution doubles from 2/70 to 4/70, while the natural and
joined-aggregate synthetic gates remain at 85.06% and 66.00%, respectively.

Two v13 training experiments were rejected. A readable slot-label prompt
changed the representation too aggressively and reached only 8.65% Spider-dev
execution. A 24,000-example hard-family curriculum reached 21.62% on Spider
dev, but regressed to 17.03% on the already-observed frozen gate and slightly
reduced both synthetic gates. Consequently, the recommended model remains the
v12 checkpoint with the current schema-grounding compiler at that stage; the
factorized v14 experiment below now supersedes it.

#### Schema-linking oracle

The component oracle replaces only table, join-path, and role-specific column
references in the raw decoded plan. It preserves the model's aggregate
function, projection/filter arity, filter operators and values, Boolean
connector, grouping/order presence, sort direction, and limit. Run it with:

```bash
python -m pocketsql.evaluation.schema_oracle \
  --data data/schema-linking-v13/human_validation.jsonl \
  --checkpoint checkpoints/base-semantic-v12-human-best-execution \
  --batch-size 16 \
  --output-dir data/schema-linking-v13/results/v12-schema-oracle
```

On all 185 unseen Spider-dev examples, every raw target parses. Perfect schema
links raise valid output from 69.73% to 95.68% and execution from 24.86%
(46/185) to 42.70% (79/185): a gain of 17.84 percentage points and 33 newly
correct queries without losing a previously correct query. The all-links
ablation shows table/join selection is the largest schema component:

| Schema information supplied | Execution | Correct |
| --- | ---: | ---: |
| none (production pipeline) | 24.86% | 46/185 |
| tables and join path only | 31.89% | 59/185 |
| all except tables/join | 32.43% | 60/185 |
| all except projection | 36.76% | 68/185 |
| all except group/order columns | 37.84% | 70/185 |
| all except filter columns | 39.46% | 73/185 |
| all schema links | 42.70% | 79/185 |

This also establishes that a schema linker alone cannot make the model ready:
only 35.14% of raw plans match every non-schema operation decision. Selection
arity matches 72.43%, filter operators plus values 69.19%, and group arity
82.70%. The next architecture should therefore factor the typed plan into a
table/join selector, role-specific schema pointers, and separate operation/
composition heads, starting with table/join selection but training and scoring
the operation heads independently. The full report and row-level oracle SQL are
stored under `data/schema-linking-v13/results/v12-schema-oracle/`.

### V14 factorized schema architecture

V14 keeps the semantic-plan language decoder for operations and arity, but adds
separate supervised heads for the base table, optional joined table, join keys,
projection columns, aggregate target, filter columns, grouping columns, and
ordering column. Decoding scores only physical table-column pairs and declared
foreign-key relationships. The production mode is conservative: a valid
language-decoded plan is retained, and the factorized pointers are used only to
rescue a schema-invalid plan. This prevents a weak pointer from overwriting an
already valid answer.

Train the selected three-epoch model from v12 with:

```bash
python -m pocketsql.training.train \
  --config configs/base_semantic_v14_factorized.yaml \
  --data data/spider-human-train-v1/mixed_train.jsonl \
  --val-data data/spider-human-train-v1/mixed_validation.jsonl \
  --checkpoint checkpoints/base-semantic-v14-factorized \
  --initialize-from checkpoints/base-semantic-v12-human-best-execution
```

Use `checkpoints/base-semantic-v14-factorized-best-execution`. At epoch 3 it
reaches 47.03% execution on the 640-example mixed validation set versus 45.94%
for v12. Its complete active-pointer bundle is correct on 31.25% of examples;
individual validation accuracy is 78.91% for the base table, 85.62% for join
presence/table, 67.18% for projection columns, 73.05% for aggregate columns,
and 61.18% for filter columns.

| Gate | v12 execution | v14 hybrid execution | v12 valid | v14 valid |
| --- | ---: | ---: | ---: | ---: |
| unseen Spider dev (185) | 24.86% (46) | **29.73% (55)** | 69.73% | **87.57%** |
| frozen human benchmark (499) | 24.85% (124) | **24.85% (124)** | 68.74% | **87.17%** |
| natural synthetic gate (1,600) | 85.06% (1,361) | **85.31% (1,365)** | 96.38% | **99.25%** |
| joined-aggregate gate (400) | 66.00% (264) | **66.50% (266)** | 90.00% | **97.25%** |

This is a real architectural improvement, but 29.73% execution on unseen human
questions is still not alpha-ready. Join-key and grouping-column pointers are
the weakest heads, and valid-but-wrong plans now dominate failures. The next
work should improve operation composition and those pointer heads rather than
adding more deterministic repair rules.

The frozen paired anti-memorization gate makes the remaining language gap more
explicit. It contains 192 query pairs on 24 schemas: 99.56% of identifiers are
absent from training, no schema or question is duplicated from training, and
the paraphrases reduce schema-token overlap from 51.73% to 2.51% while keeping
the gold SQL unchanged.

| Anti-memorization slice | v12 execution | v14 execution | v14 exact/plan match |
| --- | ---: | ---: | ---: |
| direct identifier wording (192) | 52.08% | **53.13%** | 37.50% |
| semantic paraphrase (192) | 7.81% | **8.33%** | 8.33% |
| direct + held-out composition (96) | 39.58% | 39.58% | 22.92% |
| paraphrase + held-out composition (96) | 0.00% | 0.00% | 0.00% |

V14 raises overall valid output from 82.55% to 98.96%, but overall execution
only from 29.95% to 30.73%. This confirms that the architecture mostly improves
safe schema-grounded construction. It does not yet provide robust synonym
linking or combine an unfamiliar paraphrase with an unseen SQL composition.

### V15 semantic schema-linking experiment

V15 tests whether readable schema labels and paired paraphrases can reduce the
model's dependence on literal identifier words. The language decoder still
receives the original canonical schema prompt. Only the factorized schema heads
receive an auxiliary `SCHEMA LINKS` legend containing readable table and column
labels, so the experiment does not change the decoder's target representation.
The new dataset contains 7,200 paired direct/paraphrased examples on 400
training schemas, 360 paraphrase-only examples on 40 disjoint validation
schemas, and 7,200 replay examples from the v14 training mixture. The frozen
anti-memorization vocabulary and schemas remain evaluation-only.

A schema-head-only phase was rejected: its best paraphrase execution was 9.44%,
below v14's 11.11%. Two epochs of joint training produced
`checkpoints/base-semantic-v15-joint-best-execution`. The raw `replace` pointer
policy also failed because its 6.11% complete-pointer accuracy overwrote many
valid language-decoder plans. Use the conservative fallback policy when
examining this research checkpoint:

```bash
python -m pocketsql.inference \
  --checkpoint checkpoints/base-semantic-v15-joint-best-execution \
  --factorized-schema-mode fallback \
  --schema "$SCHEMA" \
  --question "$QUESTION"
```

The fallback policy keeps a valid decoder plan and invokes factorized pointers
only to rescue an invalid one. It produces substantial transfer within the
synthetic language distribution, including on the anti-memorization gate that
was first evaluated only after checkpoint selection:

| Gate | v14 execution | v15 fallback execution |
| --- | ---: | ---: |
| paired direct wording (360) | 58.33% | **94.44%** |
| paired semantic paraphrase (360) | 11.11% | **27.78%** |
| frozen anti direct wording (192) | 53.13% | **83.33%** |
| frozen anti semantic paraphrase (192) | 8.33% | **14.58%** |
| anti direct + held-out composition (96) | 39.58% | **67.71%** |
| anti paraphrase + held-out composition (96) | 0.00% | 0.00% |
| unseen Spider dev (185) | **29.73%** | 22.16% |
| frozen human benchmark (499) | **24.85%** | 24.65% |
| joined-aggregate gate (400) | **66.50%** | 56.00% |

This is evidence of useful but narrow generalization, not a new production
winner. V15 maps several unseen paraphrases and identifier vocabularies better,
but it still cannot combine an unfamiliar paraphrase with a held-out operation
composition, and it regresses 7.57 points on unseen Spider questions. V14
therefore remains the recommended broad checkpoint. The next experiment should
retain more diverse human replay and learn a confidence-calibrated residual
schema linker instead of allowing pointers to replace complete plans.

### V16 human replay and confidence-gated residuals

V16 increases human replay from roughly 2,200 examples in v15's sampled replay
to 6,000 examples. Its 20,000-record training mixture also contains all 7,200
paired semantic examples and 6,800 broad synthetic replays. Checkpoint
selection uses a separate 1,000-record mixture: 320 resampled Spider-dev
questions, 360 paraphrase-only examples on disjoint synthetic schemas, and 320
general synthetic questions. The anti-memorization data is not loaded by the
v16 mixture builder and remains a final evaluation-only gate.

Build and train the experiment with:

```bash
python -m pocketsql.data.semantic_linking_v16 \
  --output data/semantic-linking-v16

python -m pocketsql.training.train \
  --config configs/base_semantic_v16_residual.yaml \
  --data data/semantic-linking-v16/mixed_train.jsonl \
  --val-data data/semantic-linking-v16/mixed_validation.jsonl \
  --checkpoint checkpoints/base-semantic-v16-residual \
  --initialize-from checkpoints/base-semantic-v15-joint-best-execution
```

The residual linker normalizes pointer scores only across physically valid
table/join and owner/column candidates. It can retain low-confidence roles from
the decoder and always returns a schema-valid decoder plan before attempting a
pointer rescue. Training raises complete active-pointer accuracy from 6.11% to
25.80%. Nevertheless, confidence thresholds of 0.7 and 0.9 reach only 39.3%
and 39.5% on the mixed selection set. Ordinary fallback reaches 40.3%, so the
confidence policy is rejected and the v16 checkpoint must be run with fallback:

```bash
python -m pocketsql.inference \
  --checkpoint checkpoints/base-semantic-v16-residual-best-execution \
  --factorized-schema-mode fallback \
  --schema "$SCHEMA" \
  --question "$QUESTION"
```

| Gate | v14 | v15 fallback | v16 fallback |
| --- | ---: | ---: | ---: |
| mixed selection set (1,000) | 36.40% | 39.20% | **40.30%** |
| paired direct wording (360) | 58.33% | 94.44% | **96.94%** |
| paired semantic paraphrase (360) | 11.11% | 27.78% | **28.06%** |
| unseen Spider dev (185) | **29.73%** | 22.16% | 24.86% |
| frozen human benchmark (499) | 24.85% | 24.65% | **25.25%** |
| joined-aggregate gate (400) | **66.50%** | 56.00% | 54.00% |
| frozen anti-memorization gate (384) | 30.73% | 48.96% | **49.74%** |

On the frozen anti gate, v16 reaches 84.90% on direct wording, 14.58% on
semantic paraphrases, 70.83% on direct wording plus a held-out composition, and
still 0% when both the paraphrase and composition are held out. The human-heavy
replay recovers part of v15's Spider regression and improves the frozen human
benchmark by two correct queries over v14, but it remains nine queries behind
v14 on unseen Spider dev and loses 50 joined-aggregate queries. V16 is therefore
an optional semantic-language specialist; v14 remains the recommended broad
checkpoint. The next attempt should add genuinely diverse paraphrase/composition
pairs and use a multi-gate selection objective instead of increasing epochs.

### V17 dataset expansion

V17 expands language and composition coverage before doing any further model
scaling. The Spider curator now conservatively recognizes SQLite's legacy
double-quoted filter values, but only when the value cannot name a schema
identifier; the original and normalized queries must still execute identically.
Using the 416-token v16 contract raises the usable, leakage-free Spider train
split from 1,254 to 2,086 unique questions and its validation split from 185 to
257. Thirteen train rows that duplicated a normalized question+SQL pair in the
frozen human benchmark, plus two train/validation overlaps, are explicitly
removed.

The new synthetic portion contains 16,200 questions on 300 training schemas,
split evenly across identifier-explicit, semantic-paraphrase, and terse request
styles. It includes 4,500 examples from the four compositions that the earlier
anti-memorization experiment withheld: joined aggregates with multiple filters,
joined counts with multiple filters, filtered grouping, and
distinct/filter/limit. A separate 864-question gate uses 24 schemas from three
new domains and has zero schema, identifier, or exact-question overlap with
training. `fresh_gate.jsonl` is evaluation-only and must not be used for
checkpoint selection.

Build and audit the 30,000-record mixture with:

```bash
python -m pocketsql.data.spider_training \
  --output data/spider-human-v17-recovered \
  --compatibility-config configs/base_semantic_v16_residual.yaml

python -m pocketsql.data.semantic_expansion_v17 generate \
  --output data/semantic-expansion-v17 \
  --train-schemas 300 \
  --validation-schemas 36 \
  --gate-schemas 24

python -m pocketsql.data.semantic_expansion_v17 mix \
  --output data/semantic-expansion-v17-mixture

python -m pocketsql.training.audit \
  --config configs/base_semantic_v17_expanded.yaml \
  --data data/semantic-expansion-v17-mixture/mixed_train.jsonl
```

The final mix contains 16,200 diverse semantic/composition questions, 8,000
balanced replays from 2,086 unique human Spider questions, and 5,800 broad
synthetic replays. Its 1,548-record validation set keeps disjoint synthetic and
Spider schemas. All sequences fit the 416-token context, and exact pair checks
find no overlap from training into validation, the fresh gate, or the frozen
human benchmark.

Training has not been run on v17 yet. When ready, initialize the experiment
from v16 and keep fallback decoding for checkpoint selection:

```bash
python -m pocketsql.training.train \
  --config configs/base_semantic_v17_expanded.yaml \
  --data data/semantic-expansion-v17-mixture/mixed_train.jsonl \
  --val-data data/semantic-expansion-v17-mixture/mixed_validation.jsonl \
  --checkpoint checkpoints/base-semantic-v17-expanded \
  --initialize-from checkpoints/base-semantic-v16-residual-best-execution \
  --best-execution-checkpoint checkpoints/base-semantic-v17-expanded-best-execution \
  --log-dir runs/base-semantic-v17-expanded
```

## Inference and evaluation

```bash
python -m pocketsql.inference --checkpoint checkpoints/base-semantic-v12-human-best-execution --schema 'CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, note TEXT); CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), total REAL, status TEXT, remarks TEXT);' --question 'show me customer orders where name is max and city is dallas'
python -m pocketsql.evaluation.evaluate --data data/generated-position-robust-v8/test.jsonl --checkpoint checkpoints/base-external-augmented-best --batch-size 16
```

`pocketsql.inference.generate_sql(model, schema, question)` stops at `</sql>` or `<eos>`, extracts only SQL, and rejects multi-statement or non-`SELECT` output. It grounds direct column/value pairs, repairs a uniquely declared two-table foreign-key relationship where the supported planner applies, then asks SQLite to resolve every generated table and column against the supplied schema. Output such as `SELECT city FROM orders` is rejected when `city` belongs only to `customers`; ambiguous or disconnected join repairs are rejected rather than guessed. A checkpoint produced only via `--overfit` on a handful of examples (as in the training commands above) has not learned to generalize, so it commonly raises `model output is not a single read-only SELECT` on new schemas/questions — that is expected, not a bug; the error message includes the raw decoded text for debugging. Train longer on the full split for usable generations.

`pocketsql.evaluation.evaluate` accepts either `--checkpoint <dir>` (generates predictions itself, one per record in `--data`) or `--predictions <file>` (a pre-existing file with one SQL statement, or one `{"sql": ...}` JSON object, per line matching `--data`'s record order). It reports syntax validity, executability, normalized exact match, execution accuracy, and per-family/difficulty metrics. SQLite evaluation is query-only and caps returned rows.

## Tests

```bash
pytest
```

## Design and current limits

Synthetic questions and SQL are both rendered from immutable typed `QueryPlan` objects. The MVP supports projection, `DISTINCT`, compatible filters with `AND`/`OR`, aggregates, grouping, ordering/limit, and declared two-table joins. It intentionally excludes subqueries, CTEs, window functions, mutations, and joins beyond two tables. The verbalizer is deliberately template-driven.

For human-language experiments, the recommended checkpoint is
`checkpoints/base-semantic-v14-factorized-best-execution`. The v12 checkpoint
remains the autoregressive comparison baseline. The earlier
`checkpoints/base-external-augmented-best` is the legacy direct-SQL baseline,
with `checkpoints/base-position-robust-v8-best` and
`checkpoints/base-gretel-augmented-best` retained for comparison.

V8 reaches 97.89% execution accuracy (9,789/10,000) on its schema-disjoint test,
94.52% (8,507/9,000) on the identifier/composition challenge, 77.43%
(4,646/6,000) on the held-out casual-language set, and 99.77% (1,316/1,319) on
the all-column gate. Relative to v7, those changes are +5.25, +8.01, +3.41,
and +0.83 percentage points respectively. The new filtered-composition families
score between 96.2% and 99.0% on the v8 test, while ordinary joins reach 96.67%.

The selected combined checkpoint was evaluated on every full gate, not just the
training validation sample. The final column compares it with the previously
recommended Gretel-only interpolation:

| Gate | V8 | Gretel augmented | Combined | Change |
| --- | ---: | ---: | ---: | ---: |
| frozen WikiSQL external gate | 0.60% | 6.00% | 8.00% | +2.00 pp |
| frozen Gretel external gate | 1.60% | 11.90% | 11.20% | -0.70 pp |
| v8 schema-disjoint test | 97.89% | 98.53% | 98.95% | +0.42 pp |
| identifier/composition challenge | 94.52% | 96.92% | 98.21% | +1.29 pp |
| held-out casual language | 77.43% | 79.93% | 81.82% | +1.88 pp |
| all-column identifier copy | 99.77% | 100.00% | 100.00% | 0.00 pp |
| fully held-out v9 compositions | 27.29% | 37.29% | 40.52% | +3.23 pp |
| v9 counterfactual records | 49.25% | 55.50% | 54.50% | -1.00 pp |
| complete counterfactual pairs | 35.50% | 40.33% | 39.33% | -1.00 pp |

WikiSQL was worth adding as a complementary source: the balanced checkpoint
improves the main regression, identifier, casual-language, and unseen-
composition gates while increasing WikiSQL transfer. It gives back 0.70 points
on Gretel and 1.00 point on the counterfactual gates, so the earlier Gretel-only
checkpoint remains useful when those are the priority. Absolute external
accuracy is still only 8.00% on WikiSQL and 11.20% on Gretel; this small model is
not a general SQL system. The importers intentionally discard unsupported or
ungrounded examples because PocketSQL still excludes subqueries, CTEs, window
functions, multi-join queries, mixed Boolean trees, and many dialect-specific
constructs.

The v9 rebuild is useful but remains experimental. On identical v9-specific
evaluation data, the corrected v9 pilot compares with the selected v8 model as
follows:

| Gate | V8 | v9 pilot | Change |
| --- | ---: | ---: | ---: |
| v9 schema-disjoint test | 47.32% | 85.28% | +37.96 pp |
| counterfactual records | 49.25% | 91.00% | +41.75 pp |
| complete counterfactual pairs | 35.50% | 88.33% | +52.83 pp |
| fully held-out compositions | 27.29% | 30.42% | +3.13 pp |

This means the rebuilt data strongly improves sensitivity to the requested
column, value, operator, and aggregate, but the model still does not reliably
assemble an entirely unseen combination of otherwise familiar operations. The
pilot was therefore stopped at its composition gate instead of launching a
costly full-size v9 run. Its standalone checkpoint remains rejected; the
combined external-data interpolation above is the recommended checkpoint. The
next high-value experiment is a compact structured query-plan target followed
by a deterministic SQL renderer; it can separate understanding the requested
operations from learning SQL token order.

These numbers do not mean arbitrary text-to-SQL is solved. The model is still
trained on synthetic templates and supports a deliberately limited SQL subset.
Held-out casual phrasing is the clearest remaining weakness: joins execute at
31.83% and `DISTINCT` at 59.44% on that deliberately unseen-language set.
Future model selection should keep the position-copy, casual-language, and
identifier challenge sets as independent gates, without training on any of
them.

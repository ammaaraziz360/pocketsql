# V17 final frozen-gate comparison

This comparison evaluates the selected v14, v16, and v17 checkpoints with
`factorized_schema_mode=fallback`. Every model generated predictions for the
same 2,867 records (8,601 predictions total).

## Execution accuracy

| Evaluation slice | Records | v14 | v16 | v17 |
| --- | ---: | ---: | ---: | ---: |
| complete battery | 2,867 | 28.36% (813) | **36.83% (1,056)** | 34.15% (979) |
| fresh v17 gate | 864 | 6.37% (55) | 8.45% (73) | **9.61% (83)** |
| frozen human benchmark | 499 | 24.85% (124) | **25.25% (126)** | 25.05% (125) |
| anti-memorization | 384 | 30.73% (118) | **49.74% (191)** | 43.49% (167) |
| joined aggregate | 400 | **66.50% (266)** | 54.00% (216) | 54.25% (217) |
| paired v15 | 720 | 34.72% (250) | **62.50% (450)** | 53.75% (387) |

## Language slices

| Evaluation slice | Records | v14 | v16 | v17 |
| --- | ---: | ---: | ---: | ---: |
| fresh direct identifiers | 432 | 11.57% | 13.66% | **15.28%** |
| fresh semantic paraphrases | 432 | 1.16% | 3.24% | **3.94%** |
| paired direct identifiers | 360 | 58.33% | **96.94%** | 87.50% |
| paired semantic paraphrases | 360 | 11.11% | **28.06%** | 20.00% |
| anti direct identifiers | 192 | 53.13% | **84.90%** | 76.04% |
| anti semantic paraphrases | 192 | 8.33% | **14.58%** | 10.94% |

## Fresh-gate semantic components

| Component | v14 | v16 | v17 |
| --- | ---: | ---: | ---: |
| tables | 48.26% | 47.57% | **61.57%** |
| projection | 24.42% | 22.92% | **31.94%** |
| filters | 28.01% | 27.89% | **32.18%** |
| aggregate | 65.51% | 63.89% | **70.14%** |
| join | 56.02% | 55.21% | **64.35%** |
| complete semantic plan | 5.56% | **8.45%** | **8.45%** |

## Decision

V17 is not promoted. It improves the fresh gate by ten executions over v16 and
substantially improves individual schema-grounding components, so the expanded
data was useful as a diagnostic and a small learning signal. It does not improve
the frozen human benchmark, and it regresses by 63 paired-gate executions and
24 anti-memorization executions. Nine of eighteen fresh v17 intent families
still score 0% execution.

The bottleneck is assembling a correct plan from partially correct choices, not
producing syntactically valid SQL. The next architecture should separate:

1. an operation skeleton (select/count/sum, joins, filter count, grouping,
   ordering, and limit),
2. question-to-schema linking for every table and column role, and
3. literal copying from the question.

Use v16 as the semantic-linking baseline and v14 as the joined-aggregate
regression baseline. Do not select checkpoints on the fresh or frozen gates.

The source hashes are recorded in `manifest.json`; complete reports,
predictions, and row-level diagnostics are stored beside this file.

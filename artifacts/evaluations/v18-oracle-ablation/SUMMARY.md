# V16 development oracle ablation

The ablation uses only development data. It does not evaluate or tune against
the frozen v17, human release, anti-memorization, paired-v15, or joined-aggregate
gates.

## Inputs

| Slice | Source | Records |
| --- | --- | ---: |
| semantic paraphrases | `data/semantic-expansion-v17/validation.jsonl` | 648 |
| human Spider | `data/spider-human-v17-recovered/human_validation.jsonl` | 237 |

Checkpoint: `checkpoints/base-semantic-v16-residual-best-execution`

## Isolated rescue ceilings

Oracle plans bypass production grounding heuristics, so isolated corrections
are compared with the raw decoded plan. The rescue ceiling retains a correction
only when it turns an incorrect execution into a correct one.

| Slice | Production pipeline | Raw plan | Schema links | Operation structure | Filter literals |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic (648) | 6.33% (41) | 5.25% (34) | **32.25% (209)** | 10.34% (67) | 5.25% (34) |
| human (237) | 19.41% (46) | 12.66% (30) | **32.07% (76)** | 18.14% (43) | 13.50% (32) |

The operation intervention uses the existing factorized schema-role outputs so
that only query shape is corrected. It is eligible for 506/648 semantic records
and 192/237 human records; the remaining decoded plans do not provide enough
filter values to construct the gold filter arity without also applying a
literal oracle.

## Combined controls

| Slice | Gold schema + decoded operations + gold literals | Gold schema + gold operations + decoded literals | Fully gold plan |
| --- | ---: | ---: | ---: |
| semantic (648) | 54.01% (350) | 49.07% (318) | 100.00% (648) |
| human (237) | 41.35% (98) | 67.51% (160) | 100.00% (237) |

## Interpretation

Schema linking is the dominant isolated bottleneck. Fixing schema references
rescues 175 semantic examples and 46 human examples over the raw plans. Fixing
operation structure rescues 33 and 13 respectively; literal correction alone
rescues zero semantic and two human examples because a correct value cannot
help when its table or column is wrong.

The combined controls show strong interactions. After schema is correct,
operation mistakes still cap the human slice at 41.35%, while decoded literals
cap a perfect schema-and-operation plan at 67.51%. On semantic paraphrases,
operations and literal selection are roughly equally limiting after schema is
fixed. All three architectural changes are therefore justified.

## V18 implementation order

1. Replace pooled prompt classification with per-question-token to
   per-schema-element cross-attention pointers. Encode ownership, data type,
   primary keys, and foreign-key edges with each schema candidate.
2. Predict the operation skeleton with explicit heads for projection arity,
   aggregate, join presence, filter count/operators/connector, grouping,
   ordering, direction, and limit.
3. Select each filter value from question spans or typed value candidates and
   copy the exact source text into the deterministic compiler.
4. Train the three heads jointly with hard schema negatives, while retaining
   v14/v16 replay gates for regression checks.

Full reports and row-level diagnostics are in `v16-semantic-dev/` and
`v16-human-dev/` beside this file.

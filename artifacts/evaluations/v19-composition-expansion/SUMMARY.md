# V19 compositional dataset expansion

## Purpose

V18 improved individual filter and projection pointers, but its dedicated
schema-disjoint gate remained at 0% execution. V19 therefore trains complete
plans instead of adding more isolated filter-column examples.

## Dataset

- 14,400 training examples across 300 schemas.
- 1,728 validation examples across 36 schema-disjoint schemas.
- 1,152 fresh-gate examples across 24 lexically isolated schemas.
- 48 coordinated query plans per schema.
- 6,480 joined examples and 3,960 multi-filter examples across all splits.
- 9,000 multi-operation examples, 6,480 paired-operation examples, and 1,800
  atomic controls.
- Projection, comparison operator, Boolean connector, aggregate, grouping,
  ordering, limit, and join compositions are represented.

All 17,280 new records execute and return rows. All 2,520 counterfactual groups
produce distinct result sets, so a wrong operator or aggregate cannot receive
execution credit by coincidence. Every one of the 36,720 active schema-role
mentions is explicit after identifier canonicalization. Train, validation, and
fresh-gate schemas are disjoint, with zero exact question overlap.

The replay mixture contains all 14,400 new records plus 28,800 ordinary V17
records, giving a 1:2 composition-to-replay ratio. Four contradictory replay
duplicates are excluded, leaving zero conflicting schema/question labels.

## Training readiness

The 43,200-record replay mixture passes the V19 tokenizer audit with no
truncated targets and no targets over the 96-token generation cap. Dedicated
validation and fresh-gate records also pass. The full test suite contains 163
passing tests.

## Training result

V19 was initialized from
`checkpoints/base-semantic-v18-filter-calibrated-best-execution` and trained for
one epoch over the 43,200-record replay mixture. The selected experimental
checkpoint is `checkpoints/base-semantic-v19-composition-best-execution`.

| Gate | V18 calibrated | V19 composition | Change |
| --- | ---: | ---: | ---: |
| unchanged mixed validation | 11.82% | 10.66% | -1.16 pp |
| V19 schema-disjoint composition | 31.60% | 40.62% | +9.03 pp |
| V19 lexically fresh composition | 14.76% | 26.22% | +11.46 pp |

On the mixed validation tracks, human execution declines from 9.60% to 8.60%,
semantic paraphrases improve from 1.39% to 1.70%, and synthetic execution
declines from 31.50% to 27.75%. Complete schema pointers improve from 18.35% to
22.74%, complete structured heads from 6.01% to 6.98%, and literal start/end
accuracy from 76.47%/68.82% to 78.75%/73.09%.

The expansion therefore achieved its intended composition transfer, including
large gains on joined filters, ordering with limits, and `DISTINCT`. It is not
promoted because the unchanged mixed benchmark regressed and complete
counterfactual-group accuracy remains weak, especially on the fresh gate.

## Interpolation sweep

Inference-only weight blends were evaluated between the V18 calibrated model
and the full V19 adaptation. The V19 weight was swept across 10%, 15%, 20%,
25%, 30%, 35%, 40%, 50%, and 75%. The best unchanged-mixed result is the 25%
blend at `checkpoints/base-semantic-v19-composition-blend-25`.

| Gate | V18 calibrated | 25% V19 blend | Full V19 |
| --- | ---: | ---: | ---: |
| unchanged mixed validation | 11.82% | 11.63% | 10.66% |
| V19 schema-disjoint composition | 31.60% | 33.68% | 40.62% |
| V19 lexically fresh composition | 14.76% | 18.58% | 26.22% |

The blend gives up only three correct queries on the 1,548-example mixed set
while retaining +2.08 points on schema-disjoint composition and +3.82 points
on the fresh gate. Mixed executability improves from 70.67% to 71.96% and
human execution remains 9.60%, but the strict no-regression promotion rule is
not satisfied. Counterfactual-group completion also does not improve, so the
blend remains an experimental tradeoff rather than the new default.

## Memory correction

The first batch-16 run exhausted unified memory. MLX evaluates lazily, and the
old accumulation path retained every micro-batch graph until the optimizer
step, so lowering the visible batch size did not actually lower peak memory.
`accumulated_train_step` now evaluates each gradient tree eagerly. The V19
configuration uses batch size 4 with four-way accumulation, an 8 GiB MLX memory
ceiling, and a 256 MiB cache limit. A low-memory smoke test and the full epoch
both completed successfully on the 16 GB Mac.

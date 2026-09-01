# V18 contrastive filter-linking experiment

## Question

Would more targeted data fix V18's 15.81% filter-column accuracy, or is the
structured architecture unable to learn the mapping?

## Data and controls

- 6,300 contrastive training examples from 300 schemas.
- 504 schema-disjoint validation examples from 36 schemas.
- 2,916 complete counterfactual groups in total.
- Every group holds schema, projection, operator, and literal constant while
  changing only the filter column and its phrase in the question.
- Every gold filter column remains explicit after identifier canonicalization.
- Text and numeric filter columns are both represented.
- A 1:2 mixture adds 12,600 ordinary replay examples to prevent schema-head
  forgetting.

The unadapted V18 checkpoint scored 10.12% filter-column accuracy and 0%
execution on the dedicated schema-disjoint contrastive gate. A 64-example
overfit control reached 100% filter-column accuracy, proving the head has enough
capacity to represent the task.

## Mixed-validation results

| Checkpoint | Execution | Filter column | Projection column | Complete pointers | Complete structured |
| --- | ---: | ---: | ---: | ---: | ---: |
| V18 joint baseline | 11.24% | 15.81% | 31.55% | 5.94% | 2.45% |
| target-only adaptation | 7.56% | 24.34% | 27.39% | 7.95% | 2.91% |
| contrast + ordinary replay | 10.27% | **51.10%** | **43.64%** | **18.35%** | 5.30% |
| replay + operation/literal calibration | **11.82%** | **51.10%** | **43.64%** | **18.35%** | **6.01%** |
| replay + literal-only calibration | 10.72% | **51.10%** | **43.64%** | **18.35%** | not selected |

The best calibrated checkpoint is
`checkpoints/base-semantic-v18-filter-calibrated-best-execution`. Its evaluation
tracks are 9.60% human, 1.39% semantic paraphrases, and 31.50% synthetic. The
original V18 joint checkpoint was 9.40%, 1.70%, and 29.00% respectively.
On the dedicated schema-disjoint contrastive gate it reaches 47.02%
filter-column accuracy, 44.64% complete pointers, and 13.29% complete structured
heads, but still 0% execution; literal-span endings and remaining structure
errors prevent those partial predictions from assembling into a correct query.

## Conclusion

Targeted data clearly teaches schema pointers: filter-column accuracy improves
by 35.29 points and complete pointer accuracy more than triples. The gain does
not translate proportionally to execution because operation and literal heads
must remain calibrated to the changed filter context. Even the best result is
only 0.58 points above V18 joint execution and remains far below V16's 27.07%
mixed-validation score. This checkpoint is therefore an experimental V18
candidate, not a promoted release.

The next work should coordinate schema, operation, and literal training without
letting simple contrastive examples dominate operation distributions. Adding
more undifferentiated data is not supported by this experiment.

# Test fixtures

This directory holds optional ground-truth files that cannot be checked into
the repo because of size or licensing.

## OOF predictions for metric validation

To complete the BLOCKER validation in Step 5 of Session 1, drop a public
Kaggle OOF prediction file here, then run:

```bash
uv run python -m amex.evaluation.validate_oof \
    --oof tests/fixtures/<your_oof>.parquet \
    --expected 0.79xxx \
    --tolerance 0.001
```

### Where to get OOF files

Several top Kaggle solutions published their OOF predictions:

- 1st place writeup discussion (search "amex 1st place OOF" on Kaggle)
- Chris Deotte's RAPIDS SVR baseline
- The competition's official `sample_submission.csv` (but that's all 0s; not useful)

The file format expected: Parquet or CSV with columns
`customer_ID`, `prediction` (other common names like `preds`, `score`,
`probability` are accepted as aliases).

The published CV/LB score from the solution writeup goes into `--expected`.
The script asserts |computed - expected| <= tolerance and exits non-zero on
mismatch -- use this in CI gates.

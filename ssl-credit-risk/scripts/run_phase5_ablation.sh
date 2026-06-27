#!/usr/bin/env bash
# Run the 3 Phase-5 ablation GBM trainings sequentially.
# Each takes ~60 min; total ~3 h on this CPU.

set -euo pipefail
# (ensure uv is on PATH)
export PYTHONIOENCODING=utf-8

WANDB_FLAG="${WANDB_FLAG:---wandb}"  # set to --no-wandb to disable

run_one() {
    local tag="$1"
    local features="$2"
    echo
    echo "======================================================"
    echo "==  ABLATION ${tag}"
    echo "==  features: ${features}"
    echo "======================================================"
    uv run python -m amex.baseline.lgbm \
        --features "${features}" \
        --oof-out "data/processed/v1/oof_lgbm_${tag}.parquet" \
        --metrics-out "data/processed/v1/oof_lgbm_${tag}_metrics.json" \
        "${WANDB_FLAG}"
}

# (ii) hand - top100
run_one "hand_minus_top100" "data/processed/v1/features_hand_minus_top100.parquet"

# (iii) (hand - top100) + SSL
run_one "hand_minus_top100_plus_ssl" "data/processed/v1/features_hand_minus_top100_plus_ssl.parquet"

# (iv) SSL only
run_one "ssl_only" "data/processed/v1/features_ssl_only.parquet"

echo
echo "ALL 3 ABLATIONS DONE"
ls -la data/processed/v1/oof_lgbm_*ablation*.json 2>/dev/null || true
ls -la data/processed/v1/oof_lgbm_{hand_minus_top100,hand_minus_top100_plus_ssl,ssl_only}_metrics.json

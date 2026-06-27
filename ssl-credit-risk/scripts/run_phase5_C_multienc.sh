#!/usr/bin/env bash
# Phase 5-C: stack all 4 encoder embeddings (4 x 128 = 512 SSL cols) and
# retrain LightGBM. Expects all 4 features_ssl_*.parquet to exist.

set -euo pipefail
# (ensure uv is on PATH)
export PYTHONIOENCODING=utf-8

# 1. Sanity: all 4 ssl parquets present.
for name in hybrid masked nextstep contrastive; do
  f="data/processed/v1/features_ssl_${name}.parquet"
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    exit 1
  fi
done
echo "all 4 SSL embedding parquets present."

# 2. Build augmented features (1291 hand + 4 x 128 SSL = 1803 cols).
echo
echo "===  augment 4-encoder  ==="
uv run python scripts/make_augmented_features.py \
  --ssl data/processed/v1/features_ssl_hybrid.parquet \
  --ssl data/processed/v1/features_ssl_masked.parquet \
  --ssl data/processed/v1/features_ssl_nextstep.parquet \
  --ssl data/processed/v1/features_ssl_contrastive.parquet \
  --out data/processed/v1/features_augmented_4enc.parquet

# 3. Train LightGBM on the multi-encoder augmented feature set.
echo
echo "===  GBM 5-fold on 4-encoder augmented features  ==="
uv run python -m amex.baseline.lgbm \
  --features data/processed/v1/features_augmented_4enc.parquet \
  --oof-out data/processed/v1/oof_lgbm_4enc.parquet \
  --metrics-out data/processed/v1/oof_lgbm_4enc_metrics.json \
  --wandb

echo
echo "PHASE 5-C DONE"

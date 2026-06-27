#!/usr/bin/env bash
# Phase 5-D: multi-seed robustness for the hybrid SSL pipeline.
# For each seed: pretrain hybrid -> extract embeddings -> augment -> train GBM.
# Each cycle ~2.5h. SEEDS env var lets you control which ones to run.

set -euo pipefail
# (ensure uv is on PATH)
export PYTHONIOENCODING=utf-8

SEEDS="${SEEDS:-1 2}"   # default: two new seeds (existing run is the de-facto seed-42 baseline)
WANDB_FLAG="${WANDB_FLAG:---wandb}"

for seed in $SEEDS; do
  echo
  echo "######################################################"
  echo "##  D seed=${seed}"
  echo "######################################################"

  # 1) pretrain hybrid with this seed (Hydra +seed=$seed adds it to the cfg
  # so the run_name picks up the -s$seed suffix and a distinct config hash)
  echo
  echo "===  pretrain hybrid seed=${seed}"
  uv run python -m amex.ssl.pretrain --config-name hybrid \
    +seed=${seed} \
    trainer.max_epochs=10 trainer.patience=3 \
    trainer.limit_train_batches=360 data.batch_size=512 \
    trainer.wandb=true

  # Find the freshly-created checkpoint dir for this seed.
  ckpt_dir=$(ls -td "checkpoints/hybrid-"*"-s${seed}"/ 2>/dev/null | head -1)
  if [ -z "${ckpt_dir}" ]; then
    echo "!! no checkpoint dir matched 'checkpoints/hybrid-*-s${seed}/'"
    exit 1
  fi
  ckpt="${ckpt_dir%/}/encoder.pt"
  echo "ckpt: ${ckpt}"

  # 2) extract embeddings (full label pool + Kaggle test, like Phase 4)
  emb="data/processed/v1/features_ssl_hybrid_s${seed}.parquet"
  echo
  echo "===  extract emb seed=${seed} -> ${emb}"
  uv run python -m amex.baseline.ssl_features \
    --encoder "${ckpt}" \
    --out "${emb}" \
    --prefix "ssl_hybrid_s${seed}"

  # 3) augment features with this seed's embedding
  aug="data/processed/v1/features_augmented_hybrid_s${seed}.parquet"
  echo
  echo "===  augment seed=${seed}"
  uv run python scripts/make_augmented_features.py \
    --ssl "${emb}" \
    --out "${aug}"

  # 4) GBM 5-fold CV
  echo
  echo "===  GBM seed=${seed}"
  uv run python -m amex.baseline.lgbm \
    --features "${aug}" \
    --oof-out "data/processed/v1/oof_lgbm_hybrid_s${seed}.parquet" \
    --metrics-out "data/processed/v1/oof_lgbm_hybrid_s${seed}_metrics.json" \
    "${WANDB_FLAG}"
done

echo
echo "PHASE 5-D DONE"
ls -lh data/processed/v1/oof_lgbm_hybrid_s*_metrics.json

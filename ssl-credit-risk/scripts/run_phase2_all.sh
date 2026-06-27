#!/usr/bin/env bash
# Run all 4 SSL objectives sequentially, then linear-probe each.
# Designed to fit a Session-2 budget on a single RTX 4070L: ~3-4 hours total.
#
# Logs everything to W&B (project amex-ssl). Set WANDB=false to disable.
#
# Usage:
#   bash scripts/run_phase2_all.sh

set -euo pipefail
# (ensure uv is on PATH)
export PYTHONIOENCODING=utf-8

WANDB="${WANDB:-true}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
# 367k customers / batch 512 ~= 717 batches per full epoch; 360 ~= half.
# IterableDataset requires limit_train_batches to be 1.0 or an int.
LIMIT_TRAIN="${LIMIT_TRAIN:-360}"
BATCH_SIZE="${BATCH_SIZE:-512}"

OBJECTIVES=(masked nextstep contrastive hybrid)

for obj in "${OBJECTIVES[@]}"; do
    echo
    echo "======================================================"
    echo "==  PRETRAIN  ${obj}"
    echo "======================================================"
    uv run python -m amex.ssl.pretrain \
        --config-name "${obj}" \
        trainer.max_epochs="${MAX_EPOCHS}" \
        trainer.patience="${PATIENCE}" \
        trainer.limit_train_batches="${LIMIT_TRAIN}" \
        data.batch_size="${BATCH_SIZE}" \
        trainer.wandb="${WANDB}"

    # Pick the most-recently-modified checkpoint dir for this objective.
    ckpt_dir=$(ls -td "checkpoints/${obj}-"*/ 2>/dev/null | head -1)
    if [ -z "${ckpt_dir}" ]; then
        echo "!! no checkpoint dir found for ${obj} -- aborting"
        exit 1
    fi
    echo "linear-probing encoder at: ${ckpt_dir}encoder.pt"

    echo
    echo "======================================================"
    echo "==  LINEAR PROBE  ${obj}"
    echo "======================================================"
    uv run python -m amex.finetune.linear_probe \
        --encoder "${ckpt_dir}encoder.pt"
done

echo
echo "ALL 4 OBJECTIVES DONE"
ls -la data/processed/v1/probes/

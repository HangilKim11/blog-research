#!/usr/bin/env bash
# Fine-tune each of the 4 pretrained encoders end-to-end.
# Usage: bash scripts/run_phase3_finetune.sh

set -euo pipefail
# (ensure uv is on PATH)
export PYTHONIOENCODING=utf-8

WANDB="${WANDB:-true}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
PATIENCE="${PATIENCE:-3}"
LIMIT_TRAIN="${LIMIT_TRAIN:-800}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# Map objective -> the encoder.pt to fine-tune.
declare -A ENCODERS=(
    [masked]="checkpoints/masked-4e482fd6/encoder.pt"
    [nextstep]="checkpoints/nextstep-274fdf8c/encoder.pt"
    [contrastive]="checkpoints/contrastive-d1a83e18/encoder.pt"
    [hybrid]="checkpoints/hybrid-e3d5d881/encoder.pt"
)

for obj in masked nextstep contrastive hybrid; do
    ckpt="${ENCODERS[$obj]}"
    if [ ! -f "${ckpt}" ]; then
        echo "!! missing encoder: ${ckpt} -- skipping ${obj}"
        continue
    fi
    out_dir="checkpoints/finetune-$(basename "$(dirname "$ckpt")")"
    rm -rf "${out_dir}"   # ensure fresh run, no resume from previous best

    echo
    echo "======================================================"
    echo "==  FINETUNE  ${obj}    (encoder: ${ckpt})"
    echo "======================================================"
    uv run python -m amex.finetune.full_finetune \
        encoder.ckpt="${ckpt}" \
        trainer.max_epochs="${MAX_EPOCHS}" \
        trainer.patience="${PATIENCE}" \
        trainer.limit_train_batches="${LIMIT_TRAIN}" \
        data.batch_size="${BATCH_SIZE}" \
        trainer.wandb="${WANDB}"
done

echo
echo "ALL 4 FINETUNES DONE"
ls -la data/processed/v1/finetunes/

#!/usr/bin/env bash
# Few-shot study: GBM vs best SSL fine-tune, across labeled-data fractions.
#
# Both methods see the SAME stratified subset of trainval (stratified_trainval_subset)
# at each fraction. Test set (10% holdout) is never subsetted.
#
# Usage:
#   bash scripts/run_few_shot.sh                       # uses BEST_ENC default
#   BEST_ENC=checkpoints/finetune-masked-4e482fd6/best.ckpt bash scripts/run_few_shot.sh

set -euo pipefail
# (ensure uv is on PATH)
export PYTHONIOENCODING=utf-8

WANDB="${WANDB:-true}"
FRACTIONS="${FRACTIONS:-0.01 0.05 0.25 1.0}"
# Default: fine-tune from the pretrained ENCODER of the best Phase 2 objective
# (next-step). When the Phase 3 fine-tunes finish we can also point at the
# *fine-tuned* encoders -- but for few-shot we want to start from pretrain.
BEST_ENC="${BEST_ENC:-checkpoints/nextstep-274fdf8c/encoder.pt}"

# Pick a per-fraction batch_size + train-batch cap that fits the data size.
# (At 1% we have ~4k customers; batch=512 leaves <1 val batch after carve-out.)
pick_batch() {
    case "$1" in
        0.01)  echo 64  ;;
        0.05)  echo 128 ;;
        0.25)  echo 256 ;;
        *)     echo 512 ;;
    esac
}
pick_limit_train() {
    # number of train batches per epoch (int, IterableDataset needs int)
    case "$1" in
        0.01)  echo 64  ;;
        0.05)  echo 200 ;;
        0.25)  echo 400 ;;
        *)     echo 800 ;;
    esac
}

out_dir="data/processed/v1/few_shot"
mkdir -p "${out_dir}"

for f in ${FRACTIONS}; do
    echo
    echo "######################################################"
    echo "## FRACTION ${f}"
    echo "######################################################"

    # ---- GBM ----
    echo "-- lgbm subset=${f} --"
    uv run python -m amex.baseline.lgbm \
        --subset-fraction "${f}" \
        --oof-out "${out_dir}/gbm_f${f}_oof.parquet" \
        --metrics-out "${out_dir}/gbm_f${f}.json" \
        $( [ "${WANDB}" = "true" ] && echo "--wandb" || echo "--no-wandb" )

    # ---- SSL fine-tune ----
    echo "-- finetune subset=${f} --"
    bs=$(pick_batch "${f}")
    lt=$(pick_limit_train "${f}")
    # Wipe any prior ckpt for this run-name so we don't accidentally reuse weights.
    run_ckpt_dir="checkpoints/finetune-$(basename "$(dirname "${BEST_ENC}")")"
    rm -rf "${run_ckpt_dir}"
    uv run python -m amex.finetune.full_finetune \
        encoder.ckpt="${BEST_ENC}" \
        data.subset_fraction="${f}" \
        trainer.max_epochs=8 \
        trainer.patience=3 \
        trainer.limit_train_batches="${lt}" \
        data.batch_size="${bs}" \
        trainer.wandb="${WANDB}" \
        hydra.run.dir="outputs/few_shot_${f}"

    # finetune.py writes to data/processed/v1/finetunes/finetune-<encoder-dirname>.json
    # but that gets overwritten across fractions; rename a copy:
    src_json="data/processed/v1/finetunes/finetune-$(basename "$(dirname "${BEST_ENC}")").json"
    if [ -f "${src_json}" ]; then
        cp "${src_json}" "${out_dir}/ssl_f${f}.json"
    fi
done

echo
echo "FEW-SHOT DONE -- artifacts in ${out_dir}/"
ls -la "${out_dir}/"

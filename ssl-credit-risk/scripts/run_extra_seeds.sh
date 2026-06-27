#!/usr/bin/env bash
# ============================================================================
# STRENGTHEN (A): more seeds.
#
# Runs 5 ADDITIONAL hybrid-SSL pretrain seeds (3,4,5,6,7) end-to-end through the
# full Phase 5-D pipeline -- pretrain -> ssl_features embed -> make_augmented
# -> lgbm 5-fold -- exactly mirroring scripts/run_phase5_D_seeds.sh. Each seed
# produces a distinct  -s{seed}  checkpoint / embedding / augmented feature /
# OOF, and its test-AMEX is appended to a machine-readable JSON.
#
# Combined with the existing seed-42 baseline + seeds 1,2, this gives 8 seeds
# total for the SSL-vs-baseline robustness claim.
#
# LAUNCH:
#     bash scripts/run_extra_seeds.sh
#
# Override which seeds run (e.g. just one):
#     SEEDS="3" bash scripts/run_extra_seeds.sh
#
# Disable W&B:
#     WANDB_FLAG="--no-wandb" bash scripts/run_extra_seeds.sh
#
# WALL-CLOCK: ~2.5 h per seed (pretrain ~2h on the RTX 4070 + embed/augment/GBM
#             ~30 min) => ~12-13 h for all five seeds. Resumable: a seed whose
#             GBM metrics JSON already exists is skipped, so you can Ctrl-C and
#             re-launch.
#
# OUTPUTS:
#   checkpoints/hybrid-<hash>-s{seed}/encoder.pt
#   data/processed/v1/features_ssl_hybrid_s{seed}.parquet
#   data/processed/v1/features_augmented_hybrid_s{seed}.parquet
#   data/processed/v1/oof_lgbm_hybrid_s{seed}.parquet
#   data/processed/v1/oof_lgbm_hybrid_s{seed}_metrics.json
#   reports/strengthen_seeds.json        <-- aggregated per-seed test AMEX
#   logs/extra_seeds_<runname>.log       <-- full stdout (via launch redirect)
# ============================================================================

set -euo pipefail
# (ensure uv is on PATH)
export PYTHONIOENCODING=utf-8

SEEDS="${SEEDS:-3 4 5 6 7}"
WANDB_FLAG="${WANDB_FLAG:---wandb}"
RUNNAME="${RUNNAME:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p logs reports
RESULTS_JSON="reports/strengthen_seeds.json"
LOG="logs/extra_seeds_${RUNNAME}.log"

# Tee everything to a per-run log while still showing on stdout.
exec > >(tee -a "${LOG}") 2>&1
echo "=== run_extra_seeds.sh  runname=${RUNNAME}  seeds='${SEEDS}'  wandb='${WANDB_FLAG}'"
echo "=== logging to ${LOG}"

# Initialise the aggregate results file if missing.
if [ ! -f "${RESULTS_JSON}" ]; then
  echo '{"experiment": "strengthen_seeds", "splits": "v1", "encoder": "hybrid", "results": {}}' > "${RESULTS_JSON}"
fi

# Append (seed -> test AMEX + full test block) into the aggregate JSON, reading
# the per-seed metrics file the lgbm step just wrote. Idempotent per seed.
record_result () {
  local seed="$1"
  local metrics="data/processed/v1/oof_lgbm_hybrid_s${seed}_metrics.json"
  uv run python - "$seed" "$metrics" "$RESULTS_JSON" <<'PY'
import json, sys
seed, metrics_path, results_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(metrics_path, encoding="utf-8") as f:
    m = json.load(f)
test = m.get("test", {}) or {}
oof = m.get("oof", {}) or {}
with open(results_path, encoding="utf-8") as f:
    agg = json.load(f)
agg.setdefault("results", {})[f"seed_{seed}"] = {
    "seed": int(seed),
    "test_amex": test.get("amex"),
    "oof_amex": oof.get("amex"),
    "test_auc": test.get("auc"),
    "test_block": test,
    "metrics_file": metrics_path,
}
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(agg, f, indent=2)
    f.write("\n")
print(f"recorded seed={seed} test_amex={test.get('amex')}")
PY
}

for seed in $SEEDS; do
  echo
  echo "######################################################"
  echo "##  EXTRA seed=${seed}"
  echo "######################################################"

  metrics="data/processed/v1/oof_lgbm_hybrid_s${seed}_metrics.json"
  if [ -f "${metrics}" ]; then
    echo "== seed=${seed} already complete (${metrics} exists) -- recording + skipping."
    record_result "${seed}"
    continue
  fi

  # 1) pretrain hybrid with this seed. Hydra `+seed=$seed` adds it to the cfg so
  #    the run_name picks up the -s$seed suffix and a distinct config hash.
  ckpt_dir=$(ls -td "checkpoints/hybrid-"*"-s${seed}"/ 2>/dev/null | head -1 || true)
  ckpt="${ckpt_dir%/}/encoder.pt"
  if [ -n "${ckpt_dir}" ] && [ -f "${ckpt}" ]; then
    echo "== pretrain seed=${seed} already done -> ${ckpt} (skip pretrain)"
  else
    echo
    echo "===  pretrain hybrid seed=${seed}"
    uv run python -m amex.ssl.pretrain --config-name hybrid \
      +seed=${seed} \
      trainer.max_epochs=10 trainer.patience=3 \
      trainer.limit_train_batches=360 data.batch_size=512 \
      trainer.wandb=true
    ckpt_dir=$(ls -td "checkpoints/hybrid-"*"-s${seed}"/ 2>/dev/null | head -1 || true)
    if [ -z "${ckpt_dir}" ]; then
      echo "!! no checkpoint dir matched 'checkpoints/hybrid-*-s${seed}/'"
      exit 1
    fi
    ckpt="${ckpt_dir%/}/encoder.pt"
  fi
  echo "ckpt: ${ckpt}"

  # 2) extract embeddings (full label pool + Kaggle test).
  emb="data/processed/v1/features_ssl_hybrid_s${seed}.parquet"
  if [ -f "${emb}" ]; then
    echo "== embeddings seed=${seed} already exist -> ${emb} (skip embed)"
  else
    echo
    echo "===  extract emb seed=${seed} -> ${emb}"
    uv run python -m amex.baseline.ssl_features \
      --encoder "${ckpt}" \
      --out "${emb}" \
      --prefix "ssl_hybrid_s${seed}"
  fi

  # 3) augment hand-crafted features with this seed's embedding.
  aug="data/processed/v1/features_augmented_hybrid_s${seed}.parquet"
  if [ -f "${aug}" ]; then
    echo "== augmented features seed=${seed} already exist -> ${aug} (skip augment)"
  else
    echo
    echo "===  augment seed=${seed}"
    uv run python scripts/make_augmented_features.py \
      --ssl "${emb}" \
      --out "${aug}"
  fi

  # 4) GBM 5-fold CV (writes the metrics file that gates resumability).
  echo
  echo "===  GBM seed=${seed}"
  uv run python -m amex.baseline.lgbm \
    --features "${aug}" \
    --oof-out "data/processed/v1/oof_lgbm_hybrid_s${seed}.parquet" \
    --metrics-out "${metrics}" \
    "${WANDB_FLAG}"

  record_result "${seed}"
done

echo
echo "=== EXTRA SEEDS DONE"
echo "=== aggregate results: ${RESULTS_JSON}"
uv run python - "$RESULTS_JSON" <<'PY'
import json, sys
agg = json.load(open(sys.argv[1], encoding="utf-8"))
res = agg.get("results", {})
print(f"{'seed':<10}{'test_amex':>12}{'oof_amex':>12}")
for k in sorted(res):
    r = res[k]
    ta = r.get("test_amex"); oa = r.get("oof_amex")
    ta = f"{ta:.5f}" if isinstance(ta, (int, float)) else str(ta)
    oa = f"{oa:.5f}" if isinstance(oa, (int, float)) else str(oa)
    print(f"{k:<10}{ta:>12}{oa:>12}")
PY

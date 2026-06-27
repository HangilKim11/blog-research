#!/usr/bin/env bash
# ============================================================================
# STRENGTHEN (B): out-of-time (temporal) validation.
#
# Re-establishes the headline (GBM baseline vs GBM+SSL fusion) on the TEMPORAL
# split data/splits/temporal_v1.parquet, where test customers' last statements
# are strictly LATER (by calendar day within 2018-03) than train customers'.
#
# ---------------------------------------------------------------------------
# DEFAULT = RIGOROUS  (~3-4 h):
#   The hybrid encoder is RE-PRETRAINED on temporal-TRAIN customers ONLY, by
#   pointing pretrain at the temporal splits file (data.splits_path=...). The
#   SSL encoder therefore never sees a test-period customer during pretraining
#   -- the strongest version of the no-leakage claim. ONE pretrain seed (42) to
#   cap runtime. Then embed -> augment -> run BOTH:
#     (1) GBM baseline : hand features only,  --splits temporal_v1.parquet
#     (2) GBM + SSL    : hand + temporal SSL emb, --splits temporal_v1.parquet
#   Delta (fusion - baseline) on the out-of-time test set is the result.
#
# ---------------------------------------------------------------------------
# CHEAP FALLBACK  (~2 h) -- if the rigorous run is too slow / GPU busy:
#   Skip re-pretraining. REUSE the existing seed-42 hybrid encoder
#   (checkpoints/hybrid-e3d5d881/encoder.pt) and its already-extracted
#   embedding (data/processed/v1/features_ssl_hybrid.parquet), and ONLY re-fit
#   the GBMs on the temporal split. This tests temporal generalisation of the
#   *fusion model* but NOT of the SSL pretraining itself (the encoder saw all
#   customers' history during its original v1 pretrain -- though never their
#   labels, and never the test-period *position* in time). Launch with:
#       MODE=cheap bash scripts/run_oot.sh
#   In cheap mode the embedding step is skipped and the existing
#   features_ssl_hybrid.parquet is reused for augmentation.
#
# ---------------------------------------------------------------------------
# LAUNCH (rigorous, default):
#     bash scripts/run_oot.sh
# LAUNCH (cheap fallback):
#     MODE=cheap bash scripts/run_oot.sh
# Disable W&B:
#     WANDB_FLAG="--no-wandb" bash scripts/run_oot.sh
#
# WALL-CLOCK: rigorous ~3-4 h (pretrain ~2h + embed ~10m + 2x GBM ~1h).
#             cheap    ~2 h   (2x GBM ~1h, no pretrain/embed).
#
# OUTPUTS:
#   checkpoints/hybrid-<hash>-soot/encoder.pt        (rigorous only; seed tag 'oot')
#   data/processed/v1/features_ssl_hybrid_oot.parquet (rigorous only)
#   data/processed/v1/features_augmented_oot.parquet
#   data/processed/v1/oof_lgbm_oot_baseline{,_metrics}.{parquet,json}
#   data/processed/v1/oof_lgbm_oot_fusion{,_metrics}.{parquet,json}
#   reports/strengthen_oot.json    <-- baseline test AMEX, fusion test AMEX, delta
#   logs/oot_<runname>.log         <-- full stdout
# ============================================================================

set -euo pipefail
# (ensure uv is on PATH)
export PYTHONIOENCODING=utf-8

MODE="${MODE:-rigorous}"          # rigorous | cheap
OOT_SEED="${OOT_SEED:-oot}"       # tag used in the temporal pretrain run_name (-s$OOT_SEED)
WANDB_FLAG="${WANDB_FLAG:---wandb}"
RUNNAME="${RUNNAME:-$(date +%Y%m%d_%H%M%S)}"

SPLITS="data/splits/temporal_v1.parquet"
BASE_FEATS="data/processed/v1/features_lgbm.parquet"

mkdir -p logs reports
RESULTS_JSON="reports/strengthen_oot.json"
LOG="logs/oot_${RUNNAME}.log"

exec > >(tee -a "${LOG}") 2>&1
echo "=== run_oot.sh  mode=${MODE}  runname=${RUNNAME}  splits=${SPLITS}  wandb='${WANDB_FLAG}'"
echo "=== logging to ${LOG}"

if [ ! -f "${SPLITS}" ]; then
  echo "!! temporal splits file missing: ${SPLITS}"
  echo "!! generate it first: uv run python -m amex.data.splits_temporal"
  exit 1
fi

# --------------------------------------------------------------------------
# 1) Obtain the SSL embedding parquet to fuse.
#    rigorous: re-pretrain on temporal-TRAIN only, then embed all customers.
#    cheap   : reuse the existing v1 seed-42 embedding.
# --------------------------------------------------------------------------
if [ "${MODE}" = "cheap" ]; then
  SSL_EMB="data/processed/v1/features_ssl_hybrid.parquet"
  echo "== cheap mode: reusing existing SSL embedding ${SSL_EMB}"
  if [ ! -f "${SSL_EMB}" ]; then
    echo "!! expected existing embedding ${SSL_EMB} not found -- run the v1 pipeline first or use rigorous mode."
    exit 1
  fi
else
  SSL_EMB="data/processed/v1/features_ssl_hybrid_oot.parquet"

  # 1a) re-pretrain hybrid on temporal-TRAIN customers ONLY.
  ckpt_dir=$(ls -td "checkpoints/hybrid-"*"-s${OOT_SEED}"/ 2>/dev/null | head -1 || true)
  ckpt="${ckpt_dir%/}/encoder.pt"
  if [ -n "${ckpt_dir}" ] && [ -f "${ckpt}" ]; then
    echo "== rigorous: temporal encoder already exists -> ${ckpt} (skip pretrain)"
  else
    echo
    echo "===  pretrain hybrid on TEMPORAL-TRAIN only  (splits_path=${SPLITS}, seed=${OOT_SEED})"
    # data.splits_path override makes pretrain.py select train/val customers from
    # the temporal file, so the encoder never sees test-period customers.
    uv run python -m amex.ssl.pretrain --config-name hybrid \
      +seed=${OOT_SEED} \
      data.splits_path=${SPLITS} \
      trainer.max_epochs=10 trainer.patience=3 \
      trainer.limit_train_batches=360 data.batch_size=512 \
      trainer.wandb=true
    ckpt_dir=$(ls -td "checkpoints/hybrid-"*"-s${OOT_SEED}"/ 2>/dev/null | head -1 || true)
    if [ -z "${ckpt_dir}" ]; then
      echo "!! no checkpoint dir matched 'checkpoints/hybrid-*-s${OOT_SEED}/'"
      exit 1
    fi
    ckpt="${ckpt_dir%/}/encoder.pt"
  fi
  echo "ckpt: ${ckpt}"

  # 1b) embed ALL customers with the temporal encoder (ssl_features has no split
  #     filter -- it embeds every customer in the train+test trees; only the GBM
  #     respects the temporal split, so embeddings cover train/val/test alike).
  if [ -f "${SSL_EMB}" ]; then
    echo "== temporal embedding already exists -> ${SSL_EMB} (skip embed)"
  else
    echo
    echo "===  extract temporal emb -> ${SSL_EMB}"
    uv run python -m amex.baseline.ssl_features \
      --encoder "${ckpt}" \
      --out "${SSL_EMB}" \
      --prefix "ssl_hybrid_oot"
  fi
fi

# --------------------------------------------------------------------------
# 2) Build augmented features (hand + chosen SSL embedding).
# --------------------------------------------------------------------------
AUG="data/processed/v1/features_augmented_oot.parquet"
if [ "${MODE}" = "cheap" ]; then
  AUG="data/processed/v1/features_augmented_oot_cheap.parquet"
fi
if [ -f "${AUG}" ]; then
  echo "== augmented features already exist -> ${AUG} (skip augment)"
else
  echo
  echo "===  augment (base=${BASE_FEATS}  ssl=${SSL_EMB})"
  uv run python scripts/make_augmented_features.py \
    --base "${BASE_FEATS}" \
    --ssl "${SSL_EMB}" \
    --out "${AUG}"
fi

# --------------------------------------------------------------------------
# 3a) GBM BASELINE on the temporal split -- hand features only.
# --------------------------------------------------------------------------
BASE_OOF="data/processed/v1/oof_lgbm_oot_baseline.parquet"
BASE_METRICS="data/processed/v1/oof_lgbm_oot_baseline_metrics.json"
if [ -f "${BASE_METRICS}" ]; then
  echo "== baseline GBM already done -> ${BASE_METRICS} (skip)"
else
  echo
  echo "===  GBM baseline (hand features, temporal split)"
  uv run python -m amex.baseline.lgbm \
    --features "${BASE_FEATS}" \
    --splits "${SPLITS}" \
    --oof-out "${BASE_OOF}" \
    --metrics-out "${BASE_METRICS}" \
    "${WANDB_FLAG}"
fi

# --------------------------------------------------------------------------
# 3b) GBM + SSL FUSION on the temporal split -- augmented features.
# --------------------------------------------------------------------------
FUSE_OOF="data/processed/v1/oof_lgbm_oot_fusion.parquet"
FUSE_METRICS="data/processed/v1/oof_lgbm_oot_fusion_metrics.json"
if [ -f "${FUSE_METRICS}" ]; then
  echo "== fusion GBM already done -> ${FUSE_METRICS} (skip)"
else
  echo
  echo "===  GBM + SSL fusion (augmented features, temporal split)"
  uv run python -m amex.baseline.lgbm \
    --features "${AUG}" \
    --splits "${SPLITS}" \
    --oof-out "${FUSE_OOF}" \
    --metrics-out "${FUSE_METRICS}" \
    "${WANDB_FLAG}"
fi

# --------------------------------------------------------------------------
# 4) Record baseline / fusion / delta on the out-of-time test set.
# --------------------------------------------------------------------------
echo
echo "===  writing ${RESULTS_JSON}"
uv run python - "$MODE" "$SPLITS" "$BASE_METRICS" "$FUSE_METRICS" "$RESULTS_JSON" <<'PY'
import json, sys
mode, splits, base_p, fuse_p, out_p = sys.argv[1:6]
def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)
b, f = load(base_p), load(fuse_p)
bt = (b.get("test") or {}).get("amex")
ft = (f.get("test") or {}).get("amex")
delta = (ft - bt) if (isinstance(bt, (int, float)) and isinstance(ft, (int, float))) else None
out = {
    "experiment": "strengthen_oot",
    "mode": mode,
    "splits": splits,
    "baseline": {
        "test_amex": bt,
        "oof_amex": (b.get("oof") or {}).get("amex"),
        "test_block": b.get("test"),
        "metrics_file": base_p,
    },
    "fusion": {
        "test_amex": ft,
        "oof_amex": (f.get("oof") or {}).get("amex"),
        "test_block": f.get("test"),
        "metrics_file": fuse_p,
    },
    "delta_test_amex": delta,
}
with open(out_p, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
    fh.write("\n")
print(f"baseline test AMEX = {bt}")
print(f"fusion   test AMEX = {ft}")
print(f"delta (fusion-baseline) = {delta}")
PY

echo
echo "=== OOT DONE (mode=${MODE})"
echo "=== results: ${RESULTS_JSON}"

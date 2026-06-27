#!/usr/bin/env bash
# Extract embeddings for the three Phase-2 encoders we haven't dumped yet.
# Each run is ~22 min on RTX 4070L 8 GB.

set -euo pipefail
# (ensure uv is on PATH)
export PYTHONIOENCODING=utf-8

declare -A ENCODERS=(
  [masked]=checkpoints/masked-4e482fd6/encoder.pt
  [nextstep]=checkpoints/nextstep-274fdf8c/encoder.pt
  [contrastive]=checkpoints/contrastive-d1a83e18/encoder.pt
)

for name in masked nextstep contrastive; do
  ckpt="${ENCODERS[$name]}"
  out="data/processed/v1/features_ssl_${name}.parquet"
  if [ -f "$out" ]; then
    echo "[skip] $out already exists"
    continue
  fi
  echo
  echo "======================================================"
  echo "==  extract ${name}  ${ckpt}"
  echo "======================================================"
  uv run python -m amex.baseline.ssl_features \
    --encoder "$ckpt" \
    --out "$out" \
    --prefix "ssl_${name}"
done

echo
echo "ALL 3 EXTRACTIONS DONE"
ls -lh data/processed/v1/features_ssl_*.parquet

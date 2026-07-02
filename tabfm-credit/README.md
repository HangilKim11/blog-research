# TabFM vs GBM on credit default — public-data validation

> Does Google's zero-shot **TabFM** (tabular foundation model) beat a well-tuned GBM at credit-default prediction? Reproduction code, tested on public data (UCI Taiwan credit-card default).

Full write-up: **[han-co.com/ko/blog/tabfm-credit](https://han-co.com/ko/blog/tabfm-credit)** · [JA](https://han-co.com/ja/blog/tabfm-credit) — 한국어 [README.ko.md](README.ko.md) · 日本語 [README.ja.md](README.ja.md)

## Key result (5-fold, natural class prior — no class_weight)

| Arm | ROC-AUC | PR-AUC | KS | ECE ↓ |
|---|---:|---:|---:|---:|
| GBM tuned (LightGBM) | 0.789 | 0.566 | 0.443 | 0.010 |
| **TabFM zero-shot** | **0.785** | **0.558** | **0.441** | **0.022** |
| GBM raw | 0.779 | 0.554 | 0.429 | 0.013 |

A well-tuned GBM edges out zero-shot TabFM (~0.4pp, within fold noise); effort-matched, TabFM beats raw GBM; calibration is a tie. The dataset ceilings around 0.79. TabFM is not a "GBM-beater" here, but a strong zero-effort baseline.

## Run

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git"
jupyter notebook tabfm_credit.ko.ipynb   # or tabfm_credit.ja.ipynb
```

- Data auto-downloads via `ucimlrepo` (UCI id=350).
- **The TabFM model is not bundled** — it downloads from Hugging Face (`google/tabfm-1.0.0-pytorch`, ~6.5GB VRAM) at runtime. A CUDA GPU is required for the TabFM arm.
- 8GB-GPU settings (context capped at 1,000 rows); raise `CONTEXT_MAX` and use the ensemble preset on a ≥16GB GPU.

## Notebook

Three arms — raw GBM, tuned GBM (Optuna), TabFM zero-shot — compared with stratified 5-fold. Discrimination (ROC-AUC, PR-AUC, KS) + calibration (Brier, LogLoss, ECE). The full six-arm run (incl. CatBoost, XGBoost, TabFM ensemble) lives in the original experiment code (config-driven).

## License / caveats

Code: free to use. Data: UCI (public). Conclusions are for a single public dataset; other default data or out-of-time validation may differ. TabFM weights/architecture belong to Google (see the upstream repo).

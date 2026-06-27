# SSL 신용 리스크 — 공개 데이터 검증

> SSL 단독으로는 GBM에 못 미칩니다. 그런데 GBM에 합치면 개선됩니다.
>
> Self-supervised pretraining alone underperforms a tuned GBM on tabular credit risk. Merged into the GBM as auxiliary features, it improves it.

신용 리스크 실무는 사실상 GBM 단일 문화입니다. NLP와 비전에서 자기지도학습(SSL)이 성공한 것을 보고, 같은 접근이 신용 데이터에도 통하는지 공개 데이터로 검증했습니다. 결론은 두 줄로 요약됩니다. 트랜잭션 시퀀스를 SSL로 사전학습한 인코더는 단독 분류기로는 잘 튜닝된 GBM에 못 미칩니다. 하지만 그 임베딩을 GBM의 추가 피처로 합치면 GBM 단독보다 안정적으로 개선됩니다.

전체 글: **[han-co.com/ko/blog/ssl-credit-risk](https://han-co.com/ko/blog/ssl-credit-risk)**

## 핵심 결과

| 접근 | test AMEX | 베이스라인 대비 |
|---|---:|---:|
| **GBM 베이스라인** (hand 피처 1,291개) | **0.79558** | 0 |
| SSL 단독 최고 (Hybrid full fine-tune) | 0.79267 | -0.00291 (못 미침) |
| **GBM + SSL 병합** (시드 6개 평균) | **0.79675** | **+0.00117** |
| GBM + SSL 병합, 최고 시드 | 0.79807 | +0.00249 |

병합 결과는 시드 6개 평균 test AMEX 0.79675, 베이스라인 대비 +0.00117, 표준편차 0.00098입니다. t값은 약 2.9(자유도 5)이고, 시드 6개가 모두 베이스라인을 초과했습니다. 개선폭은 작지만 통계적으로 의미가 있습니다.

처음에는 시드 3개로 +0.00142, t=4.1이라 더 낙관적인 값이 나왔습니다. 시드를 6개로 늘리니 +0.00117, t≈2.9로 내려갔고, 이게 더 정직한 값입니다.

## 방법 요약

인코더는 4-layer 트랜잭션 트랜스포머(d=128, 약 87만 파라미터)입니다. SSL 목적함수 4종(마스킹 복원, 다음스텝 예측, 대조 학습, 혼합)으로 트랜잭션 시퀀스를 사전학습합니다. 평가는 3가지 프로토콜로 했습니다(linear probe, full fine-tune, GBM 병합). 데이터는 고객 단위 80/10/10 split으로 나눴습니다.

두 가지 분해 결과가 병합이 무엇을 더하는지 설명합니다.

- **Ablation**: 상위 100개 hand 피처를 제거하면 test AMEX가 0.00592 하락합니다. 여기에 SSL 임베딩을 더하면 그중 0.00324를 회복합니다. 회복률 약 55%입니다. SSL이 전문가의 피처 엔지니어링을 raw 트랜잭션에서 절반가량 비지도로 재발견한다는 뜻입니다.
- **Segment**: 평균 개선폭은 균일하지 않습니다. GBM이 "안전"하다고 본 예측 하위 0~3분위(silent default 구간)에 이득이 +0.02~+0.03으로 집중됩니다. 프라임 고객 중 false negative는 실무에서 가장 비싼 실패 모드인데, SSL이 바로 그 구간을 잡아줍니다.

## 재현 방법

코드는 [github.com/HangilKim11/blog-research/tree/main/ssl-credit-risk](https://github.com/HangilKim11/blog-research/tree/main/ssl-credit-risk)에 있습니다.

```bash
uv sync

# 1. 데이터 다운로드 (Kaggle 자격증명 필요)
uv run python -m amex.data.kaggle_download --mode full

# 2. 시퀀스 빌드 → split → 피처 엔지니어링
uv run python -m amex.data.sequence_builder
uv run python -m amex.data.splits
uv run python -m amex.data.feature_engineering

# 3. GBM 베이스라인
uv run python -m amex.baseline.lgbm

# 4. SSL 사전학습 (4 objective) → 풀 파인튜닝
bash scripts/run_phase2_all.sh
bash scripts/run_phase3_finetune.sh

# 5. 임베딩 추출 → GBM 병합
#    (scripts/make_augmented_features.py → amex.baseline.lgbm)
```

Kaggle 자격증명(`~/.kaggle/kaggle.json`)이 필요하고, AMEX 대회 규정을 먼저 수락해야 합니다. W&B 로깅은 선택입니다(각 스크립트에 `--no-wandb`로 끌 수 있습니다).

## 재현 시간 / compute

단일 RTX 4070 Laptop(8GB) + Ryzen 9 7940HS 기준으로 약 20~22 GPU+CPU시간입니다. 디스크는 약 70GB가 필요하고, 클라우드 비용은 0원입니다(전부 로컬에서 돌아갑니다).

## 데이터 출처

AMEX Default Prediction(Kaggle, 2022)입니다. 고객 45만 9천 명 × 최대 13개월의 익명화된 월별 프로파일입니다.

| 데이터 | 규모 | 출처 |
|---|---|---|
| AMEX Default Prediction | 45만 9천 고객 × 최대 13개월 | [kaggle.com/competitions/amex-default-prediction](https://www.kaggle.com/competitions/amex-default-prediction) |

Kaggle 약관상 재배포가 불가능합니다. 직접 다운로드해야 합니다(위 `kaggle_download` 단계가 받습니다).

## 논문

- `reports/paper/paper.pdf` — 영문
- `reports/paper/paper_ko.pdf` — 한국어

## 라이선스 / 주의

코드는 자유롭게 사용하세요. 데이터는 Kaggle/AMEX 약관을 따릅니다(재배포 금지, 직접 다운로드). 본문의 수치는 AMEX 2022 공개 데이터 한 종에 대한 것입니다. +0.00117이라는 개선폭이 다른 신용 데이터로 그대로 일반화된다고 보장하지 않습니다.

# TabFM은 신용 대손에서 GBM을 이길까 — 공개 데이터 검증

> 구글 TabFM(제로샷 테이블 파운데이션 모델)이 신용 대손 예측에서 잘 튜닝한 GBM을 이길까? 공개 데이터(UCI 대만 신용카드 대손)로 겨뤄본 재현 코드입니다.
>
> Does Google's zero-shot TabFM beat a well-tuned GBM at credit-default prediction? Reproduction code, tested on public data (UCI Taiwan credit-card default).

전체 글: **[han-co.com/ko/blog/tabfm-credit](https://han-co.com/ko/blog/tabfm-credit)**

## 핵심 결과 (5-fold, class_weight 없이 자연 비율)

| Arm | ROC-AUC | PR-AUC | KS | ECE ↓ |
|---|---:|---:|---:|---:|
| GBM 튜닝 (LightGBM) | 0.789 | 0.566 | 0.443 | 0.010 |
| **TabFM 제로샷** | **0.785** | **0.558** | **0.441** | **0.022** |
| GBM 날것 | 0.779 | 0.554 | 0.429 | 0.013 |

- 잘 튜닝한 GBM이 TabFM 제로샷을 근소하게 앞섭니다(격차 0.4%p, 폴드 노이즈 안).
- 무노력끼리면 TabFM이 날것 GBM보다 위. 캘리브레이션은 대등.
- 이 데이터 천장이 ~0.79라 어느 쪽도 그 위로 못 올라갑니다.
- 결론: TabFM은 "이기는 모델"이 아니라 "노력 없이 근접하는 빠른 베이스라인".

## 실행

```bash
pip install -r requirements.txt
# TabFM (GPU + CUDA 필요)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git"
jupyter notebook tabfm_credit.ko.ipynb
```

- **데이터는 노트북이 자동으로 내려받습니다** (`ucimlrepo`, UCI id=350).
- **TabFM 모델은 포함하지 않습니다.** 실행 시 Hugging Face(`google/tabfm-1.0.0-pytorch`)에서 받아옵니다(VRAM ~6.5GB).
- TabFM arm은 CUDA GPU가 필요합니다. GBM만 볼 거면 노트북의 `ARMS`에서 `tabfm_zeroshot`을 빼세요.
- 8GB GPU 기준 설정(컨텍스트 1,000행). 16GB 이상이면 `CONTEXT_MAX`를 키우고 앙상블 프리셋을 쓰세요.

## 데이터 출처

| 데이터 | 규모 | 출처 |
|---|---|---|
| UCI Default of Credit Card Clients (대만, 2005) | 3만 건 / 다음 달 부도 | [UCI 350](https://archive.ics.uci.edu/dataset/350) |

## 노트북 구성

날것 GBM · 튜닝 GBM(Optuna) · TabFM 제로샷 세 arm을 층화 5-fold로 비교합니다. 판별(ROC-AUC·PR-AUC·KS) + 캘리브레이션(Brier·LogLoss·ECE). CatBoost·XGBoost·TabFM 앙상블까지 포함한 전체 6개 arm은 원 실험 코드(config 기반)에서 돌립니다.

## 라이선스 / 주의

코드는 자유 사용. 데이터는 UCI 공개. 결론은 단일 공개 데이터에 대한 것이며, 다른 대손 데이터나 시계열(out-of-time) 검증에선 다르게 나올 수 있습니다. TabFM 모델 가중치·구조는 구글 소유입니다(원 저장소 참조).

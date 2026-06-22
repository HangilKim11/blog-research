# 신용 한도 디바이어싱 — 공개 데이터 검증

> 한도를 높이면 부도가 늘어날까? 데이터는 정반대(한도↑ → 부도↓)를 보입니다. 이 역설을 순차 잔차화(DML형) 디바이어싱으로 다루고, 공개 데이터 3종으로 검증합니다.
>
> Does raising a credit limit increase default? Raw data says the opposite. This project debiases the paradox (DML-style residualization) and tests it across three public datasets.

원시 데이터의 "한도가 높을수록 부도율이 낮다"는 역설은 **선택 편향**(신용이 좋은 사람에게 높은 한도가 부여됨) 때문입니다. 이 저장소는 그 편향을 제거하는 방법을 구현하고, **언제 역설이 실제로 뒤집히는지**를 세 데이터로 규명합니다.

전체 글: **[han-co.com/ko/blog/credit-limit-debiasing](https://han-co.com/ko/blog/credit-limit-debiasing)**

## 핵심 결과

| 신용의 종류 | raw 한도-부도 | 디바이어싱 후 | 사례 |
|---|---|---|---|
| 미사용 리볼빙 한도 | 음 (역설) | ≈ 0 | UCI · Lending Club · Home Credit 카드 |
| 인출 여신 + 약한 선택편향 | 양 | 양 | Lending Club 대출금 |
| **인출 여신 + 강한 선택편향** | **음 (역설)** | **양 (뒤집힘)** | **Home Credit 본 대출** |

역설이 디바이어싱으로 뒤집히려면 ① 인출된 여신(실부담)과 ② 강한 선택편향이 동시에 충족돼야 합니다. (전제0: 대손 정의가 진짜 신용손실을 잡아야 함. Home Credit 카드의 `SK_DPD≥90`은 '방치된 소액 잔액'을 잡아 부호가 망가지는 함정.)

**결론:** 방법(디바이어싱)은 타당하고 이식 가능하지만, "한도↑→부도↑"는 보편 법칙이 아닙니다. 실무 포트폴리오에서 ① 전이율 `dBalance/dLimit`와 ② 대손 정의를 직접 점검해야 합니다.

## 실행

```bash
pip install -r requirements.txt
jupyter notebook credit_limit_debiasing.ko.ipynb
```

- **UCI** 데이터는 노트북이 자동으로 내려받습니다.
- **Lending Club · Home Credit**은 용량과 약관 때문에 포함하지 않았고, 노트북(또는 `download_data.py`)이 받습니다. Home Credit은 Kaggle 자격증명(`~/.kaggle/kaggle.json`)이 필요합니다.

## 데이터 출처

| 데이터 | 규모 | 출처 |
|---|---|---|
| UCI Default of Credit Card Clients (대만, 2005) | 3만 건 / 1개월 연체 | [UCI 350](https://archive.ics.uci.edu/dataset/350) |
| Lending Club 2007–2013 (만기 완료) | 23만 건 / charge-off | Lending Club 아카이브 |
| Home Credit (`credit_card_balance`, `application_train`) | 카드 약 10만 / 신청 약 30만 | Kaggle |

## 방법 요약

K-fold 교차적합(cross-fitting) 잔차화 + isotonic 캘리브레이션 + 잔차 가중 + 선형 2차 스테이지(DML). 한도·잔액·부도를 신용도 피처에서 잔차화해 한도→잔액→부도 경로를 분리하고, 반사실(counterfactual)로 한도 변화의 효과를 추정합니다. 약한 잔차 신호에는 GBM 대신 선형 모델을 써 과적합을 피합니다.

## 라이선스 / 주의

코드는 자유 사용. 데이터는 각 출처의 약관을 따릅니다(UCI 공개, Lending Club 아카이브 공개, Home Credit은 Kaggle 약관상 재배포 금지 — 직접 다운로드). 본문의 결론은 공개 데이터에 대한 것이며, 실무 데이터의 부호는 위 두 가지(전이율·대손 정의)로 직접 검증해야 합니다.

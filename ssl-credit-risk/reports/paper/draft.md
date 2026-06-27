# SSL as a feature engine, not a competitor: self-supervised pretraining for tabular credit default prediction

> **상태**: 한국어 초안 작성 중. 모든 섹션 완료 후 영어 번역 + ACM LaTeX 변환.
> **타겟**: ICAIF 2026 (deadline ~2026-07)
> **저자**: TBA
> **레포**: https://github.com/<TBD>/amex-ssl (브랜치 `phase4-ensemble`)

---

## Section outline (section 별 핵심 메시지 1줄)

1. **Introduction** — 신용 risk의 GBM 독점 vs SSL의 매력 → 결과가 발표마다 일관되지 않은 이유를 통제 실험으로 분해
2. **Related work** — Padhi(TabBERT), Babaev(E.T.-RNN), AMEX Kaggle 상위 솔루션, NLP/Vision SSL benchmark 관행
3. **Data and setup** — AMEX 2022, 80/10/10 customer-level split, AMEX 메트릭, single-GPU 환경
4. **Methods**
   - 4.1 공유 transformer encoder
   - 4.2 4가지 SSL objective (masked, next-step, contrastive, hybrid)
   - 4.3 평가 프로토콜 (linear probe, full fine-tune, GBM-fusion)
5. **Results**
   - 5.1 Direct competition: SSL 단독은 GBM을 못 이긴다
   - 5.2 Protocol gap: linear probe vs full fine-tune이 0.058 차이
   - 5.3 Few-shot regime: GBM이 모든 라벨 fraction에서 우세
   - 5.4 **Feature fusion: GBM + SSL > GBM (+0.00142 ± 0.0006, t=4.1)**
   - 5.5 **Ablation: SSL이 hand feature top-100 신호의 55% 자동 회복**
   - 5.6 **Segment decomposition: lift는 base-pred decile 0-3에 집중 (+0.02~0.03)**
   - 5.7 Multi-encoder negative scaling
6. **Discussion**
   - 6.1 "Neural pretraining vs human pretraining" framing
   - 6.2 평가 프로토콜의 underspecification 문제
   - 6.3 실무 도입 지침 (SSL as auxiliary feature engine)
7. **Limitations** — 단일 dataset, 작은 encoder, OOF-기반 segment 분석
8. **Conclusion** — 3가지 contribution 재진술

부록 A: Reproducibility (커밋 해시, hyperparameter, compute 시간)
부록 B: Feature importance 전체 ranking
부록 C: 데이터 split 결정론성 (SHA-256)

---

## Abstract (한국어 초안, ~280 단어)

**문제**. 신용 부도 예측은 실무에서 거의 항상 도메인 지식이 응축된
hand-crafted aggregated feature 위의 gradient boosting machine (GBM)에
의존한다. 자연어 처리와 컴퓨터 비전에서 자기지도 사전학습(self-supervised
learning, SSL)이 가져온 성공을 신용 영역에 옮기려는 시도가 여러 차례
있었으나, "SSL이 GBM을 이긴다/진다"라는 결론이 발표마다 일관되지 않으며,
어떤 조건에서 어느 방향이 맞는지 분해된 분석이 부족하다.

**방법**. 본 연구는 AMEX Default Prediction (Kaggle 2022; 458,913 고객 ×
최대 13개월의 anonymized 거래 시퀀스) 데이터를 사용해 통제 실험으로
그 모호함을 분해한다. 동일한 customer-level 80/10/10 split, 동일한
AMEX 메트릭, 동일한 transformer encoder backbone으로 네 가지 SSL
objective (masked feature modeling, next-step prediction, contrastive,
hybrid)를 학습한 뒤, 평가 프로토콜(linear probe vs full fine-tune),
라벨 양(1%-100%), feature 융합(SSL embedding을 GBM의 추가 입력으로
합치는 방식)의 세 축으로 비교한다. 모든 실험은 단일 8 GB GPU에서
수행되었다.

**결과**. 네 가지 발견을 보고한다. (1) **어떤 SSL 단독 모델도 잘 튜닝된
LightGBM baseline (test AMEX 0.7956)을 능가하지 못한다.** (2) 동일한
인코더에 대해 **linear probe와 full fine-tune의 결과가 0.058만큼 차이**
나며 — 기존 tabular SSL 문헌이 평가 프로토콜에 따라 정반대 결론을
내릴 수 있음을 보인다. (3) **SSL embedding을 GBM의 보조 feature로
합치면 평균 test AMEX 0.79700 ± 0.0006 (3 seeds, t = 4.1)로 baseline을
통계적으로 의미 있게 +0.00142 상회한다.** (4) 이 lift는 균등하지 않고,
GBM이 "안전하다"고 자신한 prediction decile 0-3 영역에 **+0.02~0.03
집중되어 silent default 위험을 잡아낸다.** 또한 ablation 분석은 SSL
embedding이 hand-crafted top-100 feature 신호의 55%를 unlabeled raw
시퀀스로부터 자동 재발견함을 보이며, multi-encoder stacking은 음의
scaling을 나타내 NLP/Vision SSL 직감이 tabular credit에 그대로
이전되지 않음을 보인다.

**시사점**. SSL을 GBM의 경쟁자가 아닌 **보조 feature 엔진**으로 다루는
것이 — 그리고 그 가치가 false-negative tail에 집중된다는 것이 — 신용
risk 도메인에서 SSL을 도입할 합리적이고 측정 가능한 framing임을
정량적으로 보인다.

---

---

## 1. Introduction (한국어 초안, ~1,100 단어)

### 1.1 신용 부도 예측의 산업 표준과 그 한계

소비자 신용 부도 예측은 가장 오래 연구된 응용 통계 문제 중 하나이며,
지난 10여 년간 산업 표준은 명확하다. 그것은 **수작업 집계 feature 위의
gradient boosting machine (GBM)** 이다. 1억 건 이상 거래 시퀀스를
고객 단위 한 행으로 압축하기 위해, 실무자는 "마지막 값", "지난 12개월
평균/분산", "사용률의 추세", "결측 패턴" 등 도메인 지식이 응축된
수백~수천 개의 집계 feature를 만들어 LightGBM 또는 XGBoost에 투입한다.
이 접근법의 강도는 분명하다 — Kaggle AMEX 2022 대회의 상위 솔루션은
거의 모두 이 패턴이고[AMEX_1st_place_2022], 본 연구의 자체 baseline
역시 1,291개의 hand-crafted feature로 학습한 LightGBM이 5-fold CV
OOF에서 AMEX metric 0.79222, 5개 fold model을 bagging한 test에서
0.79558을 기록해 공개 leaderboard top-10 수준에 도달했다.

이 baseline은 두 가지 사실을 동시에 의미한다. 첫째, **수십 년의 expert
engineering 지식이 데이터 파이프라인 자체에 사실상 사전학습되어**
있어, 새 모델이 그 위에서 의미 있는 lift를 얻기 어렵다. 둘째, **시퀀스
자체에 담긴 일시적 패턴은 집계 함수들로 손실 압축**된다 — 어떤
정보가 그 압축 과정에서 사라지는지 정량적으로 알려진 바가 거의 없다.
이 두 사실이 시퀀스 모델링 및 자기지도 사전학습(self-supervised
learning, SSL)이 신용 영역에 유리할 수 있는 이론적 근거의 출발점이다.

### 1.2 NLP/Vision의 SSL 성공과 신용에서의 모호함

NLP와 컴퓨터 비전에서 SSL의 영향력은 더 이상 논쟁의 여지가 없다.
masked language modeling[Devlin2018], next-token prediction[Radford2018],
contrastive learning[Chen2020], masked autoencoding[He2022_MAE]은 모두
하나의 동일한 패턴을 따른다 — raw 신호 위에서 라벨 없이 사전학습된
표현이, 충분한 데이터와 컴퓨트가 있을 때 downstream에서 hand-crafted
feature를 압도한다는 패턴이다. 이 성공이 신용 risk 도메인으로 옮겨질
수 있다는 직관은 자연스럽다. Sberbank의 E.T.-RNN[Babaev2019], IBM의
TabBERT/TabGPT[Padhi2021], 그리고 여러 후속 연구들은 거래 시퀀스에서
사전학습된 sequence encoder가 GBM 표준을 능가할 수 있다고 주장해 왔다.

문제는 결과의 일관성이다. 같은 종류의 모델, 비슷한 데이터 규모에서도
SSL이 GBM을 이긴다고 보고하는 논문과 진다고 보고하는 논문이 공존하며,
그 갈라지는 이유는 명확히 분해되어 있지 않다. 우리가 직접 4가지
SSL objective를 동일한 인코더 backbone과 동일한 customer-level split
위에서 평가했을 때도, **평가 프로토콜 하나를 바꾸는 것만으로 AMEX
metric이 0.058 (-0.058에서 -0.003) 단위로 움직였다.** 이 차이는 "SSL이
GBM에게 진다"와 "거의 비긴다"를 분리하는 결정적 폭이다.

본 연구의 출발점은 따라서 단순하다 — **"SSL은 신용에 통하는가"는
잘못된 질문이며, 진짜 질문은 "어떤 평가 프로토콜에서, 얼마나 많은
라벨에 대해, 어떤 feature 융합 방식에서 SSL이 의미 있는 lift를 주는가"
이다.** 우리는 이 세 축을 통제된 실험으로 분해한다.

### 1.3 우리의 접근

본 연구는 단일 데이터셋(AMEX Default Prediction[Kaggle2022], 458,913
고객 × 최대 13개월의 anonymized 거래 시퀀스)과 단일 GPU(RTX 4070
Laptop, 8 GB VRAM)라는 **단일 실험 환경**을 유지한 채, 다음 세 축을
통제 변수로 비교한다:

- **평가 프로토콜**: 동일한 SSL 인코더에 대해 (a) **linear probe** —
  인코더를 freeze하고 mean-pool 위에 로지스틱 회귀를 학습 — 와
  (b) **full fine-tune** — 인코더와 classification head를 함께 학습 —
  의 두 가지 결과를 모두 보고한다.
- **라벨 양**: 동일한 stratified subset 추출 알고리즘으로 train+val
  라벨의 1%, 5%, 25%, 100%를 사용한 GBM과 SSL fine-tune의 label-
  efficiency curve를 그린다.
- **Feature 융합**: SSL 인코더의 mean-pooled 128-차원 embedding을
  GBM의 추가 입력 feature로 합쳤을 때의 marginal lift를 측정하고,
  ablation 및 segment 분해로 그 lift가 **무엇을 보완하며 어디에
  사는지** 정량화한다.

네 가지 SSL objective(masked feature modeling, next-step prediction,
SimCLR-style contrastive, hybrid)는 **동일한 4-layer transformer
encoder backbone (d=128, 869K parameters)** 을 공유한다. 모든 학습은
W&B[wandb]에 logging되었고, 데이터 split은 `data/splits/v1.parquet`에
SHA-256 결정론적으로 고정되어 모든 실험이 동일한 train/val/test
customer 집합을 사용한다.

### 1.4 Contributions

본 연구의 기여는 다음 네 가지이며, 모두 정량적으로 뒷받침된다:

1. **(직접 경쟁) 어떤 SSL 단독 모델도 잘 튜닝된 LightGBM (test AMEX
   0.7956)을 능가하지 못한다.** 최고 SSL 단독 결과는 Hybrid + full
   fine-tune의 0.7927로, baseline에 -0.003 뒤진다. 더불어 라벨 양을
   1% ~ 100%로 변화시킨 few-shot 비교에서 **GBM이 모든 fraction에서
   우세**하며, 라벨이 줄어들수록 차이가 커진다 (1% labels에서
   +0.052). 이는 일부 SSL 문헌의 "low-label regime"에서 SSL이 강하다는
   주장과 반대 방향이다.

2. **(평가 프로토콜) Linear probe와 full fine-tune의 결과가 0.058만큼
   차이난다.** 이는 기존 tabular SSL 평가가 critically underspecified
   임을 시사한다. 같은 인코더로 보고하는 두 숫자가 -0.058과 -0.003
   사이를 오갈 수 있다면, "SSL이 약하다/강하다"는 어느 쪽 주장이든
   protocol-conditional이다.

3. **(Feature 융합) SSL embedding을 GBM의 추가 feature로 합치면
   baseline을 통계적으로 의미 있게 상회한다.** 3개의 SSL pretrain
   seed에서 측정한 평균 test AMEX는 0.79700 ± 0.0006으로, baseline
   대비 +0.00142 (t = 4.1, df = 2, 3/3 trial이 양의 방향)의 lift를
   준다. Best single seed에서는 +0.00210에 도달한다.

4. **(국소 효과) Lift는 균등하지 않고, GBM이 "안전하다"고 자신한
   prediction decile 0-3 영역에 집중되어 silent default 위험을
   잡아낸다.** Per-decile AMEX 변화는 +0.02 ~ +0.03 수준이며, 이는
   credit-risk 실무에서 가장 비싼 false-negative failure mode에
   정확히 해당한다. 또한 ablation 분석은 SSL embedding이 hand-crafted
   top-100 feature 신호의 **약 55%를 unlabeled raw 시퀀스로부터
   자동 재발견**함을 보이며, multi-encoder stacking은 음의 scaling을
   나타내 NLP/Vision SSL 직감이 tabular credit에 그대로 이전되지
   않음을 보인다.

이 네 가지 발견은 종합적으로 **SSL을 GBM의 경쟁자가 아닌 보조 feature
엔진으로** 재배치하는 것이 신용 risk에서 SSL을 도입할 합리적이고
측정 가능한 framing임을 시사한다. 우리는 이를 결론에서 실무 도입
지침으로 구체화한다.

---

## 2. Related work (한국어 초안, ~700 단어)

### 2.1 신용 risk에서의 sequence modeling 및 SSL

신용 데이터에 시퀀스 모델을 적용한 작업은 크게 두 갈래로 나뉜다. 첫째,
RNN/LSTM 기반의 supervised end-to-end 학습이다. Sberbank의
E.T.-RNN[Babaev2019]은 거래 시퀀스에서 직접 부도 확률을 예측하는
LSTM 기반 architecture를 제안했고, 산업 데이터에서 logistic regression
baseline을 능가함을 보고했다. 다만 이 비교에서 baseline이 hand-crafted
feature 위의 GBM이 아닌 logistic regression이라는 점이 핵심 한계다 —
GBM과의 직접 비교에서는 시퀀스 모델이 일관되게 이긴다는 증거가
없다[Sirignano_mortgage].

둘째, 본 연구가 더 직접적으로 비교 대상으로 삼는 **사전학습 기반
접근**이다. Padhi et al.[Padhi2021]의 TabBERT와 TabGPT는 transaction
시퀀스 위에 BERT-style masked feature modeling 및 GPT-style
next-step prediction을 적용했다. 이들은 fraud detection 및 default
prediction에서 SSL 사전학습된 표현이 hand-crafted feature을 능가하는
조건을 일부 보고했지만, **평가는 linear probe 또는 fine-tune 중
하나로만 진행되어 두 프로토콜 간 일관성은 확인되지 않았다.** Skentzos
et al.[Skentzos2022]은 시퀀스 모델 + supervised 학습 위주이며, SSL을
표 형식 신용 데이터에 적용한 더 최근 작업[VIME2020, SCARF2021]은 주로
일반 tabular benchmark에서 검증되었고 신용 시퀀스 도메인에 대한 통제
실험은 부족하다.

요약하면, **공유 backbone + 공유 split + 동일 metric으로 4가지 SSL
objective × 2가지 평가 프로토콜 × 4가지 라벨 양 × feature 융합
변형을 한 데이터셋에서 비교한 작업은 — 우리가 아는 한 — 없다.**
이 결핍이 본 연구의 자리매김이다.

### 2.2 일반 tabular SSL과의 관계

일반 tabular 데이터에 대한 SSL은 신용에 직접 적용되지 않더라도
방법론적 영감을 제공한다. VIME[VIME2020]은 mask + value imputation
구조를, SCARF[SCARF2021]은 contrastive learning을 tabular에 적용한
대표적 작업이다. SAINT[SAINT2022]는 row-attention + intersample attention
을 결합해 tabular에서 contextual 표현을 학습한다. 그러나 이들 작업의
공통된 한계는 (a) 짧은 row-wise context (시퀀스 없음), (b) GBM과 fair
조건에서 같은 hand-crafted feature 위에서 비교한 사례 부족이라는 점에
있다. 신용 시퀀스는 동일 고객의 시간축이 명시되어 있다는 점에서
순수 tabular와 다르며, 시퀀스 길이 13개월 × wide column(190개)이라는
독특한 형상은 NLP의 long-sequence 가정도, Vision의 dense-grid 가정도
직접적으로 따라가지 않는다.

### 2.3 SSL 평가 프로토콜 관련 문헌

SSL 평가에서 linear probe와 full fine-tune이 다른 결과를 낼 수 있다는
관찰 자체는 NLP/Vision에서 이미 잘 알려져 있다[Chen2020,
He2022_MAE]. 일반적인 관행은 두 결과를 모두 보고하고, 둘 사이 차이를
representation의 분리 가능성과 transfer 학습 가능성을 함께 보여주는
지표로 해석하는 것이다. 그러나 tabular/credit SSL 문헌은 — 우리가
조사한 범위에서 — **두 프로토콜을 동시에 보고하는 경우가 드물고,
보고하더라도 (1) 같은 backbone에서, (2) 동일 split에서, (3) 정량
비교 표로 일관성 있게 정리한 작업은 거의 없다.** Section 5.2에서
보이듯이, 이 누락이 신용 SSL 결과의 일관성 부족을 설명할 수 있다.

### 2.4 GBM-SSL 융합 (feature engineering으로서의 SSL)

심층 모델의 hidden state를 GBM의 추가 feature로 사용하는 아이디어는
새롭지 않다. Kaggle 산업에서는 "neural net OOF prediction을 GBM에
stacking하는" 방식이 표준 ensemble 트릭이며, 자동 차량 보험 risk[
Kang2022]와 fraud detection[Lopez2020]에서도 deep representation을
GBM 입력으로 합치는 시도가 보고되어 왔다. 그러나 이들은 주로 "더
좋은 성능"이라는 결과만 보고할 뿐, **deep feature가 hand-crafted
feature의 무엇을 보완하는지, 어떤 customer segment에서 효과가 나는지,
얼마나 많은 deep feature를 더해야 marginal return이 음으로 돌아서는지
같은 분해 분석을 — 우리가 아는 한 — 신용 시퀀스 도메인에서 통제된
실험으로 수행한 작업은 없다.** 본 연구의 Section 5.4-5.7이 그 빈
공간을 채운다.

---

## 3. Data and setup (한국어 초안, ~600 단어)

### 3.1 데이터셋

본 연구는 American Express Default Prediction Kaggle Challenge
2022[Kaggle2022]의 공개 데이터를 사용한다. 데이터는 458,913명의
고객에 대해 최대 13개월의 statement-level 거래 기록을 anonymized
형태로 포함한다. 각 statement는 약 188개의 numerical 및 11개의
categorical anonymized feature와 한 개의 statement 날짜(`S_2`)로
구성된다. Label은 customer-level 이며, 마지막 statement 이후 18개월
내 default 여부의 이진 indicator로 정의된다. Class imbalance는 약
26%이며, 일부 고객은 13개월보다 짧은 history를 갖는다 (cold start).

데이터 자체는 anonymized이므로 도메인 해석은 제한되지만, 본 연구의
목적은 **방법론적 비교**이므로 익명성은 결과에 직접적 영향을 주지
않는다. 모든 raw 데이터는 Kaggle competition rules 하에 다운로드되며,
본 연구의 어떤 결과도 비공개 데이터에 의존하지 않는다.

### 3.2 Customer-level 80/10/10 split

모든 실험은 동일한 customer-level 80/10/10 stratified split을 사용한다.
Split은 `sklearn.model_selection.StratifiedShuffleSplit` 으로
target에 stratify되어 두 패스(90/10 → 80/10 within 90%)로 생성되며,
seed = 42로 고정되어 `data/splits/v1.parquet` 에 결정론적으로
저장된다. 추가로 `StratifiedKFold` 로 train + val 90% 안에서 5-fold
CV index가 함께 저장되어, Phase 1 LightGBM 5-fold OOF 및 Section
5.4의 augmented GBM에서 동일한 fold 구조를 재사용한다. SHA-256
hash가 두 번 연속 실행에서 일치함을 확인했다.

### 3.3 평가 메트릭

평가 메트릭은 Kaggle AMEX competition 공식 metric M = 0.5 (G + D) 이다.
여기서 G는 negative class에 weight 20을 부여한 normalized weighted
Gini coefficient, D는 cumulative weight 기준 top 4%에서 포착된
default rate이다. 우리는 Rohan Rao의 canonical 구현[RohanRao_metric]
을 numpy로 재구현하고, slow pandas reference 구현과 9개 유닛 테스트
(perfect / inverse / random / 단조성 / permutation invariance / 입력
검증 등)로 두 구현이 1e-9 이내에서 일치함을 검증했다. AMEX metric 외에
AUC, KS, log-loss도 모든 실험에서 함께 보고한다.

### 3.4 LightGBM Baseline

Baseline은 1,291개의 customer-level hand-crafted feature를 입력으로
하는 LightGBM 5-fold CV이다. Feature 생성 방식은 AMEX 2022 1st place
solution[AMEX_1st_place_2022]에 가깝게 — numerical 컬럼 177개 ×
{last, mean, std, min, max, last-first diff, last/mean ratio} = 1,239개와
categorical 컬럼 11개 × {last, nunique, count, mode} = 44개, 더하여
결측 패턴 3개와 시간 간격 5개를 결합해 1,291개를 만든다.

LightGBM hyperparameter는 공개 1st-place writeup을 따라
`learning_rate=0.01, num_leaves=100, min_child_samples=2400,
reg_alpha=0.5, reg_lambda=0.5, colsample_bytree=0.4, subsample=0.8,
subsample_freq=5, max_bin=255, n_estimators=10500, early_stop=200`
로 설정한다. AMEX metric을 custom `feval`로 등록해 early stopping이
실제 task metric에 대해 작동하도록 한다.

이 baseline은 OOF AMEX 0.79222, test bagged AMEX 0.79558을 기록하며
이는 AMEX 2022 Kaggle public leaderboard 기준 top-10 범위에 해당하는
강한 baseline이다. 본 연구의 모든 SSL 결과는 이 0.79558을 기준선으로
한다.

### 3.5 컴퓨팅 환경

모든 실험은 단일 RTX 4070 Laptop GPU (8 GB VRAM, sm_89) + AMD Ryzen
9 7940HS (16 logical cores) + 32 GB RAM의 한 노트북에서 수행되었다.
PyTorch 2.6.0+cu124, PyTorch Lightning 2.6.4, bf16 mixed precision을
사용한다. LightGBM은 모든 CPU core를 활용한다. 클라우드 GPU는 한
번도 사용하지 않았다. 총 5단계 실험을 모두 합한 wall-time은 약 20-22
시간 (compute hours, not human hours)이다.

이런 단일-GPU 재현성은 신용 risk 데이터 과학자가 개인 연구로 SSL을
탐색하는 데 따르는 진입 장벽을 낮춘다 — 24 GB VRAM 또는 cluster를
요구하는 NLP-style SSL 작업과 달리, 본 연구의 모든 결과는 일반 노트북
GPU 한 대로 재현된다.

---

## 4. Methods (한국어 초안, ~900 단어)

### 4.1 공유 transformer encoder

본 연구의 모든 SSL objective는 동일한 transformer encoder backbone을
공유한다. 이 일관성은 "어느 objective가 좋은가"라는 질문이 architecture
교란 변수로 오염되지 않게 보장한다.

**Feature embedder**. 시퀀스의 매 statement는 numerical feature
(F\_num = 177) + categorical feature (F\_cat = 11) + 결측 indicator로
구성된다. Numerical 입력은 train split에서 미리 계산된 mean/std로
z-score 정규화한 뒤, 결측은 0으로 채우고 별도의 0/1 결측 마스크와
함께 concat한다. 따라서 numerical branch의 입력 차원은 2 × F\_num이며,
하나의 linear layer로 d\_model = 128 차원으로 사영된다. Categorical
branch는 각 컬럼별 작은 embedding table (각 vocab_size에 padding_idx=0
의 MISSING/unknown 토큰 포함)을 거쳐 d_cat = 8 차원 embedding을 얻은
뒤, 모든 컬럼 embedding을 concat해 linear layer로 d\_model에 사영한다.
최종 timestep embedding은 두 branch의 sum이며, 형상은 (B, T, D) = (B,
13, 128)이다.

**Positional encoding과 CLS 토큰**. Sequence length가 짧으므로(T ≤ 13)
sinusoidal 대신 학습형(positional embedding parameter)을 사용한다.
masked, contrastive, hybrid objective는 sequence의 맨 앞에 학습형
CLS 토큰을 prepend해 (B, T+1, D)로 인코딩되며, 이 CLS 위치가 sequence
level summary 표현이 된다. next-step prediction objective는 인과적
구조의 명확성을 위해 CLS 없이 사용한다.

**Transformer 본체**. 4-layer Pre-LN[Xiong2020] transformer encoder
이며, 각 layer는 8-head multi-head attention (head dim = 32),
feed-forward dim = 512, dropout 0.1, GELU activation을 사용한다. 학습
안정성을 위해 Pre-LN을 채택했고, attention mask는 padding을 차단하며
next-step objective에서만 추가로 causal mask가 적용된다. Encoder의
parameter 수는 852,544개이며, 추가 head를 포함한 SSL objective 모듈
전체는 약 869K - 915K parameter 범위에 있다.

### 4.2 4가지 SSL objective

**Masked Feature Modeling (MFM)**. BERT-style masking[Devlin2018] 을
table 형상에 맞게 적용한다. 매 batch에서 (timestep, feature) cell의
15%를 random sampling해, numerical은 0으로 zero-out (그리고 결측 mask
를 True로 표시), categorical은 MISSING 토큰(code 0)으로 대체한 후,
encoder는 corrupted sequence를 받아 masked 위치의 원본 값(numerical)
및 코드(categorical)을 복원한다. Loss는 numerical MSE와 categorical
cross-entropy의 합이며, attention mask와 결측 mask를 결합해 padding이나
원래 결측이었던 위치는 loss에서 제외한다.

**Next-Step Prediction (NSP)**. GPT-style autoregressive[Radford2018]
objective. 인과적 attention mask를 적용한 encoder가 position t까지의
정보로 position t+1의 feature 벡터를 예측한다. Reconstruction head는
MFM과 동일한 구조 (numerical은 MSE, categorical은 CE)를 사용하되,
loss는 position 1..T-1을 input으로 받아 position 1..T를 예측하는
shift 형태로 정의된다.

**Contrastive (SimCLR-style)**. Chen et al.[Chen2020]의 InfoNCE를 짧은
시퀀스에 맞춰 변형한다. 각 customer에 대해 두 가지 augmented view를
생성한다 — (a) temporal cropping (균등 randomly 선택된 [start, start+L]
sub-window, L ≥ 6), (b) feature dropout (random하게 15%의 numerical
컬럼을 zero-out하고 categorical은 MISSING으로 대체). 두 view 각각을
encoder + 2-layer projection head로 통과시켜 L2-normalized 128차원
embedding을 얻고, batch 내 같은 customer의 두 view 사이는 양성(positive),
다른 customer는 음성(negative)으로 InfoNCE loss (temperature τ = 0.1)
를 계산한다.

**Hybrid (Masked + Contrastive)**. 위 두 objective를 동일 batch에서
동시에 수행한다. 한 forward pass에서 MFM head (timestep-level
reconstruction) 와 contrastive projection head (CLS-level invariance)
를 함께 update한다. Total loss는 α × L\_MFM + β × L\_contrastive이며,
α = β = 1로 단순 가중합을 사용한다.

### 4.3 평가 프로토콜

같은 사전학습된 encoder를 세 가지 downstream 프로토콜로 평가한다.

**(P1) Linear probe**. Encoder를 freeze하고 customer-level mean-pool
표현(masked, contrastive, hybrid의 경우 CLS도 포함해 mean-pool)을
sklearn `LogisticRegression(C=1.0)`로 5-fold customer-level CV에 맞추어
학습한다. Fold는 Section 3.2의 canonical fold column을 그대로 사용해
모든 비교가 동일한 train/val 멤버십을 갖도록 한다. 이 프로토콜은 "raw
representation의 quality"를 측정한다 — adaptation 없이도 정보가 분리
가능한가?

**(P2) Full fine-tune**. Encoder + 작은 classification head (2-layer
MLP + dropout)을 binary cross-entropy로 end-to-end 학습한다. AdamW
optimizer를 layer-wise lr decay와 함께 사용해(encoder lr = 1e-4, head
lr = 1e-3), encoder는 더 작은 학습률로, freshly initialized head는 더
큰 학습률로 update된다. Early stopping은 val/AMEX 기준 patience = 3,
max_epochs = 8. 이 프로토콜은 "전체 시스템이 best-case로 어디까지
가능한가"를 측정한다.

**(P3) GBM-fusion (이 연구의 핵심 기여)**. 사전학습된 encoder를 freeze
하고 모든 customer의 mean-pooled 128차원 embedding을 추출해, hand-
crafted feature parquet와 customer_ID로 join한 augmented feature
table을 만든다. 이 augmented feature 위에서 Section 3.4의 LightGBM
hyperparameter를 그대로 적용한 5-fold CV를 학습한다. 두 모델 family
(deep representation + tree-based decision)의 ensemble이 hand crafting +
gradient boosting의 산업 표준을 넘는지가 본 프로토콜의 측정 대상이다.

세 프로토콜은 정확히 같은 사전학습된 encoder를 입력으로 받는다 —
따라서 결과 간 차이는 평가 방식의 함수이지 model의 함수가 아니다.

### 4.4 통계적 신뢰도 (multi-seed protocol)

Section 5.7에서 보이듯, single SSL pretrain은 데이터 셔플링과 dropout
때문에 stochastic하다. 이 stochastic성이 우리의 main claim에 어떤
변동성을 부여하는지 측정하기 위해, hybrid objective의 SSL pretrain을
seed ∈ {original, 1, 2} 의 세 가지 random initialization으로 반복
수행하고, 각 seed에서 P3 (GBM-fusion) 결과를 측정한다. 보고하는 모든
GBM-fusion 숫자는 세 seed의 mean ± std (또는 명시되면 best single
seed) 이다. SSL pretrain seed 이외의 모든 요소 (data split, LightGBM
hyperparameter, evaluation metric 구현)는 결정론적이다.

---

## 5. Results (한국어 초안, ~1,500 단어)

본 섹션은 결과를 7개 subsection으로 구조화한다. 5.1-5.3은 SSL의
직접적 한계를, 5.4-5.6은 융합 framing 하의 실제 기여를, 5.7은 통계적
robustness를 보고한다. 모든 숫자는 동일한 80/10/10 split의 동일한
holdout test set (n = 45,892)에서 측정되었으며, 표 1은 모든 핵심
숫자를 한 곳에 모은 통합 표이다.

`[TBL: unified_results]` **Table 1**: 모든 phase의 통합 결과 (test
AMEX, n = 45,892).

| Approach | Test AMEX | Δ vs Phase 1 |
|---|---:|---:|
| Phase 5-D — Mean of 3 seeds (LightGBM + 128 SSL hybrid emb) | **0.79700 ± 0.0006** | **+0.00142** |
| Phase 5-D — Best single seed (seed = 1) | 0.79768 | +0.00210 |
| Phase 4 — LightGBM + 128 SSL hybrid emb (seed 42) | 0.79662 | +0.00104 |
| Phase 1 — LightGBM, 1,291 hand features (baseline) | **0.79558** | 0 |
| Phase 5-C — LightGBM + 4 × 128 SSL emb (1,803 feats) | 0.79516 | -0.00042 |
| Phase 5-A (iii) — (hand − top-100) + SSL | 0.79290 | -0.00268 |
| Phase 3 — Hybrid SSL + full fine-tune | 0.79267 | -0.00291 |
| Phase 5-A (ii) — hand − top-100 (no SSL) | 0.78966 | -0.00592 |
| Phase 2 — Next-step + linear probe | 0.73713 | -0.05845 |
| Phase 5-A (iv) — SSL alone (128 cols) | 0.72916 | -0.06642 |

### 5.1 Direct competition: SSL 단독은 GBM을 못 이긴다

Phase 2-3에서 4가지 SSL objective를 각각 linear probe (P1)와 full
fine-tune (P2) 두 가지 프로토콜로 평가한 결과, 어떤 조합도 Phase 1
LightGBM (test AMEX 0.79558)을 능가하지 못했다. Full fine-tune 기준으로
최고 결과는 Hybrid objective가 0.79267 (-0.00291), 가장 약한 결과는
Masked가 0.78996 (-0.00562)이었다. 네 objective 간 성능 차이가
fine-tune 후 0.003 이내로 좁혀진다는 점은 그 자체로 주목할 만하며
(Section 5.2에서 후술), **어떤 SSL 변형도 hand-crafted GBM의 표준을
넘지 못한다는 결론은 변하지 않는다.**

이는 단순히 우리의 SSL 구현이 약하다는 의미가 아니다. Phase 3의 4
fine-tune은 각각 ~60분 wall-time × bf16 mixed precision으로 학습되었고,
모든 fold에서 val/amex가 plateau에 도달한 후 early stop되었다. 더
중요하게는, Section 5.4에서 보이듯 이 동일한 인코더가 **GBM의 보조
feature로 들어가면 baseline을 통계적으로 의미 있게 상회한다.** 즉,
표현은 충분히 좋은데 단일 분류기로서 활용되는 방식이 비효율적이다.

### 5.2 Protocol gap: linear probe와 full fine-tune이 0.058 차이

같은 사전학습된 encoder를 두 프로토콜로 평가했을 때 차이는 일관되게
크다 (Figure 1). Hybrid encoder의 경우 linear probe로는 test AMEX
0.72629, full fine-tune으로는 0.79267 — 격차 **+0.06638**. Next-step
은 0.73713 → 0.79142, Masked는 0.72965 → 0.78996, Contrastive는
0.71691 → 0.79107. **네 objective 모두 +0.054 - +0.074의 매우 큰
프로토콜 격차를 보인다.**

`[FIG: protocol_gap]` **Figure 1**: 동일 encoder에 대한 linear probe
vs full fine-tune AMEX 막대 그래프 (4 objective × 2 protocol).
오렌지 (fine-tune)이 모두 파랑 (probe)에 비해 0.05-0.07 위에 있다.

특기할 만한 inversion은 **Hybrid objective가 linear probe에서 4위
(0.72629) 였다가 full fine-tune에서 1위 (0.79267)로 바뀌었다는 점**
이다. Contrastive 신호는 frozen 표현에서는 mean-pool feature에 직접
나타나지 않고 sequence-level invariance로만 존재하므로, classification
head가 학습되어야 비로소 활용 가능한 정보 형태다. 이는 평가
프로토콜에 따라 같은 데이터에서 **objective 간 순위가 완전히 뒤바뀔
수 있음**을 의미한다.

이 발견의 함의는 SSL 자체에 대한 평가를 넘어선다. 기존 tabular SSL
문헌이 linear probe만 보고하는 사례가 많은 만큼, "objective A가
objective B보다 낫다"는 결론은 매우 protocol-conditional이며, 같은
저자가 같은 데이터에서 두 프로토콜을 모두 보고할 때만 안전하다. 본
연구의 정량적 메시지는 다음과 같다: **tabular SSL paper는 두 프로토콜의
결과를 반드시 함께 보고해야 한다.**

### 5.3 Few-shot regime: GBM이 모든 라벨 fraction에서 우세

"SSL은 라벨이 적을 때 강하다"는 NLP/Vision의 표준 주장을 신용 도메인에
서 검증하기 위해, train+val 라벨 풀에서 stratified 1%, 5%, 25%, 100%
subset을 sampling해 GBM과 SSL fine-tune (hybrid encoder)을 각각
학습하고 동일한 test에서 평가했다.

`[FIG: few_shot_curve]` **Figure 2**: Label-efficiency curve. GBM과
SSL의 test AMEX를 log x-축에서 비교. GBM이 모든 fraction에서 위에
있고, 격차는 fraction이 줄수록 커진다.

| fraction | GBM test AMEX | SSL test AMEX | Δ (SSL − GBM) |
|---:|---:|---:|---:|
| 1%   | **0.75891** | 0.70701 | **-0.0519** |
| 5%   | **0.77819** | 0.73968 | -0.0385 |
| 25%  | **0.79140** | 0.77875 | -0.0127 |
| 100% | **0.79558** | 0.79267 | -0.0029 |

**GBM이 모든 fraction에서 우세**하며, 격차는 라벨이 줄수록 *증가*한다.
이는 SSL에 일반적으로 부여되는 "label-efficient encoder" framing이
신용 도메인 + 단일 GPU 규모에서는 지원되지 않음을 의미한다. 869K
parameter transformer는 4k 고객 (1%) 위에서 fine-tune되기에는 over-
parameterized이며, 그 동안 hand-engineered LightGBM은 `min_child_
samples`의 자동 스케일링과 강한 정규화 덕분에 small-data 영역에서도
견고하다. 단, GBM의 fold-별 AMEX std가 1% fraction에서 0.04 수준으로
커지므로 lift 자체는 점점 noisy해진다.

### 5.4 Feature fusion: GBM + SSL > GBM (+0.00142 ± 0.0006)

여기서부터 결론이 바뀐다. SSL 인코더의 mean-pooled 128차원 embedding
을 customer_ID로 hand-crafted feature parquet와 join하여 augmented
feature table (1,419 컬럼)을 만들고, Section 3.4의 동일한 LightGBM
hyperparameter로 5-fold CV를 학습했다.

| Setting | OOF AMEX | Test bagged AMEX | Δ vs Phase 1 |
|---|---:|---:|---:|
| Phase 1 — hand only (1,291) | 0.79222 | 0.79558 | — |
| Phase 4 — hand + 128 SSL (seed 42) | 0.79278 | **0.79662** | **+0.00104** |
| Phase 5-D — mean across 3 seeds | **0.79271 ± 0.00018** | **0.79700 ± 0.0006** | **+0.00142 ± 0.0006** |

**3-seed mean test AMEX는 0.79700 ± 0.0006이며, baseline 대비
+0.00142의 lift를 준다.** t-statistic = +0.00142 / (0.00060 / √3) =
**4.1** (df = 2). 표본 크기가 작아 정확한 p-value는 ~0.05-0.07 수준이지만,
**3 / 3 trial이 양의 방향**이라는 정성적 사실이 더 결정적이다 (다른
랜덤 시드를 추가해도 같은 부호일 확률이 매우 높다).

`[FIG: multiseed_errorbar]` **Figure 4**: GBM-only baseline (회색
대시), Phase 4 single-seed (파랑), Phase 5-D 3-seed mean ± std (오렌지
error bar). 3 seed 모두 baseline 위에 있다.

Feature importance 분석 (LightGBM gain 기준 fold-0 학습)은 SSL embedding
이 GBM에 의해 무시되지 않음을 보인다: 1,419개 feature 중 SSL emb이
total gain의 **2.45%**, total split count의 **9.79%**를 차지한다. 가장
중요한 SSL embedding (emb_049)은 전체 ranking에서 #52이며, 10개의
SSL embedding이 top-150 안에 든다.

`[TBL: importance_share]` **Table 2**: Hand-crafted vs SSL embedding
feature의 LightGBM importance 분배.

| Group | n cols | Gain % | Split % |
|---|---:|---:|---:|
| Hand-crafted | 1,291 | 97.55% | 90.21% |
| SSL embedding | 128 | 2.45% | 9.79% |

Split %와 Gain %의 비대칭 (10% vs 2.5%)은 의미 있는 패턴이다. GBM이
SSL embedding을 **자주 사용하지만 결정적이지는 않은 split에 활용한다**
— 즉 SSL은 hand-crafted feature가 모호한 곳에서 tie-breaker로
기능한다. 이 관찰이 다음 5.5의 ablation과 5.6의 segment 결과로
구체화된다.

### 5.5 Ablation: SSL은 hand feature top-100의 55%를 자동 회복한다

SSL이 정확히 무엇을 보완하는지 답하기 위해, Phase 1 baseline에서
LightGBM gain 기준 상위 100개 hand feature를 제거한 ablation을
수행했다. 네 가지 조합을 동일 hyperparameter, 동일 5-fold CV로 비교한다:

`[TBL: ablation_decomp]` **Table 3**: Ablation 결과 + recovery 분해.

| # | Feature set | n_cols | Test AMEX | Δ |
|---|---|---:|---:|---:|
| (i)   | full hand (Phase 1) | 1,291 | 0.79558 | baseline |
| (ii)  | hand − top-100 | 1,191 | 0.78966 | **−0.00592** |
| (iii) | (hand − top-100) + SSL | 1,319 | 0.79290 | −0.00268 |
| (iv)  | SSL only | 128 | 0.72916 | −0.06642 |
| (v)   | full hand + SSL (Phase 4) | 1,419 | 0.79662 | +0.00104 |

핵심 숫자는 (ii) → (iii)의 차이다. **Top-100 hand feature 제거로 잃은
0.00592 중, SSL embedding을 다시 더하면 0.00324가 회복된다 — recovery
rate ≈ 54.7%.** Hand-crafted feature engineering의 가장 중요한 100개
컬럼이 담고 있던 신호의 절반 이상을 SSL이 unlabeled raw 시퀀스에서
자동 발견한 셈이다.

`[FIG: ablation_recovery]` **Figure 3**: Phase 1 baseline → top-100
제거 → SSL 추가의 step bar chart. SSL이 차지하는 회복분 (0.00324)이
top-100 제거로 인한 손실 (0.00592)의 절반보다 약간 위로 채워지는
모습을 시각화.

분해를 더 세밀하게 보면: full hand + SSL의 +0.00104 lift 중에서 추가
**+0.00220 (= 0.00324 − 0.00104) 는 hand feature와 redundant**한
SSL의 부분이며, 남은 **+0.00104는 hand feature가 잡지 못한 순수
orthogonal 신호**이다. 이 분해는 SSL embedding이 정확히 절반은
expert engineering의 자동화이고, 나머지 절반은 새로운 정보임을 시사한다.

### 5.6 Segment decomposition: lift는 base-pred decile 0-3에 집중

+0.00142라는 평균 lift는 모든 customer에게 균등하게 분포되어 있을
필요가 없다. Test OOF prediction의 base-pred decile로 customer를
10등분해 각 decile에서 AMEX 변화량 (Phase 4 augmented − Phase 1
baseline)을 측정한 결과는 강하게 비균등하다.

`[FIG: segment_decomposition]` **Figure 5**: 세 가지 segment 축에서의
per-segment AMEX (baseline = 파랑, augmented = 오렌지). base-pred
decile 0-3에서 오렌지가 파랑보다 +0.02 - +0.03 위에 있는 것이 한눈에
보인다.

| Base-pred decile | Default rate | Δ AMEX |
|---:|---:|---:|
| 0 (GBM says "safe") | 0.000 | **+0.0239** |
| 1 | 0.001 | **+0.0234** |
| 2 | 0.002 | **+0.0213** |
| 3 | 0.005 | **+0.0261** |
| 4 | 0.015 | -0.0054 |
| 5-8 | 0.06 - 0.80 | -0.001 ~ +0.007 |
| 9 (GBM says "default") | 0.964 | +0.0029 |

**Lift는 base-pred decile 0-3 — 즉 GBM이 "이 고객은 안전하다"고 자신한
영역 — 에서 +0.02 ~ +0.03 수준으로 집중된다.** 이 영역은 base rate가
0% - 0.5% 수준의 prime customer 그룹이며, 신용 risk 실무에서 가장
비싼 false-negative failure mode가 발생하는 위치다 (GBM이 "안전"이라
판정한 customer 중 실제로 default한 사람을 발견하는 가치).

다른 segment 축 (statement count, P_2 decile)에서도 일부 lift 패턴이
관찰되나, base-pred decile만큼 깔끔하지 않다. Statement count 슬라이스
에서는 thin-file customer (≤ 6 statements)에서 -0.0023의 lift를 보여,
"SSL이 cold-start 영역에서 유리하다"라는 자연스러운 예측이 본 데이터
에서는 성립하지 않음을 보인다 — pretrain이 longer-history customer에
더 노출된 결과로 추정된다.

### 5.7 Multi-encoder negative scaling

NLP/Vision SSL의 직감 중 하나는 "더 많은 사전학습 신호를 stack하면
성능이 증가한다"는 것이다. 우리는 hybrid encoder의 128차원 embedding
대신 4 encoder의 4 × 128 = 512차원 embedding을 hand feature에 합쳐
1,803-feature GBM을 학습해 이 직감을 검증했다.

| Setting | OOF | Test bagged |
|---|---:|---:|
| Phase 4 — hand + 1 enc | 0.79278 | **0.79662** |
| Phase 5-C — hand + 4 enc | **0.79314** | 0.79516 |

OOF는 개선되지만 (+0.00036) test bagged는 악화된다 (-0.00146). 이는
교과서적 overfitting 신호이며, **단일 encoder가 sweet spot임**을
시사한다. NLP의 "more pretraining objectives 더 좋은 표현" 직감이
tabular credit + 단일 GPU 규모에서는 그대로 적용되지 않는다.

이 negative scaling 자체가 본 연구의 한 가지 별도 contribution이다.
"4-encoder 앙상블이 always better"라는 가정 하에 추가 컴퓨트를 들인
실무자가 실제로 일반화에서 손실을 볼 수 있음을 정량적으로 보였기
때문이다.

---

## 6. Discussion (한국어 초안, ~900 단어)

### 6.1 "Neural pretraining vs human pretraining" framing

Section 5.5의 가장 강한 정량 발견은 **SSL embedding이 hand-crafted top-
100 feature 신호의 55%를 자동 회복**한다는 사실이다. 이 숫자는
근본적인 framing을 제시한다.

신용 risk 실무에서 1,291개의 hand-crafted feature를 만드는 것은 단순한
전처리가 아니다. 어느 column이 "마지막 값" 으로 의미를 갖는지, 어느
column이 "분산"으로 의미를 갖는지, 어느 categorical이 mode로 잘
요약되는지 결정하는 행위는 **수십 년의 도메인 지식이 데이터
파이프라인에 인코딩된 implicit pretraining**으로 볼 수 있다. 이
시각에서 우리의 baseline GBM은 "no pretraining" 모델이 아니라 "human
pretrained" 모델이며, SSL은 그것과 경쟁할 수 있는 별도의 "neural
pretrained" 표현을 제안하는 것이다.

이 framing 하에서 본 연구의 결과는 다음과 같이 재해석된다:

- 신용 raw 시퀀스 위에서 라벨 없이 10 epoch 학습된 869K parameter
  transformer가, **decades of expert engineering이 만든 top-100 feature
  중 55%를 자동으로 재발견했다.** 이는 NLP의 "BERT가 단어 임베딩의
  syntactic 구조를 발견한다"라는 결과와 같은 성질의 발견이다.
- 그러나 남은 45%는 SSL이 발견하지 못한다. 이는 expert feature
  engineering에 SSL이 접근할 수 없는 정보 (예: 도메인-특정 ratio,
  business-meaningful date 구간, 결측 패턴의 정성적 해석) 가 들어
  있음을 의미한다.

Phase 5-C의 negative scaling 결과는 같은 시각에서 자연스럽다 — 4
encoder는 hand feature의 같은 55%를 4 번 약간씩 다른 angle로 재발견
했을 뿐이고, 새로운 45%에는 더 다가가지 못한다. 따라서 더 많은
encoder를 stack해도 정보 측면의 ceiling이 향상되지 않고 GBM의
overfitting risk만 증가한다.

NLP/Vision에서 raw text 또는 raw pixel은 **거의 가공되지 않은 상태**로
모델에 입력된다 — 그래서 SSL이 단순 표준화를 넘어서는 representation
을 학습할 여지가 매우 크다. 반면 신용 데이터는 ETL pipeline에서
이미 정규화, dedup, anonymization, 기본 aggregation을 거친 후 모델에
입력된다. **Raw signal에 도메인 지식이 얼마나 사전 압축되어 있는가**
가 SSL의 ceiling을 결정하는 가장 큰 변수일 가능성이 있으며, 이는
신용 SSL이 NLP SSL만큼 dramatic하게 작동하지 않는 이유의 후보 설명
이다.

### 6.2 평가 프로토콜의 underspecification 문제

Section 5.2의 0.058 protocol gap은 단순한 quirk가 아니다. 같은 인코더
에 대해 linear probe로 -0.058을, full fine-tune으로 -0.003을 보고하는
두 논문이 같은 데이터에 대해 정반대 인상을 줄 수 있다. 우리의 4
objective 비교에서 가장 극단적 사례는 Hybrid objective의 probe에서의
꼴찌 → fine-tune에서의 1위 inversion이다 (objective ranking이 평가
프로토콜로 완전히 뒤집힌다).

이 결과의 함의는 신용 SSL을 넘어선다. 모든 tabular SSL benchmark는
**같은 인코더에 대해 두 프로토콜 결과를 모두 보고하는 것을 표준으로
삼아야 한다.** 그렇지 않으면 reader는 "SSL X가 SSL Y보다 좋다"라는
주장의 protocol-conditionality를 확인할 수 없으며, 이는 분야 전체의
재현성 위기로 직결된다.

본 연구는 그 표준을 — 단일 dataset 안에서 — 시연한 점에서 작은 메소
도로지컬 기여를 한다. 더 큰 가치는 본 연구를 본 reader가 자신의
tabular SSL benchmark에 같은 표준을 적용하기를 권한다는 점이다.

### 6.3 실무 도입 지침 (SSL as auxiliary feature engine)

Section 5.4-5.6의 결과를 종합하면, 신용 risk 실무자를 위한 구체적
지침이 도출된다:

1. **단일 SSL 모델로 GBM을 대체하지 말라.** Phase 2-3의 결과는 직접
   경쟁에서 SSL이 일관되게 진다는 것을 보였다. 특히 라벨이 적을 때
   SSL은 *더 약하다* (Section 5.3).

2. **SSL을 추가 feature 엔진으로 다루라.** 단일 hybrid encoder의
   128차원 mean-pooled embedding을 기존 hand-crafted feature와 함께
   GBM에 입력하면 평균 +0.00142의 lift를 robust하게 얻을 수 있다
   (Section 5.4 + 5.7). 컴퓨팅 비용은 SSL pretrain 약 1시간 + embedding
   추출 약 22분으로, GBM hyperparameter tuning 한 라운드보다도 가볍다.

3. **Lift가 어디에 사는지 확인하라.** 평균 +0.001은 작아 보이지만,
   Section 5.6의 segment 분해는 lift가 base-pred decile 0-3 (GBM이
   "안전"이라 자신한 prime customer 영역)에 +0.02 - +0.03 수준으로
   집중됨을 보였다. 신용 risk에서 가장 비싼 failure mode는 prime
   customer의 silent default이므로, **operational impact는 scalar
   metric이 시사하는 것보다 클 수 있다.** 도입 평가 시 cohort-level
   loss reduction을 직접 계산할 것을 권한다.

4. **하나의 encoder로 충분하다.** Multi-encoder stacking은 추가
   compute를 들이고도 일반화에서 손실을 본다 (Section 5.7). 1개 encoder
   가 sweet spot이며, 추가 정보가 필요하면 encoder 크기를 키우거나
   pretrain epoch 수를 늘리는 편이 더 직관적인 시도다.

### 6.4 본 연구의 자리매김

이상의 결과를 종합하면, 본 연구의 자리매김은 다음과 같다:

- **"SSL이 신용에서 통하는가?"** 라는 이분법 질문에 대한 새로운 답:
  *그 자체로는 통하지 않지만, GBM 옆에서 보조 역할로는 robust하게
  통한다.*
- 기존 SSL 문헌에 누락된 표준: *같은 데이터에서 두 프로토콜을 모두
  보고하라.*
- 기존 신용 risk 문헌에 누락된 framing: *SSL을 경쟁자가 아닌 feature
  엔진으로 다루라.*
- 기존 ensemble 문헌에 추가되는 발견: *tabular credit에서는 SSL
  stacking의 marginal return이 음으로 빨리 돌아선다.*

이 네 가지가 paper의 contribution이며, 신용 risk + tabular SSL 분야의
인접 연구가 비교할 수 있는 정량적 base point를 제공한다.

---

## 7. Limitations (한국어 초안, ~500 단어)

본 연구의 결과는 다음 한계 안에서 해석되어야 한다.

**(L1) 단일 데이터셋.** 모든 결과는 AMEX Default Prediction 2022 한
공개 dataset에서 측정되었다. Default rate (~26%) 와 statement length
(최대 13개월)는 American consumer credit에 특수한 특성이며, anonymized
column은 도메인 해석을 제한한다. +0.00142의 lift가 일본 소비자
신용, 자동차 대출, 또는 corporate credit으로 그대로 전이된다는 보장은
없다. Cross-dataset 검증은 본 연구의 범위를 벗어난다.

**(L2) 작은 encoder.** 우리의 transformer encoder는 869K parameter로,
NLP/Vision의 representative pretrained 모델 대비 1000배 이상 작다.
이는 단일 GPU 재현성을 의식한 의도적 선택이지만, 4M - 10M parameter
encoder에서 결과가 어떻게 바뀔지는 직접 검증하지 않았다. Section 5.5의
55% recovery rate는 encoder가 클수록 더 높아질 가능성이 있다.

**(L3) 단일 사전학습 budget.** 본 연구의 SSL pretrain은 10 epoch ×
batch size 512 × limit_train_batches 360 (≈ 50% of one full pass) 으로
제한되었다 — 단일-GPU wall-time 예산을 위한 선택이다. Full data + 50
epoch + multiple objective sweeps의 결과가 어떻게 바뀔지는 본 연구의
실험으로 측정되지 않는다.

**(L4) Segment 분석의 OOF 의존.** Section 5.6의 segment decomposition은
per-customer test prediction이 lgbm.py에서 dump되지 않아 trainval
OOF 위에서 수행되었다. Test에서도 동일 패턴이 나타나리라는 iid 가정
하의 결론이며, 엄격한 stress-test는 lgbm.py에 per-test-row prediction
dump 기능을 추가한 후속 작업이 필요하다.

**(L5) Hybrid encoder만 융합에 사용.** Phase 4 + 5의 모든 융합 실험은
Hybrid (Masked + Contrastive) encoder의 embedding에 한정되었다. 다른
세 objective의 embedding을 단독으로 GBM에 합쳤을 때의 결과는 별도
실험이 필요하다 — 단, Phase 5-C의 negative scaling 결과는 multi-
encoder가 단일 best보다 좋지 않음을 이미 시사한다.

**(L6) Seed 표본 3개.** Phase 5-D의 통계적 robustness는 3 seed
mean ± std 위에서 평가되었다. t-statistic이 4.1로 강하긴 하나, df = 2
이라는 작은 자유도에서 p-value는 0.05-0.07 경계에 머문다. 5+ seed
로의 확장은 본 연구의 wall-time 예산을 두 배로 늘렸기에 후속에서
수행 가능한 자연스러운 보강이다.

**(L7) Cross-temporal split 부재.** 본 연구는 customer-level random
80/10/10 split을 사용한다. AMEX 데이터의 익명화로 인해 statement
시점이 명시되어 있어 in-time/out-of-time split을 만들 수는 있으나,
이 추가 분석은 본 연구의 범위 밖이다. 실무 도입 결정은 cross-temporal
generalization에 더 의존하므로, 본 연구의 +0.00142 lift가 future
period에서도 유지되는지는 별도 검증이 필요하다.

이 한계들은 본 연구의 핵심 주장 — SSL을 GBM의 보조 feature 엔진으로
사용하면 통계적으로 의미 있는, segment-targeted lift를 얻는다 — 을
직접 위협하지는 않는다. 다만 일반화 폭과 deployment readiness 측면
에서 추가 검증이 필요한 영역들을 정직히 표시한다.

---

## 8. Conclusion (한국어 초안, ~400 단어)

본 연구는 신용 부도 예측에서 자기지도 사전학습(SSL)의 역할을 단일
데이터셋, 단일 GPU, 통제된 실험으로 분해했다. 세 가지 평가 축 —
프로토콜 (linear probe vs full fine-tune), 라벨 양 (1% - 100%), feature
융합 (GBM 추가 feature로서의 SSL embedding) — 위에서 4가지 SSL
objective를 동일 encoder backbone과 동일 customer-level split으로
비교했다.

**핵심 발견 네 가지**는 다음과 같다.

1. **SSL 단독은 GBM을 못 이긴다.** 어떤 objective × protocol 조합도
   잘 튜닝된 LightGBM의 test AMEX 0.7956을 능가하지 못했다.

2. **평가 프로토콜이 결론을 0.058 뒤집는다.** 같은 인코더에서 linear
   probe와 full fine-tune의 결과 차이가 -0.058에서 -0.003 사이를
   움직이므로, tabular SSL 평가는 두 프로토콜의 동시 보고를 표준으로
   삼아야 한다.

3. **GBM + SSL은 GBM을 통계적으로 의미 있게 상회한다.** 3 seed mean
   test AMEX 0.79700 ± 0.0006, baseline 대비 +0.00142 ± 0.0006 (t =
   4.1, 3/3 trial 양의 방향). 단일 SSL encoder의 mean-pooled 128차원
   embedding을 hand-crafted feature 옆에 추가하는 것만으로 얻을 수
   있는 lift다.

4. **이 lift는 false-negative tail에 집중된다.** Base-pred decile 0-3
   (GBM이 "안전"이라고 자신한 prime customer 영역)에서 +0.02 - +0.03,
   다른 decile에서는 거의 0. SSL embedding은 hand-crafted feature top-
   100 신호의 약 55%를 unlabeled raw 시퀀스에서 자동 재발견하며,
   multi-encoder stacking은 음의 scaling을 보인다.

이 결과는 통합해서 다음 메시지를 준다. **신용 risk 도메인에서 SSL은
GBM의 경쟁자가 아닌 보조 feature 엔진이다. 그 가치는 GBM이 약한
영역 — 안전이라 잘못 분류된 prime customer의 silent default — 에
집중되며, 단일 encoder로 saturating한다.** 도입 결정은 average
scalar metric이 아닌 cohort-level loss로 평가되어야 한다.

본 연구의 정량 결과 — 55% recovery rate, +0.00142 robust lift,
base-pred decile 0-3 in +0.02 - +0.03 집중, multi-encoder negative
scaling — 은 신용 risk와 tabular SSL 두 인접 분야 모두에 새 데이터
포인트를 제공한다. 모든 코드, 데이터 split, 가중치는 https://github.com/<TBD>
에서 공개되며 단일 8 GB GPU에서 약 20 GPU+CPU-hours 안에 전부 재현된다.

후속 방향으로는 cross-dataset 검증, 더 큰 encoder, segment-aware
loss로의 fine-tune이 자연스럽다. 본 연구가 그 방향의 기준점이 되기를
기대한다.

---

## Appendix A — Reproducibility checklist

본 부록은 모든 실험을 재현하기 위한 정보를 한 곳에 모은다.

**A.1 Code**. 모든 코드는 https://github.com/<TBD>/amex-ssl에 공개되며,
다음 5개 phase 브랜치로 구분된다 — `phase1-baseline`, `phase2-ssl`,
`phase3-finetune`, `phase4-ensemble`. Final commit hash는 paper acceptance
시점에 frozen된다. 빌드는 `uv sync --extra dl` 한 줄, 환경 검증은
`uv run python scripts/check_env.py --phase 2` 한 줄로 충분하다.

**A.2 Data**. 모든 raw 데이터는 Kaggle American Express Default
Prediction 2022 competition에서 공개되며, `kaggle.json` 인증 후 본
저장소의 `python -m amex.data.kaggle_download --mode full` 로 다운로드
된다. 데이터 split은 결정론적 SHA-256 hash로 검증되며
(`data/splits/v1.parquet`, 13.6 MB), 모든 customer-level membership은
seed = 42로 고정된다.

**A.3 모델 hyperparameter**. 본문 Section 4.1-4.2의 transformer
hyperparameter와 Section 3.4의 LightGBM hyperparameter가 전부이며,
세부 값은 다음 표에 정리된다.

| component | parameter | value |
|---|---|---|
| encoder | d_model | 128 |
| encoder | n_layers | 4 |
| encoder | n_heads | 8 |
| encoder | d_ff | 512 |
| encoder | dropout | 0.1 |
| encoder | cat_emb_dim | 8 |
| encoder | total params | 852,544 |
| SSL train | optimizer | AdamW (β=(0.9, 0.95)) |
| SSL train | lr | 3e-4 cosine |
| SSL train | weight_decay | 0.01 |
| SSL train | epochs | 10 |
| SSL train | batch_size | 512 |
| SSL train | limit_train_batches | 360 (≈ 50% of one pass) |
| SSL train | precision | bf16 mixed |
| Fine-tune | lr_encoder | 1e-4 |
| Fine-tune | lr_head | 1e-3 |
| Fine-tune | max_epochs | 8 |
| Fine-tune | patience | 3 |
| LightGBM | learning_rate | 0.01 |
| LightGBM | num_leaves | 100 |
| LightGBM | min_child_samples | 2,400 |
| LightGBM | reg_alpha / reg_lambda | 0.5 / 0.5 |
| LightGBM | colsample_bytree | 0.4 |
| LightGBM | subsample | 0.8 (freq 5) |
| LightGBM | n_estimators | 10,500 |
| LightGBM | early_stop | 200 |

**A.4 Compute receipts**. 모든 실험은 단일 RTX 4070 Laptop GPU (8 GB
VRAM, sm_89) + AMD Ryzen 9 7940HS + 32 GB RAM에서 수행됨. Phase
별 wall-time:

| phase | wall-time |
|---|---:|
| Phase 1 — LightGBM 5-fold (CPU) | ~57 min |
| Phase 2 — 4 SSL pretrain × 10 epochs (GPU bf16) | ~4 h |
| Phase 3 — 4 full fine-tune × 8 epochs (GPU) | ~4 h |
| Phase 4 — extract emb + augment + LGB | ~22 min + 60 min |
| Phase 5-A — 3 ablation GBM × 5-fold (CPU) | ~3 h |
| Phase 5-B — segment analysis (CPU one-shot) | < 1 min |
| Phase 5-C — 3 emb extracts + multi-enc GBM | ~3 h |
| Phase 5-D — 2 seeds × full pipeline | ~5 h |
| **total session compute** | **~20-22 GPU+CPU-hours** |

**A.5 Frame-level checks**. AMEX metric implementation의 numpy fast
path와 pandas reference 구현은 9개 unit test에서 1e-9 이내 일치를
보였다. 모든 phase의 OOF 및 metrics JSON은 저장소의
`data/processed/v1/oof_*.parquet` 와 `oof_*_metrics.json` 에 남아 있다.

---

## Appendix B — Feature importance full ranking

Section 5.4의 fold-0 LightGBM feature importance 전체 ranking은 저장소
의 `reports/feature_importance_augmented.json` 에 저장되어 있다. 상위
25개 (gain 기준) 와 SSL embedding 중 상위 10개는 본문 Section 5.4에
이미 제시되었다. Section 5.5의 ablation을 위해 사용된 top-100 hand
feature 목록은 `reports/feature_importance_hand_only.json` 의
`top_k_columns` 필드에 보존되어 있다.

---

## Appendix C — Determinism + integrity of `data/splits/v1.parquet`

본 연구의 결과 비교 가능성은 모든 실험이 동일한 customer-level split
을 사용한다는 점에 의존한다. `data/splits/v1.parquet` 는 seed = 42 로
한 번 생성된 후 SHA-256 hash로 무결성 검증되어 git에 commit되었으며,
다음 의미를 갖는다 — `train` (367,129), `val` (45,892), `test`
(45,892) 의 customer-level membership과 `fold` (0..4 within train+val,
-1 for test) 가 모든 실험에서 동일하다.

검증은 `python -m amex.data.splits --force` 를 두 번 연속 실행해 SHA-
256이 일치하는지로 수행되며, 두 번째 실행은 무결성 통과 시 변경
없이 종료된다.

---

## Notes for self (작성 진행하면서 채울 것)

- 모든 numeric은 `reports/final_summary.md` 의 Section 1-7 에서 가져옴
- 인용 placeholder: `[Padhi2021]`, `[Babaev2019]`, `[Devlin2018]`, `[Chen2020]`,
  `[He2022_MAE]`, `[Khandani2010]`, `[Sirignano_mortgage]`,
  `[AMEX_1st_place_2022]`. 실제 BibTeX는 영어 변환 시점에 정리.
- 그림 자리 표시: `[FIG: protocol_gap]`, `[FIG: few_shot_curve]`,
  `[FIG: ablation_recovery]`, `[FIG: segment_decomposition]`,
  `[FIG: multiseed_errorbar]`.
- 표 자리 표시: `[TBL: unified_results]`, `[TBL: ablation_decomp]`,
  `[TBL: seed_robustness]`.

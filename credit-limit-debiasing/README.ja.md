# 限度枠バイアス除去 — 公開データで検証

> 限度枠を上げると貸し倒れは増えるのか? データは逆(限度枠↑ → 貸し倒れ↓)を示します。この逆説を逐次残差化(DML型)バイアス除去で扱い、公開データ3種で検証します。
>
> Does raising a credit limit increase default? Raw data says the opposite. This project debiases the paradox (DML-style residualization) and tests it across three public datasets.

生データの「限度枠が高いほど貸し倒れ率が低い」という逆説は、**選択バイアス**(信用の良い人に高い限度枠が付与される)によるものです。このリポジトリはそのバイアスを除く手法を実装し、**逆説が実際にいつ反転するのか**を3つのデータで解明します。

解説記事(韓国語): **[han-co.com/ko/blog/credit-limit-debiasing](https://han-co.com/ko/blog/credit-limit-debiasing)**

## 主な結果

| 与信の種類 | raw 限度枠-貸し倒れ | バイアス除去後 | 事例 |
|---|---|---|---|
| 未使用のリボ枠 | 負(逆説) | ≈ 0 | UCI · Lending Club · Home Credit カード |
| 引き出された与信 + 弱い選択バイアス | 正 | 正 | Lending Club ローン額 |
| **引き出された与信 + 強い選択バイアス** | **負(逆説)** | **正(反転)** | **Home Credit 個品割賦** |

逆説がバイアス除去で反転するには ① 引き出された与信(実負担)と ② 強い選択バイアスが同時に揃う必要があります。(前提0: 貸し倒れ定義が本物の信用損失を捉えていること。Home Credit カードの `SK_DPD≥90` は「放置された少額残高」を捉え、符号が壊れる落とし穴。)

**結論:** 手法(バイアス除去)は妥当で移植可能ですが、「限度枠↑→貸し倒れ↑」は普遍法則ではありません。実務ポートフォリオで ① パススルー `dBalance/dLimit` と ② 貸し倒れ定義を直接点検する必要があります。

## 実行

```bash
pip install -r requirements.txt
jupyter notebook credit_limit_debiasing.ja.ipynb
```

- **UCI** データはノートブックが自動でダウンロードします。
- **Lending Club · Home Credit** は容量と規約のため同梱しておらず、ノートブック(または `download_data.py`)が取得します。Home Credit は Kaggle の認証情報(`~/.kaggle/kaggle.json`)が必要です。

## データ出典

| データ | 規模 | 出典 |
|---|---|---|
| UCI Default of Credit Card Clients (台湾, 2005) | 3万件 / 1か月延滞 | [UCI 350](https://archive.ics.uci.edu/dataset/350) |
| Lending Club 2007–2013 (満期完了) | 23万件 / charge-off | Lending Club アーカイブ |
| Home Credit (`credit_card_balance`, `application_train`) | カード約10万 / 申込約30万 | Kaggle |

## 手法の要約

K-fold 交差適合(cross-fitting)残差化 + isotonic キャリブレーション + 残差加重 + 線形2次ステージ(DML)。限度枠・残高・貸し倒れを信用度の特徴量から残差化して 限度枠→残高→貸し倒れ の経路を分離し、反事実(counterfactual)で限度枠変化の効果を推定します。弱い残差シグナルには GBM ではなく線形モデルを使い、過学習を避けます。

## ライセンス / 注意

コードは自由に使用可。データは各出典の規約に従います(UCI 公開、Lending Club アーカイブ公開、Home Credit は Kaggle 規約上 再配布禁止 — 直接ダウンロード)。本文の結論は公開データに対するものであり、実務データの符号は上記2点(パススルー・貸し倒れ定義)で直接検証してください。

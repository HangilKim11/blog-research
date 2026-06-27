# SSL 信用リスク — 公開データで検証

> SSL 単体では GBM に届きません。しかし GBM に統合すると性能が上がります。
>
> Self-supervised pretraining alone underperforms a tuned GBM on tabular credit risk. Merged into the GBM as auxiliary features, it improves it.

信用リスクの実務は事実上 GBM のモノカルチャーです。NLP やビジョンで自己教師あり学習(SSL)が成功したのを受けて、同じアプローチが信用データにも通用するのかを公開データで検証しました。結論は2行に要約できます。トランザクション系列を SSL で事前学習したエンコーダは、単体の分類器としてはよくチューニングされた GBM に届きません。しかしその埋め込みを GBM の追加特徴量として統合すると、GBM 単体より安定して性能が上がります。

解説記事: **[han-co.com/ja/blog/ssl-credit-risk](https://han-co.com/ja/blog/ssl-credit-risk)**

## 主な結果

| アプローチ | test AMEX | ベースライン比 |
|---|---:|---:|
| **GBM ベースライン** (hand 特徴量 1,291個) | **0.79558** | 0 |
| SSL 単体ベスト (Hybrid full fine-tune) | 0.79267 | -0.00291(届かない) |
| **GBM + SSL 統合** (シード6個平均) | **0.79675** | **+0.00117** |
| GBM + SSL 統合、ベストシード | 0.79807 | +0.00249 |

統合の結果はシード6個の平均で test AMEX 0.79675、ベースライン比 +0.00117、標準偏差 0.00098 です。t 値は約 2.9(自由度 5)で、シード6個すべてがベースラインを上回りました。改善幅は小さいですが統計的に意味があります。

当初はシード3個で +0.00142、t=4.1 とより楽観的な値でした。シードを6個に増やすと +0.00117、t≈2.9 に下がり、こちらの方が正直な値です。

## 手法の要約

エンコーダは 4-layer のトランザクション トランスフォーマー(d=128、約87万パラメータ)です。SSL 目的関数4種(マスキング復元、次ステップ予測、コントラスティブ、ハイブリッド)でトランザクション系列を事前学習します。評価は3つのプロトコルで行いました(linear probe、full fine-tune、GBM 統合)。データは顧客単位の 80/10/10 split で分割しています。

2つの分解結果が、統合が何を加えるのかを説明します。

- **Ablation**: 上位100個の hand 特徴量を除去すると test AMEX が 0.00592 低下します。そこに SSL 埋め込みを加えると、そのうち 0.00324 を回復します。回復率は約 55% です。SSL が専門家の特徴量エンジニアリングを raw トランザクションから半分ほど教師なしで再発見している、ということです。
- **Segment**: 平均の改善幅は均一ではありません。GBM が「安全」と判断した予測下位 0〜3 分位(silent default の領域)に、ゲインが +0.02〜+0.03 で集中します。プライム顧客の false negative は実務で最もコストの高い失敗モードですが、SSL がまさにその領域を捉えてくれます。

## 再現方法

コードは [github.com/HangilKim11/blog-research/tree/main/ssl-credit-risk](https://github.com/HangilKim11/blog-research/tree/main/ssl-credit-risk) にあります。

```bash
uv sync

# 1. データのダウンロード(Kaggle 認証情報が必要)
uv run python -m amex.data.kaggle_download --mode full

# 2. 系列ビルド → split → 特徴量エンジニアリング
uv run python -m amex.data.sequence_builder
uv run python -m amex.data.splits
uv run python -m amex.data.feature_engineering

# 3. GBM ベースライン
uv run python -m amex.baseline.lgbm

# 4. SSL 事前学習(4 objective)→ フルファインチューニング
bash scripts/run_phase2_all.sh
bash scripts/run_phase3_finetune.sh

# 5. 埋め込み抽出 → GBM 統合
#    (scripts/make_augmented_features.py → amex.baseline.lgbm)
```

Kaggle の認証情報(`~/.kaggle/kaggle.json`)が必要で、AMEX コンペの規約を先に承認する必要があります。W&B ロギングは任意です(各スクリプトで `--no-wandb` で無効化できます)。

## 再現時間 / compute

単一の RTX 4070 Laptop(8GB)+ Ryzen 9 7940HS で約 20〜22 GPU+CPU 時間です。ディスクは約 70GB 必要で、クラウド費用は0円です(すべてローカルで動きます)。

## データ出典

AMEX Default Prediction(Kaggle、2022)です。顧客45万9千名 × 最大13か月の匿名化された月次プロファイルです。

| データ | 規模 | 出典 |
|---|---|---|
| AMEX Default Prediction | 45万9千 顧客 × 最大13か月 | [kaggle.com/competitions/amex-default-prediction](https://www.kaggle.com/competitions/amex-default-prediction) |

Kaggle の規約上、再配布はできません。直接ダウンロードする必要があります(上記の `kaggle_download` ステップが取得します)。

## 論文

- `reports/paper/paper.pdf` — 英語
- `reports/paper/paper_ko.pdf` — 韓国語

## ライセンス・注意

コードは自由に使用可。データは Kaggle/AMEX の規約に従います(再配布禁止、直接ダウンロード)。本文の数値は AMEX 2022 公開データ1種に対するものです。+0.00117 という改善幅が他の信用データにそのまま一般化される保証はありません。

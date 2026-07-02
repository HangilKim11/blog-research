# TabFMは信用貸し倒れでGBMに勝てるか — 公開データ検証

> GoogleのTabFM（ゼロショットのテーブル基盤モデル）は、信用の貸し倒れ予測でよくチューニングしたGBMに勝てるか。公開データ（UCI 台湾クレジットカード貸し倒れ）で競わせた再現コードです。
>
> Does Google's zero-shot TabFM beat a well-tuned GBM at credit-default prediction? Reproduction code, tested on public data (UCI Taiwan credit-card default).

記事全文: **[han-co.com/ja/blog/tabfm-credit](https://han-co.com/ja/blog/tabfm-credit)**

## 主な結果（5-fold、class_weight なしの自然な比率）

| Arm | ROC-AUC | PR-AUC | KS | ECE ↓ |
|---|---:|---:|---:|---:|
| GBM チューニング (LightGBM) | 0.789 | 0.566 | 0.443 | 0.010 |
| **TabFM ゼロショット** | **0.785** | **0.558** | **0.441** | **0.022** |
| GBM 素 | 0.779 | 0.554 | 0.429 | 0.013 |

- よくチューニングしたGBMがTabFMゼロショットをわずかに上回る（差0.4%p、foldノイズ内）。
- 労力なし同士なら TabFM が素のGBMより上。キャリブレーションは互角。
- このデータの天井が ~0.79 で、どちらもその上には行けない。
- 結論: TabFMは「勝つモデル」ではなく「労力なしで肉薄する速いベースライン」。

## 実行

```bash
pip install -r requirements.txt
# TabFM（GPU + CUDA 必要）
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git"
jupyter notebook tabfm_credit.ja.ipynb
```

- **データはノートブックが自動でダウンロードします**（`ucimlrepo`, UCI id=350）。
- **TabFMモデルは含みません。** 実行時にHugging Face(`google/tabfm-1.0.0-pytorch`)から取得します（VRAM ~6.5GB）。
- TabFMのarmはCUDA GPUが必要です。GBMだけ見るならノートブックの `ARMS` から `tabfm_zeroshot` を外してください。
- 8GB GPU 向け設定（コンテキスト1,000行）。16GB以上なら `CONTEXT_MAX` を増やしアンサンブルのプリセットを使ってください。

## データ出典

| データ | 規模 | 出典 |
|---|---|---|
| UCI Default of Credit Card Clients（台湾, 2005） | 3万件 / 翌月の貸し倒れ | [UCI 350](https://archive.ics.uci.edu/dataset/350) |

## ノートブック構成

素のGBM・チューニングGBM(Optuna)・TabFMゼロショットの3armを層化5-foldで比較します。判別（ROC-AUC・PR-AUC・KS）+ キャリブレーション（Brier・LogLoss・ECE）。CatBoost・XGBoost・TabFMアンサンブルまで含む全6armは元の実験コード（configベース）で実行します。

## ライセンス / 注意

コードは自由使用。データはUCI公開。結論は単一の公開データに対するもので、他の貸し倒れデータや時系列（out-of-time）検証では異なる場合があります。TabFMの重み・構造はGoogleが保有します（元リポジトリ参照）。

# LLMs4OL 2026 Task A CPU baseline

実装・固定validationでの公式exact評価・成功/失敗各5例まで完了。
数値と事例は [REPORT.md](REPORT.md)、全データは `results/v2/` を参照してください。
OntoLearner本体・既存評価アダプター・分割ID一覧は変更していません。
push・PR作成・mergeは行っていません。

## 基盤と依存

- 基盤: `c5626262/OntoLearner` の `research`、コミット
  `eead547edbe2684540e864ce22eb702b63f50474`。
- 保存済み分割: `research/llms4ol2026/splits/seed42/`。
  train 3,442文書、validation 861文書。共有正規化トリプル0件のholdout。
- 公式公開train: `sciknoworg/LLMs4OL-Challenge` の
  `315a9a5d883eada26e00fef1356a05802936c584`、
  `2026/TaskA-Flagship/train_task_a.json`。
- 入力・分割IDのSHA-256と公式trainのGit blobを毎回検証。
- 予測: Python標準ライブラリのみ。評価: 既存環境のNetworkXと公式アダプター。
- 検証環境: Python 3.12.14 / NetworkX 3.2.1 / pytest 9.0.3 / Ruff 0.15.12。
- GPU・外部推論API・有料モデル・モデル重みの取得は不要。
  初期準備で公開コードと公開trainだけを取得し、その後はオフライン実行。

## 再現コマンド

リポジトリルートから実行します。既存の生成物を保護するため、新しい出力先を指定します。
公式公開trainは `/workspace/llms4ol2026-data/train_task_a.json` に取得済みです。
以下の再実行ではtrainの再取得は不要です。

```bash
# 1. 保存済み分割から学習データと、正解を含まない推論入力を作る。
.venv/bin/python -m baselines.task_a_cpu.experiment prepare \
  --source /workspace/llms4ol2026-data/train_task_a.json \
  --output /tmp/task-a-inputs-new

# 2. 別プロセスで予測。validation正解ファイルを受け取る引数はない。
.venv/bin/python baselines/task_a_cpu/predict.py \
  --train /tmp/task-a-inputs-new/train.json \
  --input /tmp/task-a-inputs-new/validation-contexts.json \
  --output /tmp/task-a-run-new/predictions.json

# 3. 保存済み予測を既存公式アダプターで評価。
PYTHONHASHSEED=0 .venv/bin/python -m baselines.task_a_cpu.experiment evaluate \
  --source /workspace/llms4ol2026-data/train_task_a.json \
  --predictions /tmp/task-a-run-new/predictions.json \
  --output /tmp/task-a-run-new/scored
```

公式評価器は集合の走査順序により浮動小数点の最下位桁が変わり得るため、
評価時は `PYTHONHASHSEED=0` を指定します。予測はseedに依存しません。
予測マニフェストに入力・train・出力・実装のSHA-256と設定を記録します。
preparation.jsonには固定分割manifestのハッシュも保存します。
別プロセスによる再実行で予測・マニフェスト・評価JSONのbyte一致を検証しています。

## 入出力と漏洩防止

`predict.py` はJSON配列とJSONLを受け取ります。
trainは `id/context/triples`、推論入力は `id/context` です。
`experiment prepare` が公式の `primitive-ontology-triples` をtrainのみに変換します。
validationは既存 `TrainingSample.inference_input()` からid/contextだけを保存します。

```json
[{"id":"document-id","triples":[
  {"subject":"child","predicate":"is-a","object":"parent"}
]}]
```

出力は文書ごとのレコードを持つJSON配列。空予測でもレコードを保存し、IDと文書順を維持します。
`score_saved` は既存アダプター用の `id/primitive-ontology-triples` 形式も保存します。
この形式は既存評価アダプターの内部契約であり、正式な大会提出仕様の確認は別途必要です。
重複ID、train/input ID重複、評価時のID過不足を拒否します。
予測関数はcontextしか受け取らず、入力の追加フィールドやIDをモデル特徴量に使いません。

分割準備は分割の検証・射影のために公開train全体を読みます。
実際の予測プロセスが読むのは実験trainとvalidation-contextsだけです。
評価時に初めて保存済み予測とvalidation正解を照合します。

## 固定アルゴリズム

1. 実験trainのcontextだけにUnicode単語TF-IDFをfit。
   raw count TF、`log((1+n)/(1+df))+1` IDF、L2正規化。
2. cosine上位3文書（正の類似度かつ0.1以上）を取得。
   転置インデックスで同じ内積を計算し、同点はtrain順。
3. 候補trainトリプルの両端点の単語列がcontextに出る場合に転用。
   subject→objectの方向とpredicateを保持。
4. trainの端点ラベルから単語trieを作り、文中の用語間の短い表現をルールで判定。
   `is a`、`is a type/subclass/kind of`、`is classified as` 等からis-a、
   `instance of`からinstance-of、`part of`からpart_of、`located in`からlocated in。
   `such as`/`including`は例→カテゴリの向きにする。
   汎用表現では、trainでinstance-ofのsubjectに出現し、is-aの端点または
   instance-ofのobjectに出現しなかったラベルを個体候補として区別。
5. 辞書にない用語にも、短い全文 `X is a type of Y` 等の明示的ルールを適用。
6. 同一トリプルを重複排除し、辞書順に保存。

`--top-k`、`--min-similarity`、`--no-rules` を指定可能。
今回、閾値探索はしていません。初期転用版の集計結果がTP=0だったため、
固定分割の性質とtrain本文を確認し、辞書ルールを追加しました。
validationの個別正解や個別事例によるルール作成はしていませんが、
集計値を一度見た開発用validationであることは性能解釈に含めてください。

共起だけでは関係は保証されず、一般語・部分用語による誤検出もあります。
辞書外の用語、複雑な列挙、コード付きラベル、言い換えに弱い初期ベースラインです。

## 評価

`experiment evaluate` は既存 `research.llms4ol2026.evaluator.score_triples`
をexactモードで呼びます。公式ソースのblobを照合し、モデル不要の関数を
ASTから読み込む既存の仕組みをそのまま利用します。semanticモードは実行しません。
Graph Similarityは公式の3成分を用い、文書ごとの値をローカルで算術平均します。

関係別micro F1は公式の小文字化・空白正規化後の有向トリプル集合で算出します。
`is-a`と`instance-of`は別、残りはother。0分母は0、TP/FP/FNも保存。
単独 `evaluate.py` は正規化前の診断専用です。公式結果には必ずexperimentを使ってください。

成功例: TPが1以上の文書をF1降順で5件（部分成功）。
失敗例: FPまたはFNがある文書をF1昇順で5件。同点はID順。
全文・正解一致・誤検出・見逃しは評価JSONに保存します。

## 検査

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check . --exclude __init__.py
.venv/bin/python -m compileall -q baselines/task_a_cpu tests/test_task_a_cpu.py
```

全91テスト成功（既存72＋追加19）。実際の検証ではsocket.connectも禁止しました。
Ruffはリポジトリ全体で成功。

設定済みpre-commitは全ファイルを検査しました。自動修正フックを含むため、
同内容の一時Gitスナップショットで実行しています。
既存 `metadata/ontolearner-metadata.rdf`、`vendor/2026_README.md`、
`vendor/graph_similarity.py` の末尾改行・末尾空白のみ不合格です。
公式原文のハッシュを変えないため修正していません。ほかのフックは成功しています。
今回追加ファイルだけのpre-commit結果と全体の結果を `checks/` に保存します。

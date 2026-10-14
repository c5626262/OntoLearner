# LLMs4OL 2026 Task A — Flagship準備

2026年のEnd-to-End Ontology Learning専用の読込・統計・評価補助コードです。
既存のOntoLearner機能には変更を加えていません。モデル推論、学習、外部API呼出しは行いません。

## 対象と出典

- 公式仕様: https://sites.google.com/view/llms4ol2026/flagship-task
- 方針: https://sites.google.com/view/llms4ol2026/challenge-overview
- 固定コミット: `315a9a5d883eada26e00fef1356a05802936c584`
- 学習データ: `2026/TaskA-Flagship/train_task_a.json`、4,303件、15,594,307 bytes。
- 評価実装: `2026/metrics/graph_similarity.py`。
- データおよび評価器のGit blob SHA・SHA-256は `source_manifest.json` に記録。

`TaskA-Flagship` には取得時点で専用READMEやベースライン実装がなく、
親ディレクトリの2026 READMEを `vendor/2026_README.md` に保存しています。
公式ページの `2026/graph_similarity_metric.py` リンクは404で、現行実装は上記パスです。
評価器の原文とMITライセンスを `vendor` に保持しています。

今回読んだ実データは公開trainのみです。testファイルの内容は取得・分析していません。
公式禁止一覧の元オントロジーを取得したり、そこから追加データを再生成したりしていません。
前回調査のAgrO/OM等を使う実行例は、本Task Aの学習には採用しません。

## 入出力

観測したtrainはJSON配列で、各要素が厳密に以下の3フィールドを持ちます。

```json
{"id":"sample-id","context":"raw domain text","primitive-ontology-triples":[["child","is-a","parent"]]}
```

トリプル順はsubject, predicate, objectです。`is-a` はchild→parent、
`instance-of` はterm→typeです。読み込み時には原表記を保持し、暗黙の正規化・統合はしません。
`TrainingSample.inference_input()` はid/contextだけを抽出し、正解トリプルを推論入力から分離します。

予測検証用の内部形式はidとprimitive-ontology-triplesの配列です。
これは公式ページの記述から設けた暫定形式で、正式な提出ファイル形式・命名・圧縮・アップロード仕様は未確認です。
空の予測トリプル一覧を許容し、IDの不足・余分・重複を検出します。

## 再実行

リポジトリルートから実行してください。データはGit管理対象外の
`/workspace/llms4ol2026-data/train_task_a.json` に保存しています。
CLIは固定版trainのblobを照合し、別のデータを誤って分析することを防ぎます。

```bash
.venv/bin/python -m research.llms4ol2026 analyze \
  --train /workspace/llms4ol2026-data/train_task_a.json \
  --output research/llms4ol2026/reports/train_statistics.json

.venv/bin/python -m research.llms4ol2026 evaluate \
  --train /workspace/llms4ol2026-data/train_task_a.json \
  --predictions /absolute/path/to/actual_predictions.json \
  --mode exact --output /absolute/path/to/evaluation.json

.venv/bin/python -m pytest tests/test_llms4ol2026.py -q -p no:cacheprovider
```

evaluateは利用者が用意した予測のみを読みます。正解から予測を作る機能はありません。
文書別スコアの算術平均はローカル集計方法として明示しています。
公式のデータセット全体の集約方法は未確認です。

## モデルを使わない公式評価器の実行

公式ファイルを普通にimportすると、モジュール先頭で
`nomic-ai/nomic-embed-text-v1.5` をロードします。
`evaluator.py` は改変していない公式ソースのblobを検証し、ASTからモデル不要の
exact/fuzzy関数と定数だけを選択して実行します。
semantic modeと公式evaluate全体は今回実行していません。
これは近似モデルによる代替semantic評価ではありません。

現行コードで確認した仕様:

- Exact: 小文字化、前後・連続空白の正規化後、集合として完全トリプル一致。
- Neighborhood: OUTとINの両方の近傍をノードごとにJaccard評価。
- Taxonomy: is-aの祖先と子孫を和集合にしてJaccard評価。向きの完全反転を見逃す例があります。
- Graph Similarity: 上記3成分の算術平均。
- is-aがない完全一致例のGraph Similarityは2/3、双方空の例は1/3。
- サイクル・非連結性を直接検証するペナルティはありません。
- Fuzzy: Hungarian matchingと0.90閾値で予測を参照トリプルへ評価時に対応付けます。

公式ページはOUT近傍と祖先集合を説明しているため、実装との相違は主催者への確認事項です。
方向だけを測る `directed_taxonomy_diagnostic` は独自指標と明記し、公式スコアに混ぜません。
`ontology_graph` はMultiDiGraphで複数述語と方向を保持します。

## 集計の定義と注意

- 文書長は文字数と空白区切り語数です。LLMトークン数ではありません。
- is-a、instance-of、残りの述語を別々に集計します。`type` 等を意味推定で付け替えません。
- 双方向頻度は文書内の重複を除き、自己ループを除外し、無向のノード対を1回だけ数えます。
- 表記揺れ候補はNFKC・casefold・空白・ハイフン/アンダースコアの比較による候補群です。
- equivalent class/same as等の注釈も、人手確認なしに概念同一性・同義語とは断定しません。
- trainのサイクルや非連結性を検出しても、参照データは修正しません。

## 今後の接続設計

新しいFlagshipLearnerを追加し、InferenceDocument→根拠付きトリプル→
MultiDiGraph→提出形式、の順で処理します。候補型や関係一覧をtestの正解から作りません。
既存のText2Ontoのプロンプト・解析・Retrieverの部品を再利用し、
既存LLM非階層抽出のis-aゲートと順序依存のペア生成は新しい経路へ引き継ぎません。
Raw prediction、正規化後、クラスタ統合後を別々に記録・評価します。

クラスタリングの比較対象は、train文書の検索群、概念表記の統合、
関係ラベルの統合、グラフ整合性の後処理です。
is-a/instance-ofを別カテゴリに保ち、part_of/has partの逆関係を同義語として統合しません。
閾値や対応表はtrain内validationで決め、公式testで学習・調整しません。

## ローカル調査の生成物

以下の `reports/` はGit管理対象から除外します。学習データ、キャッシュ、モデル重みも
コミットには含めません。評価器の小規模な合成例は `tests/test_llms4ol2026.py` に収録しています。

- `reports/train_statistics.json`: 全学習データの統計と候補例
- `reports/relation_distribution.csv`: 全28ラベルの分布
- `reports/evaluator_smoke.json`: 合成例だけの公式評価器の動作
- `reports/tests.log`, `reports/tests.xml`: 既存21＋追加33＝54テストの検証結果

合成評価例の完全一致は評価器の正常系テストであり、モデル性能として報告しません。

## 再現可能なtrain/validation分割

`split` コマンドを追加しています。固定版の公開trainから、共有トリプルと類似文書を
同じ側に置く分割を作ります。保存済みID一覧・条件は `splits/seed42/`、
方式・漏洩検査・希少関係の注意点は [SPLITTING.md](SPLITTING.md) を参照してください。

```bash
.venv/bin/python -m research.llms4ol2026 split \
  --train /workspace/llms4ol2026-data/train_task_a.json \
  --seed 42 --validation-fraction 0.2 --candidates 64 --near-threshold 0.8 \
  --output research/llms4ol2026/splits/seed42
```

推論用の文書は `splitting.inference_documents()` でid/contextだけを取り出します。
正解トリプルは分割条件・関係分布・評価のためだけに参照し、推論入力へ渡しません。

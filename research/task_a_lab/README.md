# Task A 自律研究開発 — CPU実験とLLM/後処理の比較基盤

結果は [REPORT.md](REPORT.md)、測定条件・ハッシュは
[reports/provenance.json](reports/provenance.json) に保存します。
既存ベースラインは `baselines/task_a_cpu/` に保全し、変更していません。
固定分割・公式評価器・OntoLearner本体も変更していません。

## 状態を区別する

| 項目 | 状態 |
|---|---|
| 元CPUベースライン再現・詳細分析 | 固定validation 861文書で実測 |
| rules_only / conservative / surface | CPUで実測、全設定を報告 |
| A/B/C後処理 | 実装・合成テスト・surface予測上で実測 |
| Zero/few-shot LLM抽出 | 実装済み、モックのみで検証 |
| ローカルTransformers接続 | 実装済み、モデル/トークナイザーのモックのみで検証 |
| 実LLMの精度・速度・メモリ | 未測定 |
| 埋め込みや実LLMによる概念クラスタリング | 未実装・未検証 |
| 公式semantic評価 | 未実施 |
| 公式exact Graph Similarity | 既存公式アダプターを使用。文書平均はローカル集計 |

依存は既存の `.venv` を使用しました。今回新たなモデルや依存パッケージを
ダウンロードしていません。実行中のSSH接続・GPU利用・外部推論API呼出しはありません。

## ファイルと保全

- `analysis.py`: 正解を扱う**評価専用**の集計。区分別F1、長さ別、誤り区分、各10例。
- `diagnostic_cases.py`: 方向/ラベル/表記候補の対照例。推論からは呼ばない。
- `compare.py`: 文書単位の改善/悪化、正解増減を比較。推論からは呼ばない。
- `cpu.py`: 元の実装を継承した独立した3つの改善設定。fitは実験trainだけ。
- `contracts.py`, `llm.py`, `local_backend.py`, `run_llm.py`: LLM抽出契約・安全な解析・ローカル接続・CLI。
- `postprocess.py`, `postprocess_run.py`: 生予測を変更せずA/B/Cを差し替える。
- `fixtures/`: 明示的な合成入力とモック応答。実モデルの性能測定には使わない。
- `reports/`, `checks/`: 小規模な測定結果・再現条件・検査ログをGit管理。
- `runs/`: 再生成可能な生応答、予測、系譜ログをローカル保持。Git管理対象外。

公式trainや準備済みtrain/contextは `/workspace` にそのまま保持し、Gitに入れません。
既存の全文正解を含む2.2MB評価JSONも削除せずローカルに保持しています。
基盤コミットは `eead547edbe2684540e864ce22eb702b63f50474`、
既存CPU成果の保全コミットは `4791725`、新実装のコミットは `66643c7`。
作業ブランチは `research/task-a-autonomous-cpu`。この作業のコミットは未pushです。

## CPU実験の再現

以下はリポジトリルートから実行します。出力先は未使用のディレクトリにしてください。
CLIは既存成果の上書きを拒否します。固定分割seedは42、CPU抽出には乱数を使いません。

```bash
# データ準備: 元の公式trainをハッシュ検証し、推論用はid/contextのみを保存。
.venv/bin/python -m baselines.task_a_cpu.experiment prepare \
  --source /workspace/llms4ol2026-data/train_task_a.json \
  --output /tmp/task-a-inputs-new

# 元のベースラインを再現する場合
.venv/bin/python baselines/task_a_cpu/predict.py \
  --train /tmp/task-a-inputs-new/train.json \
  --input /tmp/task-a-inputs-new/validation-contexts.json \
  --output /tmp/task-a-original-new/predictions.json

# 改善版。--variantをrules_only/conservative/surfaceに切り替える。
.venv/bin/python -m research.task_a_lab.cpu \
  --train /tmp/task-a-inputs-new/train.json \
  --input /tmp/task-a-inputs-new/validation-contexts.json \
  --variant surface --output /tmp/task-a-surface-new

# 評価プロセスだけが正解を参照する。
PYTHONHASHSEED=0 .venv/bin/python -m research.task_a_lab.analysis \
  --source /workspace/llms4ol2026-data/train_task_a.json \
  --train /tmp/task-a-inputs-new/train.json \
  --predictions /tmp/task-a-surface-new/predictions.json \
  --output /tmp/task-a-surface-analysis-new.json
```

`PYTHONHASHSEED=0` は公式評価器の集合走査による浮動小数点の最下位桁を安定させます。
生トリプル数と公式正規化後の集合件数を区別して記録します。
同一validationを開発・比較に利用した探索結果であり、未観測testへの汎化性能ではありません。
今回の分析後、3設定を固定して一度比較しました。表記候補は正解を用いた**診断**であって、
それを推論で置換する辞書や後処理にはしていません。

## LLMモック検証

```bash
# Zero-shot、C後処理を通す合成スモークテスト。
.venv/bin/python -m research.task_a_lab.run_llm \
  --input research/task_a_lab/fixtures/mock-input.json \
  --mock-responses research/task_a_lab/fixtures/mock-responses.json \
  --postprocess C --output /tmp/task-a-mock-zero-new

# Few-shotは実験train由来の例だけを渡す。下記は合成データのテスト。
.venv/bin/python -m research.task_a_lab.run_llm \
  --input research/task_a_lab/fixtures/mock-input.json \
  --mock-responses research/task_a_lab/fixtures/mock-responses.json \
  --train research/task_a_lab/fixtures/mock-train.json --examples 1 \
  --postprocess C --output /tmp/task-a-mock-few-new
```

モックは入力順に応答を返す検証器で、理解・抽出能力を持ちません。
成功2件・不正JSON1件という結果はエラー処理の試験であり、LLM精度ではありません。

入力は `id/context` だけに限定し、余分なtriplesフィールドを拒否します。
`InferenceDocument` に正解フィールドはありません。Few-shotはtrain-only poolから
単語集合Jaccardで選び、同点はID順。trainと推論IDの重複を拒否します。
API利用者は、poolへ本当にtrainだけを渡す責任があります。CLIには公式train全体を
渡さず、上記prepareで作ったtrain.jsonを使ってください。

応答は厳密に `{"triples":[{"subject":...,"predicate":...,"object":...}]}`。
JSON全体または全体を囲むjsonコードフェンスのみを受け付けます。
重複キー・余分なフィールド・欠損・空文字・非文字列・過大応答を拒否します。
一部だけの救済、eval、推測によるJSON補修、自動再試行はしません。

`records.jsonl` は文書ごとにflushし、生応答、元トリプル、正規化結果、状態、
few-shot例ID、プロンプトハッシュ、後処理系譜を保存します。
失敗時も文書IDを残し、予測は空、状態は `backend_error` / `invalid_response`。
有効な空予測とは区別します。モデル例外の本文は秘密情報漏洩防止のため保存せず、型だけを残します。
失敗文書を評価対象から除外してスコアを良くする処理はありません。

## 研究室Linuxサーバでの実モデル実験（今回未実行）

この節はユーザーがサーバを操作できるようになってから実行する手順です。
この作業ではSSH接続もGPU利用もしていません。現在のブランチは未pushなので、
先にコードを手動共有してください。最終成果のGit bundleを取り込む方法も使えます。
基盤researchを持つcloneでbundleを検証し、作業ブランチをfetchしてからcheckoutします。

```bash
git bundle verify /path/to/task-a-research.bundle
git fetch /path/to/task-a-research.bundle \
  research/task-a-autonomous-cpu:research/task-a-autonomous-cpu
git switch research/task-a-autonomous-cpu
```

1. サーバ側の既存OntoLearner環境と、利用許諾を確認済みの**ローカル**instructモデルを用意。
   本実装は既存ディレクトリしか受け付けず、Hub IDによる取得はしません。
2. 固定版公開trainと保存済みsplitを配置し、prepareでtrainとvalidation-contextsを生成。
3. まず2〜5文書でモデルのchat template、出力形式、メモリ、最大コンテキストを確認。
4. その後、全861文書のzero-shotを保存し、同条件でfew-shot（例2件）を別runに保存。

```bash
# サーバに既に配置済みのモデル。具体的なモデル名・revisionは実験前に記録する。
TASK_A_MODEL_DIR=/models/your-local-instruct-model

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m research.task_a_lab.run_llm \
  --backend local --model-path "$TASK_A_MODEL_DIR" --device cuda \
  --input /path/to/prepared/validation-contexts.json \
  --max-input-tokens 8192 --max-new-tokens 2048 \
  --postprocess A --output /path/to/new-run/zero

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m research.task_a_lab.run_llm \
  --backend local --model-path "$TASK_A_MODEL_DIR" --device cuda \
  --input /path/to/prepared/validation-contexts.json \
  --train /path/to/prepared/train.json --examples 2 \
  --max-input-tokens 8192 --max-new-tokens 2048 \
  --postprocess A --output /path/to/new-run/few
```

`--device cpu`でも指定可能ですが、実モデルのCPU推論時間は未測定です。
`local_files_only=True`、`trust_remote_code=False`。サンプリングなしのgreedy生成です。
実装は重みを通常精度でロードするため、大きなモデルではメモリ不足になり得ます。
量子化、複数GPU、バッチ推論、再開/自動リトライは未実装です。
モデルコンテキスト上限と入力token予算を超えた場合は、文書を黙って切らずエラーにします。
必要ならtrain内の開発用データで予算・few-shot件数を決めてからvalidationを実行してください。

接続コードのモック検証環境はTorch 2.11.0 / Transformers 5.7.0。
実機ではCUDAドライバ、PyTorch、モデルchat templateの互換性を別途検証してください。
モデルディレクトリの重みrevision/checksum、GPU型、RAM/VRAM、依存版、token予算、
失敗率、総時間を記録してください。現状manifestはモデルパスを記録しますが、
モデル重みの全ファイルハッシュまでは計算しません。

保存後、`analysis` CLIへ `raw-predictions.json` または `predictions.json` を渡して公式exact評価。
失敗率はmanifest/recordsと合わせて報告します。実モデル間比較は同一分割・同一予算で行い、
必要なら別の未使用holdoutを確保してください。semantic評価は今回のexact値と混ぜません。

## A/B/C後処理比較

```bash
.venv/bin/python -m research.task_a_lab.postprocess_run \
  --input /path/to/prepared/validation-contexts.json \
  --predictions /path/to/new-run/zero/raw-predictions.json \
  --mode C --output /path/to/new-postprocess-C
```

A: 原表記、方向、複数述語、重複をそのまま保持。
B: 小文字化と空白正規化だけ。完全一致重複を統合し、全元インデックスを保持。
C: Bに加え、同一文書内のハイフン/アンダースコア/空白違いの候補を保守的に統合。

Cの条件は、予測グラフから得たclass/individual区分の一致、IN/OUT付き関係集合の一致、
該当文の単語集合Jaccard>=0.6、統合により既存辺が自己ループにならないことです。
未知の型やclass/individual混在は統合しません。complete-linkにより連鎖的な誤統合を抑えます。
閾値0.6は固定の試作値で、validation最適化していません。

これは文脈特徴・グラフ上の型・方向を使う**語彙的CPU試作**です。
文脈埋め込み、実LLMによる型判定、意味的同義語クラスタリングはありません。
型は予測から推定するので誤りを含み得ます。
`same as`や`equivalent class`を無条件統合せず、`part_of`/`has part`を同義語扱いしません。
述語自体の意味クラスタリングも未実装で、B同様の表記正規化だけです。

`lineage.jsonl` はraw triples、元エッジindex、概念の原表記→正規化→代表名、
述語の原表記→正規化、統合理由を保存します。生予測ファイルは上書きしません。

## 品質検査

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check . --exclude __init__.py
.venv/bin/python -m compileall -q research/task_a_lab tests/test_task_a_lab.py
```

既存91テストと追加44テスト、合計135件が成功。実測時はsocket接続も禁止しました。
追加テストはJSON異常系、方向・複数関係、ラベル混入拒否、few-shot分離、
ローカル取得限定、入力超過、完了部分のみdecode、型/方向/文脈の統合禁止、
系譜保持、完全リンク、CLI上書き拒否などを対象にしています。

既存pre-commitには自動修正フックがあるため、同じファイルの一時Gitスナップショットで
全体検査します。既存3ファイルの空白・末尾改行不合格は維持し、公式原文は修正しません。
今回の追加・変更ファイルは全フック成功を確認し、ログをchecksに保存します。

# LLMs4OL 2026 Task A 自律研究開発レポート

2026-10-15 JST。固定train 3,442文書・validation 861文書で実測しました。元のベースラインを維持し、改善版を別実装・別runで比較しました。以下でスコアは0–1、P/Rの表記だけ百分率です。

## 1. 保全と保存状況

開始時は `baseline/task-a-cpu`、HEADは `eead547e`、CPU成果は未コミットでした。必要なコード・設定・小規模予測・再現記録を保全コミット `4791725` に保存し、`research/task-a-autonomous-cpu` へ分岐しました。新実装は `66643c7` です。
**今回のコミットはGitHub未保存（ローカルのみ）です。push・PR作成・mergeは行っていません。** 元のresearch基盤はGitHub由来です。大容量train・キャッシュ・重みはGitに入れていません。既存の全文正解入り評価JSONや生データは削除せずローカル保持しています。
既存OntoLearner本体・公式評価器・固定分割・元CPU実装の内容は保全時から変更していません。元予測は独立再実行でbyte一致し、Graph Similarity 0.005497502704513878を再現しました。

## 2. 元CPUベースラインのエラー分析

生予測は1,274トリプル、正規化後も1,274件。参照11,135件は公式正規化・文書内重複除去後11,132件です。
| 関係 | 予測数 | 参照数（集合） | 一致数 | Precision | Recall | micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| is-a | 1174 | 9882 | 47 | 4.003% | 0.476% | 0.008502171 |
| instance-of | 87 | 1060 | 0 | 0.000% | 0.000% | 0.000000000 |
| other | 13 | 190 | 0 | 0.000% | 0.000% | 0.000000000 |

予測0件は **399/861文書（46.34%）**。正しいトリプルを1件以上含む文書は36件、文書全体の完全一致は0件です。
全体Precisionは3.689%、Recallは0.422%。両方低いものの、Recallが約9分の1で取りこぼしが特に大きい状態です。正解トリプルの9,479件（85.15%）は少なくとも片方の端点表記がtrain辞書にありません。**train辞書だけを使う抽出のカバレッジ不足**が示唆されます。これは表記上の未出現であり、未知概念と同一ではありません。
参照トリプルの両端点がcontextに単語列として現れるものは8,040件、片方だけは1,974件、両方とも現れないものは1,118件です。後二者には言い換えや注釈コードの復元が必要なケースが含まれ得ます。これは語彙一致の診断で、意味的な抽出上限ではありません。

### 方向・関係ラベル・表記

誤検出1,227件を優先順に排他的に分類しました。同一参照への複数対応を許す診断であり、補正後F1ではありません。

| 誤検出の区分 | 件数 |
|---|---:|
| 端点逆転だけで参照に一致 | 2 |
| 同じ有向端点でpredicateだけ異なる | 7 |
| NFKC・区切り表記の違いだけで一致する候補 | 0 |
| 両端点SequenceMatcher>=0.8の候補 | 60 |
| その他の端点違い・参照にない関係 | 1,158 |

具体例: `hyphal tip → is-a → hyphal tip, septate hypha` は参照と逆向きです。`chloroform → instance-of → one-carbon compound` は参照ではis-aです。`mass percentage → instance-of → concentration unit` は参照ではtypeであり、typeをinstance-ofへ勝手に付け替えるべきではありません。
`fast food → is-a → convenience` は参照のconvenience foodを短く取り過ぎています。`a solid substance` の冠詞や、肉製品の `(raw)` の欠落も見られます。「近い文字列」は同義語の証明ではなく、単純統合による誤りもあり得ます。参照にないトリプルが現実世界でも偽とは限りません。
詳細は [diagnostic_examples.json](reports/diagnostic_examples.json)。これらの正解対照を推論時の辞書にはしていません。

### 公式Graph Similarityの成分

| 成分 | 文書平均 |
|---|---:|
| edge_f1 | 0.007914556 |
| neighborhood_similarity | 0.004046621 |
| taxonomy_similarity | 0.004531331 |
| graph_similarity | 0.005497503 |

既存の公式exact関数を使用。Graph SimilarityはEdge F1・Neighborhood・Taxonomyの平均です。上表は文書ごとの算術平均で、公式大会のデータセット全体集約方法は未確認です。独自のmicro F1とは集計単位が違います。公式Taxonomyは祖先/子孫の和集合なので、方向誤りだけを見逃す場合があります。

### 文書長・正解数別

| 文書長（文字） | 文書数 | Graph Similarity | micro F1 |
|---|---:|---:|---:|
| 0-999 | 63 | 0.001881 | 0.003610 |
| 1000-1999 | 552 | 0.007239 | 0.011140 |
| 2000-3999 | 243 | 0.002546 | 0.004247 |
| 4000以上 | 3 | 0.000000 | 0.000000 |

| 参照トリプル数（集合） | 文書数 | Graph Similarity | micro F1 |
|---|---:|---:|---:|---:|
| 0-5 | 204 | 0.009357 | 0.016845 |
| 6-10 | 322 | 0.005948 | 0.012065 |
| 11-20 | 142 | 0.005025 | 0.011217 |
| 21以上 | 193 | 0.001014 | 0.002012 |

1,000–1,999文字の群が比較的良く、参照21件以上では低下しました。長い列挙・多い関係を少数ルールで拾い切れないことが示唆されます。ただし分野や語彙の交絡を除いておらず、4,000文字以上は3文書のみです。因果や一般的傾向の確定ではありません。

## 3. CPU改善実験

3設定を固定して同じvalidationを評価。元を上書きせず、全設定を報告します。閾値探索やvalidation例からの正解コピーはしていません。
- rules_only: trainトリプル転用と全文フォールバックを除き、既存の文中辞書ルールだけ。
- conservative: rules_onlyに一般語除去と入れ子の部分用語抑制を追加。
- surface: rules_onlyに有限節から未知用語を生成する表層ルールを追加。

| 設定 | Graph Similarity | is-a F1 | instance-of F1 | other F1 | TP | FP | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 元ベースライン | 0.005497503 | 0.008502171 | 0.000000000 | 0.000000000 | 47 | 1227 | 3.689% | 0.422% |
| rules_only | 0.005805111 | 0.008862072 | 0.000000000 | 0.000000000 | 47 | 730 | 6.049% | 0.422% |
| conservative | 0.007227452 | 0.009054227 | 0.000000000 | 0.000000000 | 46 | 257 | 15.182% | 0.413% |
| surface | 0.017306191 | 0.029724374 | 0.001793722 | 0.000000000 | 166 | 1111 | 12.999% | 1.491% |

surfaceは生予測1,280件、正規化後1,277件、一致166件、正解を含む80文書、空予測418文書です。元より空予測は増えましたが、当たった文書での一致数が増えました。is-a 165件、instance-of 1件が一致し、otherは依然0件です。
平均Graph Similarityは約3.15倍ですが、絶対値は0.0173で低性能です。**文書単位では58件改善・24件悪化・779件同値**。元の正解47件はすべて維持し、新たに119件一致しました。追加の誤検出等によりGraph Similarityが下がる文書もあります。
conservativeは32件改善・5件悪化・824件同値。Precisionは上がりましたが、正解1件を失いRecallが0.422%→0.413%に悪化しました。隠さず [paired_conservative.json](reports/paired_conservative.json) に記録しています。
CPUのfit+predict実測時間はrules_only約1.42秒、conservative約1.55秒、surface約1.95秒（この環境・単発測定）。機器間の速度比較やLLM速度を意味しません。surfaceの予測は独立再実行でbyte一致しました。
同じvalidationを分析・方式選択に使った探索結果です。汎化性能や統計的有意差は主張しません。次の調整はtrain内の開発分割で行い、追加の未使用holdoutで確認すべきです。

## 4. 成功10例・失敗10例

元ベースラインを対象に、成功はTP>0をF1降順・ID順で10件、失敗は空予測5件と非空誤り5件を選択。成功は部分成功で、完全正解とは異なります。下記の原因解釈は観測に基づく仮説です。代表1トリプルを示し、全選択データはbaseline_analysis.jsonに保存しています。

### 成功1: `df11445b146c4aaeb5554514c4ede6f0`

一致: `binary operation` → **is-a** → `operation`

TP=5 / FP=2 / FN=3、F1=0.6667。既知の数学用語の階層を5件拾えた一方、anやpointの部分語を誤検出。

### 成功2: `9eb707300b074efc9f94ac693b0e7c3e`

一致: `chenopodium pallidicaule` → **is-a** → `chenopodium`

TP=2 / FP=3 / FN=3、F1=0.4000。種→属は一部一致するが、科まで広げた参照外の階層も出力。

### 成功3: `0d909cb0d9584989a5a54fbce9128386`

一致: `retinol` → **is-a** → `primary allylic alcohol`

TP=1 / FP=0 / FN=4、F1=0.3333。化学用語は一致。肉の部位・raw修飾を含む4件を見逃し。

### 成功4: `a67907b1f9d940479bb3b66d71999c41`

一致: `elephas maximus` → **is-a** → `elephantidae`

TP=1 / FP=0 / FN=4、F1=0.3333。学名→科は一致。obsolete表記や自己ループを含む参照は未回収。

### 成功5: `bbff51cfcc4b4a32b5871bda9b839d2a`

一致: `tropical forest` → **is-a** → `forest`

TP=1 / FP=0 / FN=4、F1=0.3333。tropical forestの階層は一致。taiga等の別名・中間階層を回収できず。

### 成功6: `0b3fae2f42c64fed9751af434f672e0c`

一致: `kathetostoma giganteum` → **is-a** → `stargazer family`

TP=1 / FP=1 / FN=4、F1=0.2857。学名→科は一致するが、親aという一般語の誤検出が残る。

### 成功7: `a45b946a73844610aebbdad2df7227ea`

一致: `acetes japonicus` → **is-a** → `sergestidae`

TP=1 / FP=1 / FN=4、F1=0.2857。エビの学名→科が一致。豆の乾燥・成熟という修飾区別を見逃し。

### 成功8: `2e31f1fa13324890b03248b5cfb2541d`

一致: `global climate` → **is-a** → `climatological average`

TP=2 / FP=5 / FN=6、F1=0.2667。global/regional climateは一致。他のスケールを落とし、equivalent classも過剰出力。

### 成功9: `3f9a7f851efa44b9b84a621d84e78149`

一致: `granule` → **is-a** → `gravel`

TP=2 / FP=4 / FN=7、F1=0.2667。granule/pebbleは一致するが、dataという別文脈の一般語と混線。

### 成功10: `3e0ea8c8b1d54ca4a90f043a8120c4d0`

一致: `urea` → **is-a** → `monocarboxylic acid amide`

TP=1 / FP=2 / FN=4、F1=0.2500。ureaの化学分類は一致。aを親とする誤りと別分類の見逃しが残る。

### 失敗1: `001e034f50924f959c34fc020a76122e`

見逃し（予測空）: `aggregated carbon nanorods` → **is-a** → `carbon allotrope`

TP=0 / FP=0 / FN=5、F1=0.0000。参照5件すべてにtrain辞書未出現の端点があり、語彙依存の限界を示す。

### 失敗2: `005eceb71f8b4377b8aff8436b19556e`

見逃し（予測空）: `beef brisket (raw)` → **is-a** → `beef brisket`

TP=0 / FP=0 / FN=5、F1=0.0000。rawの修飾や自己ループを含む参照を抽出できず。未知端点だけでは説明し切れない。

### 失敗3: `0084c7e04a564d58bb9e51a1ed0ebacb`

見逃し（予測空）: `10000240 - pasta/noodles - ready to eat (perishable) (gs1 gpc)` → **is-a** → `50192900 - pasta/noodles (gs1 gpc)`

TP=0 / FP=0 / FN=7、F1=0.0000。GS1コード付き分類名を復元できず。7件すべてに辞書未出現端点。

### 失敗4: `00857be4a7494939aa6bb960c3d04839`

見逃し（予測空）: `17480 - papayas (efsa foodex2)` → **is-a** → `17470 - papayas and similar- (efsa foodex2)`

TP=0 / FP=0 / FN=6、F1=0.0000。FoodEx2番号付き分類を回収できず。6件すべてに辞書未出現端点。

### 失敗5: `0167d9505664453d9bf7423347141a70`

見逃し（予測空）: `at` → **instance-of** → `at`

TP=0 / FP=0 / FN=6、F1=0.0000。instance-of自己参照やdomain関係などの関係多様性を扱えず。

### 失敗6: `eed909f17e1f4564ac0819de22060fd7`

誤検出: `apple` → **is-a** → `product or service`

TP=0 / FP=28 / FN=33、F1=0.0000。productの代わりにproduct or serviceへ広げ、28件の誤検出。語彙既知でも階層選択が難しい。

### 失敗7: `7758236a9da84f089d741d78ebec0998`

誤検出: `air` → **is-a** → `mixture`

TP=0 / FP=12 / FN=13、F1=0.0000。mixed chemical substanceの代わりにmixtureを親に選択。意味の近さだけではexact一致しない。

### 失敗8: `5841a8c7a5f04941abfcac7e6815247b`

誤検出: `backpack sprayer` → **is-a** → `sprayer`

TP=0 / FP=11 / FN=30、F1=0.0000。農具の親クラスが参照と違い、一般語aも混入。参照30件中29件は未知端点を含む。

### 失敗9: `a64388378d8a4dbd841c20265573ec77`

誤検出: `piece(s) of turkey back` → **is-a** → `piece(s) of turkey meat`

TP=0 / FP=11 / FN=16、F1=0.0000。肉部位の(raw)修飾が脱落し、似たトリプルでも完全一致しない。

### 失敗10: `1e8a8a80ead44e61bf8d0ce5785c6233`

誤検出: `an` → **instance-of** → `duration`

TP=0 / FP=10 / FN=5、F1=0.0000。長い定義的ラベルや単位のinstance-ofを扱えず、anなどを個体と誤認。

## 5. LLM抽出器

実装済み: 交換可能なTextBackend、zero/few-shot、train-only例選択、厳密JSON解析、スキーマ検証、is-a子→親・instance-of用語→型の指示、同一端点の複数述語保持、生応答と元/正規化後予測保存、失敗状態の明示、CLI上書き拒否。推論入力はid/contextのみに限定。
不正JSONを推測で修復せず、部分抽出や自動再試行をしません。backend例外の本文は漏洩防止のため保存せず型だけを記録します。失敗文書も空予測と状態を残し、評価から除外しません。
ローカルTransformers接続も実装しましたが、**実モデルはロードしていません**。local_files_only、trust_remote_code=False、greedy生成、完了部分のみdecode、入力超過時の明示的エラーをモックで検証しました。zero/fewとも合成3文書でok=2/invalid_response=1を確認しただけであり、LLM精度ではありません。

## 6. クラスタリング比較基盤

Aは生表記保持、Bは小文字・空白正規化と重複統合、CはBに文脈単語集合・class/individual区分・IN/OUT関係署名を用いた語彙候補統合を追加。Cはcomplete-linkと自己ループ防止を備えます。元表記・元エッジindex・統合理由を追跡できます。
同義関係を無条件に統合せず、逆関係も統合しません。is-a/instance-ofの区分が違う候補や型が不明な候補は統合しません。
実際のsurface予測でA/B/Cを比較すると、**Cの追加統合は0件**、BとCの予測JSONはbyte一致しました。A/B/Cの公式exact Graph Similarityはすべて0.01730619114756232です。Aは元予測とbyte一致。Bは公式と同じ正規化なのでexact値が変わらないことを確認しました。
合成例ではT-cell/T cellの条件付き統合と型・方向・文脈不一致時の拒否を検証しました。実データでのクラスタリング有効性は示せていません。**実モデルによる文脈埋め込み・型推定・意味的クラスタリングは未実装・未検証**です。

## 7. 検査・ファイル

既存91テストを再確認後、追加44を含む**135テスト成功**。socket接続禁止・CUDA無効で実行。既存SWIGのdeprecation warningが2件あります。Ruff全体検査・compileall成功。
pre-commitは全体をスナップショットで検査し、既存3ファイルの末尾改行/空白のみ不合格。今回の追加・変更ファイルは全フック成功。公式原文を整形してハッシュを壊す修正はしていません。
実装・設定・記録の一覧は [CHANGES.md](CHANGES.md)、実行手順は [README.md](README.md)、検査ログは [checks/](checks/)。

## 8. 未検証事項と次のサーバ実験

- 実LLMのモデル品質・速度・VRAM、実モデルのtokenizer/chat template互換性は未測定。接続コードのモック成功と区別する。
- semantic評価と正式大会の集約/提出仕様は未検証。公式exactと独自診断を混同しない。
- 意味クラスタリング、量子化、バッチ推論、複数GPU、チェックポイント再開は未実装。
- 次はサーバ上の既存ローカルinstructモデルで少数文書スモーク→zero-shot全861件→同モデルfew-shot→同じ生予測でA/B/C比較。手順とコマンドはREADME。今回はSSHもGPUも使っていない。
- モデルrevision、重みchecksum、token予算、ハードウェア、失敗率を記録。改良や閾値はtrain内で調整し、今回見たvalidation以外の未使用holdoutで汎化を確認する。


## スマートフォン向け要約

既存CPUベースラインを保全コミットに保存し、固定validation861文書で元のGraph Similarity 0.005498を再現しました。予測は1,274件、空予測は399文書で、特にRecall不足が課題です。3つのCPU改善版を比較し、表層ルール追加版は0.017306、正解一致166件に改善しました。ただし24文書では悪化しており、汎化性能は未確認です。LLMのzero/few-shot抽出とA/B/C後処理は実装し、モックとCPUで検証しました。実LLM・semantic評価・意味クラスタリングは未検証です。既存91件を含む135テストが成功し、成功・失敗各10例を保存しました。変更はローカルにコミット済みですがGitHubへは未pushです。GPU、外部推論API、SSH、PR、mergeは使っていません。次は研究室サーバのローカルモデルで実抽出を確認できます。

# CPUベースライン評価結果

固定済みseed=42分割（train 3,442文書、validation 861文書）を使用。公式testは未使用。
公式アダプターの **exactモード** を使用し、モデル・GPU・外部推論APIは使用していません。
文書別スコアの算術平均はローカル集計です。公式のデータセット全体の集約方法は未確認です。

| 指標 | 値（0–1） |
|---|---:|
| edge_f1 | 0.007914556 |
| neighborhood_similarity | 0.004046621 |
| taxonomy_similarity | 0.004531331 |
| graph_similarity | 0.005497503 |

関係別スコアは公式と同じ小文字化・空白正規化後の有向トリプルmicro F1です。独自の区分別集計であり、Graph Similarityの成分ではありません。

| 区分 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| is-a | 47 | 1127 | 9835 | 0.040034072 | 0.004756122 | 0.008502171 |
| instance-of | 0 | 87 | 1060 | 0.000000000 | 0.000000000 | 0.000000000 |
| other | 0 | 13 | 190 | 0.000000000 | 0.000000000 | 0.000000000 |
| 全体 | 47 | 1227 | 11085 | 0.036891680 | 0.004222063 | 0.007576979 |

正解11,135件は、文書内の重複と公式正規化による重複を除くと11,132件です。予測1,274件のうち47件が一致しました。
初期の転用＋全文ルールはTP=0でした。共有事実を分けたholdoutの性質とtrain本文を確認し、train用語辞書による新しい組合せの生成を追加しました。validationの集計結果を見て実装を一度更新したため、完全に未観測の最終テスト結果ではありません。個別のvalidation正解・成功失敗例を使ったルール作成や閾値探索は行っていません。

## 成功例5件（部分成功）

正解トリプルを1件以上含む文書をF1降順、ID順に選びました。以下は各文書の代表的な正解1件です。文書全体の完全正解を意味しません。

### 1. `df11445b146c4aaeb5554514c4ede6f0`

Title: Mathematical Frameworks in Environmental Science

`binary operation` → **is-a** → `operation`

TP=5 / FP=2 / FN=3、F1=0.666667。

### 2. `9eb707300b074efc9f94ac693b0e7c3e`

Title: Culinary Diversity and Taxonomic Groupings in Agriculture

`chenopodium pallidicaule` → **is-a** → `chenopodium`

TP=2 / FP=3 / FN=3、F1=0.400000。

### 3. `0d909cb0d9584989a5a54fbce9128386`

Title: Understanding Lamb Cuts and Nutritional Chemistry

`retinol` → **is-a** → `primary allylic alcohol`

TP=1 / FP=0 / FN=4、F1=0.333333。

### 4. `a67907b1f9d940479bb3b66d71999c41`

Title: Agricultural Diversity: Taxonomy and Preparation

`elephas maximus` → **is-a** → `elephantidae`

TP=1 / FP=0 / FN=4、F1=0.333333。

### 5. `bbff51cfcc4b4a32b5871bda9b839d2a`

Title: The Taxonomic Diversity of Global Forest Biomes

`tropical forest` → **is-a** → `forest`

TP=1 / FP=0 / FN=4、F1=0.333333。

## 失敗例5件

誤りを含む文書をF1昇順、ID順に選びました。FPは注釈との不一致であり、現実世界で偽とは限りません。

### 1. `001e034f50924f959c34fc020a76122e`

Title: Diverse Structural Manifestations of Carbon Allotropes

予測は空でした。

見逃し例: `aggregated carbon nanorods` → **is-a** → `carbon allotrope`

TP=0 / FP=0 / FN=5、F1=0.000000。

### 2. `005eceb71f8b4377b8aff8436b19556e`

Title: Culinary Butchery and Ingredient Overview

予測は空でした。

見逃し例: `beef brisket (raw)` → **is-a** → `beef brisket`

TP=0 / FP=0 / FN=5、F1=0.000000。

### 3. `006ff9ac8e70474fa8468dd26e736394`

Title: Clinical Spectrum and Genetic Etiology of Familial Chylomicronemia Syndrome

誤検出: `familial lipoprotein lipase deficiency` → **is-a** → `autosomal recessive disease`

見逃し例: `familial apolipoprotein a5 deficiency` → **is-a** → `familial chylomicronemia syndrome`

TP=0 / FP=1 / FN=6、F1=0.000000。

### 4. `0084c7e04a564d58bb9e51a1ed0ebacb`

Title: Food Chemistry and Product Classification: Saccharin and Pasta Categories

予測は空でした。

見逃し例: `10000240 - pasta/noodles - ready to eat (perishable) (gs1 gpc)` → **is-a** → `50192900 - pasta/noodles (gs1 gpc)`

TP=0 / FP=0 / FN=7、F1=0.000000。

### 5. `00857be4a7494939aa6bb960c3d04839`

Title: Taxonomic Classification of Papayas and Similar Fruit Groupings

予測は空でした。

見逃し例: `17480 - papayas (efsa foodex2)` → **is-a** → `17470 - papayas and similar- (efsa foodex2)`

TP=0 / FP=0 / FN=6、F1=0.000000。

## 解釈・再現性

辞書にない概念、番号付き分類名、言い換え、長い節や列挙への対応が弱く、再現率は低い状態です。train内の共起トリプル転用は、対象文書で同じ関係が成立することを保証しません。用語の一部との一致や一般語も誤検出につながります。instance-ofとその他の関係は今回は正解0件です。
公式Taxonomy Similarityは祖先と子孫の和集合を比較するため、方向だけの誤りを見逃す場合があります。方向を含む完全トリプルF1を併記しています。
予測はid/contextのみの別ファイルから生成し、保存後に評価プロセスが正解を参照しました。trainの用語辞書・IDF・個体候補は実験trainだけにfitしています。
全91テスト成功（ネットワーク接続を禁止、CUDA無効）。Ruff成功。全pre-commitは既存3ファイルの末尾改行・末尾空白で不合格。公式原文のハッシュを維持するため修正していません。

保存物: [予測](results/v2/predictions.json)、[マニフェスト](results/v2/predictions.manifest.json)、[全評価・文書別誤り・10例の全文](results/v2/scored/evaluation.json)、[既存アダプター形式](results/v2/scored/official-predictions.json)。

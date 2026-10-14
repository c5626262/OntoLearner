# 公開trainの固定分割

researchにマージ済みの評価基盤（`8ad57b2b99dfe7ca390b0f7ea0c96656e68db9f8`）を使用。
公式testは取得・分析していません。モデル推論や外部APIによる類似度判定も行いません。

## 保存した分割

| 項目 | 件数 |
|---|---:|
| 公開train全体 | 4,303文書 / 55,606トリプル |
| 実験train | 3,442文書 / 44,471トリプル |
| validation | 861文書 / 11,135トリプル |

validation文書数は `ceil(4303 * 0.2) = 861`、20.0093%です。
グループを分割しない制約の下で、今回は目標件数に正確に一致しました。

`splits/seed42/` に以下を保存しています。文書本文や正解トリプルのコピーは含みません。

- `train_ids.json`、`validation_ids.json`: ソート済みID一覧
- `manifest.json`: seed、方式、閾値、丸め、候補選択条件、入力/IDのハッシュ、依存バージョン
- `relation_distribution.json`: 全28ラベルのトリプル数・文書数・割合と欠落ラベル
- `leakage_audit.json`: 分割間の重複・語彙類似・共有概念に関する診断

## 分割方式

1. 文書をID順に固定する。乱数はローカルの `random.Random(42)` のみを使用。
2. 次のいずれかで結ばれる文書を同一グループにする。
   - subject/predicate/objectを小文字化・空白正規化したトリプルが一致する。
   - contextを小文字化・空白正規化すると一致する。
   - word/bigram TF-IDF cosineが0.80以上である。
3. このグラフの連結成分を分割の最小単位にする。全3,521グループ、最大416文書。
4. seed=42から64通りのグループ順序を生成する。subset-sumで目標に最も近い件数を選ぶ。
5. 件数誤差、関係件数の重み付き誤差、validation IDの辞書順で候補を比較する。
   関係誤差は `sum((val_count - 0.2 * full_count)^2 / max(full_count, 1))`。
   採用候補の0始まり番号は4。多ラベルの分布を近づける方式で、完全な層化保証はない。

グループ制約で目標件数が実現できない場合は最も近い非空の分割を選び、条件に記録します。
全件が一つの漏洩グループになる場合は、グループを壊さずエラーにします。

TF-IDFは分割を作るためだけに公開train全体へfitします。
将来の抽出モデル・few-shot選択・Retriever・クラスタリングの学習は実験trainだけにfitしてください。
validationの正解は閾値選択・評価に用いてよいですが、予測時の入力として使いません。

## 関係分布

| 区分 | trainトリプル | train内比率 | validationトリプル | validation内比率 |
|---|---:|---:|---:|---:|
| is-a | 39,647 | 89.15% | 9,885 | 88.77% |
| instance-of | 3,816 | 8.58% | 1,060 | 9.52% |
| その他 | 1,008 | 2.27% | 190 | 1.71% |

希少関係は、公開train全体で出現文書数が5以下のラベルと定義しています。
validationから消える13ラベルは以下です。

`see also`, `same as`, `range`, `derives from`, `develops_from`, `broader`, `has role`,
`located in`, `overlaps`, `positively regulates`, `regulates`, `ro_0002220`, `ro_0002473`。

`term replaced by` は全体で1文書/1トリプルしかなく、今回validation側だけに存在します。
単一文書のラベルを両側へ出すには漏洩または追加データが必要になるため、複製していません。
欠落ラベルのvalidation性能は測れず、trainにないラベルは未学習関係の評価になります。

## 漏洩検査

- ID重複: 0
- 分割間の共有正規化トリプル: 0
- 分割間の重複正規化context: 0
- 全文書のcosine 0.80以上の9ペア: すべて同じ側に配置
- 分割間のcosine 0.70以上0.80未満: 17ペア、最大0.7982717026599467
- 分割間で共通する正規化ノード文字列: 3,203

したがって文書・共有事実を分けたholdoutであり、エンティティ非重複や分野非重複は保証しません。
意味的な言い換えによる漏洩も、モデル不要の語彙類似検査だけでは排除できません。
0.70未満のペアは検出一覧に含めていません。

参考診断としてseed=42で文書を単純にshuffleして分割すると、共有正規化トリプル463種類、
cosine 0.80以上の分割間ペア2組が残りました。この単純分割は実験ID一覧には採用していません。

## 再現と推論入力

```bash
.venv/bin/python -m research.llms4ol2026 split \
  --train /workspace/llms4ol2026-data/train_task_a.json \
  --seed 42 --validation-fraction 0.2 --candidates 64 --near-threshold 0.8 \
  --output /tmp/llms4ol2026-reproduced-split

.venv/bin/python -m pytest tests -q -p no:cacheprovider
```

Python 3.12.14、NumPy 2.3.5、scikit-learn 1.8.0、NetworkX 3.2.1で検証。
同じ入力・コード・環境・条件で5ファイルがbyte単位で一致することを確認しました。
異なる既存の分割ファイルを上書きする操作は拒否します。

保存したIDから文書を取り出す場合は、IDリストのハッシュと元trainのハッシュをmanifestで
照合し、各 `TrainingSample.inference_input()` を呼び出します。
分割作成直後は `inference_documents(samples, split, "validation")` でも同じid/contextだけを取得できます。
返る `InferenceDocument` にtriplesフィールドはありません。

追加18テストは、再現性、入力順序不変、グループ原子性、近似件数、seed、
保存・上書き拒否、分布、漏洩診断、推論入力から正解を除く契約を検証します。
公式評価器とそのアダプターには変更を加えていません。

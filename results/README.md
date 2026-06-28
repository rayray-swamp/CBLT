# results — エントロピーモデル評価・較正の結果

CBLT エントロピーモデル（凍結 step 285,000）まわりの計算結果。

| ファイル | 内容 | 生成スクリプト |
|---|---|---|
| `entropy_auc_curve_checkpoints.tsv` | checkpoint 別 ja 境界AUC（heldout_jawiki, 修正済eval）。凍結ckpt選択の根拠。step / ja_AUC / 境界平均H / 非境界H。265-275kで頭打ち~0.712、285k凍結。 | `scripts/qsub_auc_curve.sh` → `scripts/cblt_eval_corpus.py` |
| `ja_raw_5sentences.csv` | 日本語5文の生データ（文字 / バイト(hex) / 生成エントロピー(bits, BLTバイト単位) / sum / sum/√ℓ=H）。CBLT 規約B（予測バイト範囲で√ℓ集約）の確認用。 | `scripts/ja_raw_csv.py` |
| `ja_raw_wiki_corpus_samples.csv` | wiki+コーパスからランダム抽出した日本語文の同形式 raw データ。 | `scripts/ja_raw_csv_samples.py` |

関連する FLORES+ 系の結果（per-char ダンプ、monotonic θグリッド、本データθ較正表、CBLT θドリフト）は `../flores_dumps/` を参照。

凍結モデル: `blt_runs/entropy_stage_run/frozen_entropy_285000/consolidated/consolidated.pth`（リポジトリ外・TSUBAME上）。

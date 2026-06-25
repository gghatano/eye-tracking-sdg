# 合成アイトラッキングデータ生成器（Synthetic Eye-Tracking Data Generator）

**読解タスク**を対象とした合成アイトラッキングデータの生成器です。視線行動を2系統の生成器で合成し、**同一の評価基盤**で比較します。

- **ルールベース生成器** — 視線行動に関する仮定・分布・遷移ルールから合成（学習データ不要）。
- **モデルベース生成器** — 参照データから統計パラメータ・遷移確率を学習してサンプリング。

両者は**同一スキーマ**（イベント単位 / raw gaze 時系列 / AOIサマリ）で出力するため、同じ指標で品質比較できます。さらに、合成データで学習した検出器が実データへ転移するか（**sim-to-real**）を検証する仕組みも含みます。

📄 **解説レポート（可視化・文献的根拠・モデル説明・sim-to-real結果）**: https://gghatano.github.io/eye-tracking-sdg/

---

## できること

- fixation / saccade / blink の **イベント単位データ**を生成
- 任意サンプリング周波数（既定120Hz）の **raw gaze 時系列**を生成
- **AOI単位のサマリ特徴量**を生成
- 分布・系列・プライバシーの各指標で**生成品質を評価**
- **ルールベース vs モデルベース**を同一指標で比較
- 合成データで**イベント検出器を学習し、実データ（Lund2013）で転移評価**

## 再現する視線行動

fixation、saccade、前方読み、回帰（再読）、スキップ、blink / 欠損、被験者間の個人差、刺激の難易度効果。

---

## インストール（uv）

[uv](https://docs.astral.sh/uv/) を使用します。

```bash
uv sync
```

## 使い方（CLI）

```bash
# ルールベース生成
uv run synthetic-eye-tracking generate-rule \
  --config configs/rule_based.yaml \
  --output data/synthetic/rule_based/

# モデル学習（参照イベントCSVから）
uv run synthetic-eye-tracking fit-model \
  --config configs/model_based.yaml \
  --input data/processed/reference_events.csv \
  --model-output data/processed/model.pkl

# モデルベース生成
uv run synthetic-eye-tracking generate-model \
  --config configs/model_based.yaml \
  --model data/processed/model.pkl \
  --output data/synthetic/model_based/

# 評価（参照 vs 合成の比較レポート）
uv run synthetic-eye-tracking evaluate \
  --reference data/processed/reference_events.csv \
  --synthetic data/synthetic/model_based/events.csv \
  --output data/reports/model_based_report/

# sim-to-real イベント検出（合成で学習→実データで評価）
uv run synthetic-eye-tracking detect-events --output data/reports/detection
```

`generate-rule` は `events.csv` / `raw_gaze.csv` / `aoi_summary.csv` ほか、設定・品質レポート（`quality_report.md` / `.json`）・各種プロットを出力します。

---

## 出力データ

| ビュー | 粒度 | 主な列 |
|--------|------|--------|
| イベント単位 | fixation/saccade/blink 1行 | `event_type`, `start/end/duration_ms`, `x_px`, `y_px`, `aoi_id`, `saccade_amp_px`, `validity` |
| raw gaze | サンプル1行（120Hz） | `timestamp_ms`, `x_px`, `y_px`, `pupil_size`, `validity`, `event_type` |
| AOIサマリ | (被験者×試行×AOI) 1行 | `first_fixation_duration_ms`, `total_fixation_duration_ms`, `fixation_count`, `regression_count` ほか |

---

## 設定（YAML）

`configs/` 配下の YAML で全パラメータを制御します（被験者数・刺激・サンプリング周波数・参加者分布・イベント分布・遷移確率・blink など）。詳細は `configs/default.yaml` を参照してください。

## リポジトリ構成

```text
src/synthetic_eye_tracking/
  config.py / schema.py        # 設定モデルと共通データスキーマ（契約）
  aoi.py / participants.py / stimuli.py / events.py / raw_gaze.py
  generators/                  # base / rule_based / model_based
  models/                      # parametric / markov / hmm（任意）
  evaluation/                  # metrics / plots / report
  detection/                   # sim-to-real イベント検出（M3-A）
  io/ , cli.py
configs/  data/  scripts/  tests/  docs/
```

---

## マイルストーン

- **M1 ✅** ルールベース生成器、reading-grid AOIレイアウト、fixation/saccade/blink、120Hz raw gaze 展開、基本評価レポート
- **M2 ✅** 疑似参照データへのフィッティング、パラメトリック + マルコフ遷移のモデルベース生成器、ルール vs モデル比較
- **M3-A ✅** 合成データで学習したイベント検出器の sim-to-real 転移評価（実データ: Lund2013 / Andersson et al. 2017）

### sim-to-real 結果（要約）

実データ（Lund2013, 人手ラベル）でのテスト:

| 学習データ | macro F1 | fixation F1 | saccade F1 |
|---|---|---|---|
| 合成のみ (TSTR) | 0.717 | 0.948 | 0.485 |
| 実のみ (TRTR) | 0.942 | 0.985 | 0.898 |
| 実+合成 (R+S) | 0.931 | 0.984 | 0.879 |

**合成データのみ・人手ラベルゼロで、実データ学習の約76%の性能**に到達。fixation検出はほぼ完璧に転移する一方、saccade検出が弱点（現実的なサッケード力学の不足）。詳細・図・注意点はレポートを参照してください。

> 注意: Lund2013は読解ではなく自由視のため、これは「汎用ドメインの転移」です。

---

## 開発

```bash
uv sync
uv run pytest                                  # テスト一式
uv run python scripts/make_report_figures.py   # レポート用の図を再生成
uv run --with markdown python scripts/build_html.py   # docs/index.html を再生成
```

すべて乱数シード固定（`seed = 42`）のため、図・統計は同一に再現されます。

## 実装方針

説明可能な生成器を優先し、raw gaze を直接作らず**イベント単位→raw gaze 展開**の順で生成します。ルールベースとモデルベースは同一スキーマで出力し、評価は分布一致だけでなく系列構造・下流タスク有用性・プライバシーも見ます。実データを使う場合は個人再識別リスクの proxy 評価を含めます。

## ライセンス

MIT License（`pyproject.toml`）。sim-to-real 評価で使用する Lund2013 データセットは GPL-3.0 で、初回実行時に GitHub から取得します（リポジトリには含めません）。

"""正方形領域上の合成視線（line-of-sight / gaze）データ生成器。

数理モデル: Constrained Lévy Exploration (CLE)
    Boccignone, G. & Ferraro, M. (2004),
    "Modelling gaze shift as a constrained random walk", Physica A 331, 207-218.

概要:
    視線移動（サッケード）を「サリエンシー地形 s(x,y) に制約された Lévy フライト」
    としてモデル化する。

      1. サッケード振幅 l は重い裾を持つ Lévy/Cauchy 型分布から引く
         （P(l) ~ l^{-(1+alpha)}）。短い跳躍が大半だが、稀に大きく飛ぶ
         （ヒトの探索的眼球運動に観測される性質）。
      2. 跳躍方向は一様ランダム。
      3. 提案された移動先は Metropolis 則でサリエンシーに基づき採否を決める。
         より顕著（salient）な点へは必ず移動、そうでなければ
         exp(-Δs / T) の確率で受理 → 顕著領域へ自然に引き寄せられる。
      4. 注視（fixation）中は微小な固視微動を Ornstein–Uhlenbeck 過程で表現。
      5. 注視時間はガンマ分布（平均 ~250 ms）から引く。

標準ライブラリのみで動作（numpy 不要）。seed 固定で再現可能。
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

# ----------------------------------------------------------------------------
# パラメータ
# ----------------------------------------------------------------------------
SEED = 42
SQUARE = 1000.0          # 正方形領域の一辺 [px]
N_FIXATIONS = 90         # 生成する注視点の数
SAMPLE_HZ = 250          # raw gaze のサンプリング周波数 [Hz]

# Lévy フライト
ALPHA = 1.6              # 裾の重さ（1<alpha<3。小さいほど長距離跳躍が増える）
STEP_SCALE = 70.0        # サッケード振幅のスケール [px]

# Metropolis 受理（サリエンシー制約）
TEMP = 0.18              # 温度。小さいほど顕著領域に強く吸着

# 注視時間（ガンマ分布）
FIX_DUR_SHAPE = 4.0
FIX_DUR_SCALE = 62.5     # 平均 = shape*scale = 250 ms

# 固視微動（Ornstein–Uhlenbeck）
OU_THETA = 0.08          # 平均回帰の強さ
OU_SIGMA = 3.2           # 揺らぎの大きさ [px/sqrt(step)]

# サッケード遷移（raw gaze 補間用）
SACCADE_MS = 40.0        # 1サッケードの所要時間 [ms]


# ----------------------------------------------------------------------------
# サリエンシー地形: 正方形内に置いた複数のガウシアン「関心領域」の和
# ----------------------------------------------------------------------------
SALIENCY_BLOBS = [
    # (cx, cy, sigma, weight)
    (0.27 * SQUARE, 0.30 * SQUARE, 0.10 * SQUARE, 1.0),
    (0.72 * SQUARE, 0.26 * SQUARE, 0.13 * SQUARE, 0.9),
    (0.52 * SQUARE, 0.58 * SQUARE, 0.16 * SQUARE, 1.1),
    (0.20 * SQUARE, 0.78 * SQUARE, 0.09 * SQUARE, 0.7),
    (0.80 * SQUARE, 0.80 * SQUARE, 0.11 * SQUARE, 0.85),
]


def saliency(x: float, y: float) -> float:
    """点 (x,y) のサリエンシー値（0..~1超）を返す。"""
    s = 0.0
    for cx, cy, sigma, w in SALIENCY_BLOBS:
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        s += w * math.exp(-d2 / (2.0 * sigma ** 2))
    return s


def levy_step(rng: random.Random) -> float:
    """重い裾を持つ Lévy/Pareto 型のサッケード振幅を引く。"""
    u = rng.random()
    # 逆関数法による Pareto 型: l = scale * u^{-1/alpha}
    return STEP_SCALE * (u ** (-1.0 / ALPHA) - 1.0)


def gamma_sample(rng: random.Random, shape: float, scale: float) -> float:
    """ガンマ分布サンプル（標準ライブラリの random.gammavariate）。"""
    return rng.gammavariate(shape, scale)


def in_square(x: float, y: float) -> bool:
    return 0.0 <= x <= SQUARE and 0.0 <= y <= SQUARE


# ----------------------------------------------------------------------------
# CLE による注視列の生成
# ----------------------------------------------------------------------------
def generate_fixations(rng: random.Random) -> list[dict]:
    # 初期注視点は領域中央付近
    x, y = SQUARE * 0.5, SQUARE * 0.5
    s_cur = saliency(x, y)

    fixations: list[dict] = []
    t_ms = 0.0

    for i in range(N_FIXATIONS):
        dur = gamma_sample(rng, FIX_DUR_SHAPE, FIX_DUR_SCALE)
        fixations.append(
            {
                "index": i,
                "x": round(x, 2),
                "y": round(y, 2),
                "start_ms": round(t_ms, 1),
                "duration_ms": round(dur, 1),
                "saliency": round(s_cur, 4),
            }
        )
        t_ms += dur + SACCADE_MS

        # --- 次の注視点を CLE で提案 ---
        # Metropolis 受理されるまで Lévy 跳躍を再提案
        for _attempt in range(40):
            length = levy_step(rng)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            nx = x + length * math.cos(angle)
            ny = y + length * math.sin(angle)
            if not in_square(nx, ny):
                continue
            s_new = saliency(nx, ny)
            # サリエンシー制約付き Metropolis 則
            if s_new >= s_cur or rng.random() < math.exp((s_new - s_cur) / TEMP):
                x, y, s_cur = nx, ny, s_new
                break

    return fixations


# ----------------------------------------------------------------------------
# 注視列 → raw gaze 時系列に展開（固視微動 + サッケード補間）
# ----------------------------------------------------------------------------
def expand_raw_gaze(rng: random.Random, fixations: list[dict]) -> list[dict]:
    dt = 1000.0 / SAMPLE_HZ
    samples: list[dict] = []

    for i, fx in enumerate(fixations):
        # --- 固視微動（Ornstein–Uhlenbeck で中心に回帰しつつ揺れる）---
        cx, cy = fx["x"], fx["y"]
        px, py = cx, cy
        n = max(1, int(fx["duration_ms"] / dt))
        t = fx["start_ms"]
        for _ in range(n):
            px += -OU_THETA * (px - cx) + rng.gauss(0.0, OU_SIGMA)
            py += -OU_THETA * (py - cy) + rng.gauss(0.0, OU_SIGMA)
            samples.append(
                {"t": round(t, 1), "x": round(px, 2), "y": round(py, 2), "phase": "fixation"}
            )
            t += dt

        # --- 次の注視点へのサッケード（最小ジャーク的な滑らかな補間）---
        if i + 1 < len(fixations):
            nx, ny = fixations[i + 1]["x"], fixations[i + 1]["y"]
            m = max(1, int(SACCADE_MS / dt))
            for k in range(1, m + 1):
                u = k / m
                # 最小ジャーク速度プロファイル: 6u^5 -15u^4 +10u^3
                e = 6 * u**5 - 15 * u**4 + 10 * u**3
                samples.append(
                    {
                        "t": round(t, 1),
                        "x": round(cx + (nx - cx) * e, 2),
                        "y": round(cy + (ny - cy) * e, 2),
                        "phase": "saccade",
                    }
                )
                t += dt

    return samples


def build_saliency_grid(res: int = 60) -> list[list[float]]:
    """可視化の背景用に粗いサリエンシー格子を作る。"""
    grid = []
    for j in range(res):
        row = []
        for i in range(res):
            x = (i + 0.5) / res * SQUARE
            y = (j + 0.5) / res * SQUARE
            row.append(round(saliency(x, y), 4))
        grid.append(row)
    return grid


def main() -> None:
    rng = random.Random(SEED)
    fixations = generate_fixations(rng)
    raw = expand_raw_gaze(rng, fixations)

    amps = [
        math.hypot(fixations[i + 1]["x"] - fixations[i]["x"],
                   fixations[i + 1]["y"] - fixations[i]["y"])
        for i in range(len(fixations) - 1)
    ]
    payload = {
        "model": "Constrained Lévy Exploration (Boccignone & Ferraro, 2004)",
        "seed": SEED,
        "square_px": SQUARE,
        "sample_hz": SAMPLE_HZ,
        "params": {
            "alpha": ALPHA,
            "step_scale": STEP_SCALE,
            "temperature": TEMP,
            "fix_dur_mean_ms": FIX_DUR_SHAPE * FIX_DUR_SCALE,
            "ou_theta": OU_THETA,
            "ou_sigma": OU_SIGMA,
        },
        "stats": {
            "n_fixations": len(fixations),
            "n_raw_samples": len(raw),
            "total_duration_ms": round(fixations[-1]["start_ms"] + fixations[-1]["duration_ms"], 1),
            "mean_saccade_amp_px": round(sum(amps) / len(amps), 1),
            "max_saccade_amp_px": round(max(amps), 1),
        },
        "saliency_blobs": [
            {"cx": b[0], "cy": b[1], "sigma": b[2], "weight": b[3]} for b in SALIENCY_BLOBS
        ],
        "saliency_grid": build_saliency_grid(),
        "fixations": fixations,
        "raw_gaze": raw,
    }

    out_dir = Path(__file__).parent
    json_path = out_dir / "los_data.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {json_path}  ({len(fixations)} fixations, {len(raw)} raw samples)")

    # HTML テンプレートにデータを埋め込んで単一ファイルの自己完結デモを生成
    tmpl_path = out_dir / "index.template.html"
    if tmpl_path.exists():
        html = tmpl_path.read_text(encoding="utf-8")
        html = html.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
        (out_dir / "index.html").write_text(html, encoding="utf-8")
        print(f"wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

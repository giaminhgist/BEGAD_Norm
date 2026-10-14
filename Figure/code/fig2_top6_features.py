"""Figure 2 - the 6 processed_dataset features with the smallest p-values.

One Kruskal-Wallis test per (feature, task) over the 25 features x 4 test
tasks (King-Devick cards 3, 4, 5 and the reading task 9; = 100 tests);
groups HC / MCI / AD. Features are ranked by their smallest-p task and the
top 6 distinct features are selected (greedy over the ascending p list, so
no feature repeats). Each panel shows the feature at the task with its
smallest p.

Sources:
  - processed_dataset/features.csv + metadata.csv (CO relabelled HC)

Outputs:
  - Figure/outputs/fig2_top6_features.svg (+ .png preview, 300 dpi)
  - Figure/outputs/fig2_feature_task_pvalues.csv (all 100 tests)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

GROUP_ORDER = ["HC", "MCI", "AD"]
COLORS = {"HC": "#0072B2", "MCI": "#E69F00", "AD": "#D55E00"}  # Okabe-Ito
N_PANELS = 6
TASKS = [3, 4, 5, 9]  # King-Devick cards 3-5 + reading task


def kruskal_p(f, feat, task):
    sub = f[f.task == task]
    vals = [sub[sub.group_label == g][feat].dropna().to_numpy()
            for g in GROUP_ORDER]
    if any(len(v) == 0 for v in vals):
        return np.nan
    return kruskal(*vals).pvalue

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- data ---
    f = pd.read_csv(ROOT / "processed_dataset" / "features.csv")
    meta = pd.read_csv(ROOT / "processed_dataset" / "metadata.csv")
    f = f.merge(meta[["participant_id", "group_label"]], on="participant_id")
    f["group_label"] = f["group_label"].replace({"CO": "HC"})
    feats = [c for c in f.columns
             if c not in ("participant_id", "task", "group", "group_label")]

    # --- 100 Kruskal-Wallis tests (25 features x 4 tasks), then top-6 ---
    rows = [(feat, task, kruskal_p(f, feat, task))
            for feat in feats for task in TASKS]
    ptab = pd.DataFrame(rows, columns=["feature", "task", "p"])
    ptab = ptab.sort_values("p", kind="stable").reset_index(drop=True)
    ptab.to_csv(OUT_DIR / "fig2_feature_task_pvalues.csv", index=False)

    picked, seen = [], set()
    for r in ptab.itertuples(index=False):
        if r.feature in seen:
            continue
        seen.add(r.feature)
        picked.append((r.feature, r.task, r.p))
        if len(picked) == N_PANELS:
            break
    print("top-6:")
    for feat, task, p in picked:
        print(f"  {feat:20s} task {task:2d} p={p:.3e}")

    # --- figure: 2 x 3 panels ---
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.6),
                             gridspec_kw={"wspace": 0.55, "hspace": 0.45})
    axes = axes.ravel()
    rng = np.random.default_rng(0)
    for k, (ax, (feat, task, p)) in enumerate(zip(axes, picked)):
        sub = f[f.task == task]
        by_group = [sub[sub.group_label == g][feat].dropna().to_numpy()
                    for g in GROUP_ORDER]
        ax.boxplot(by_group, positions=[1, 2, 3], widths=0.45,
                   showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "white", "edgecolor": "black",
                             "linewidth": 1},
                   medianprops={"color": "black", "linewidth": 1.4},
                   whiskerprops={"color": "black", "linewidth": 1},
                   capprops={"color": "black", "linewidth": 1})
        for i, x in enumerate(by_group):
            jx = rng.normal(i + 1, 0.05, len(x))
            ax.scatter(jx, x, s=9, color=COLORS[GROUP_ORDER[i]],
                       alpha=0.5, edgecolors="none", zorder=3)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_ylabel(feat)
        ax.text(-0.12, 1.06, f"({chr(97 + k)})", transform=ax.transAxes,
                 fontsize=10, fontweight="bold")
        ax.text(1.0, 1.06, f"task {task}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
        med = [np.median(x) for x in by_group]
        print(f"  [{chr(97 + k)}] {feat} task {task}: "
              f"n={[len(x) for x in by_group]} "
              f"medians={[f'{m:.3g}' for m in med]}")

    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"fig2_top6_features.{ext}",
                    dpi=300 if ext == "png" else None)
    print(f"saved -> {OUT_DIR / 'fig2_top6_features.svg'} (+ .png)")


if __name__ == "__main__":
    main()

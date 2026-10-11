"""Week-1 report figures: CO/MCI/AD comparison for the 8 features with
Kruskal-Wallis p < 0.1 in all three King-Devick cards (tasks 3, 4, 5).

Style follows EDA/eda_group_comparison.ipynb (box + jitter, same palette),
except that panel titles are replaced by corner "Task N" labels: statistics
go into the report captions (no titles inside figures).

Usage:
    python plot_week1_figures.py --feature pup_mean
    python plot_week1_figures.py --all
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FEATURES_CSV = ROOT / "processed_dataset" / "features.csv"
META_CSV = ROOT / "processed_dataset" / "metadata.csv"
STATS_CSV = ROOT / "EDA" / "group_comparison_stats_per_task.csv"
FIG_DIR = Path(__file__).resolve().parent

TASKS = [3, 4, 5]
GROUPS = ["CO", "MCI", "AD"]
COLORS = {"CO": "#0072B2", "MCI": "#E69F00", "AD": "#009E73"}

# The 8 features with kw_p < 0.1 in all of tasks 3, 4, 5 (unit per
# EDA/features_dictionary.md).
FEATURES8 = {
    "geo_nn_dist_mean": "px",
    "geo_revisit_rate": "fraction",
    "geo_sacc_amp_mean": "px",
    "pup_mean": "mm",
    "pup_median": "mm",
    "sac_count": "count",
    "sac_dur_total": "ms",
    "spa_fix_count": "count",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

rng = np.random.default_rng(42)  # fixed jitter seed -> reproducible panels


def load_data():
    feat = pd.read_csv(FEATURES_CSV)
    meta = pd.read_csv(META_CSV)[["participant_id", "group_label"]]
    df = feat.merge(meta, on="participant_id", how="left")
    stats = pd.read_csv(STATS_CSV)
    return df, stats


def draw_panel(ax, df, stats, task, feature):
    for gi, g in enumerate(GROUPS):
        vals = df[(df["task"] == task) & (df["group_label"] == g)]
        vals = vals[[feature, "participant_id"]].dropna(subset=[feature])
        vals = vals[feature].to_numpy()
        # data-honesty check: plotted points must match the group n of the
        # Kruskal-Wallis test in the stats file
        row = stats[(stats["task"] == task) & (stats["feature"] == feature)]
        assert not row.empty, (task, feature)
        n_stat = int(row[f"n_{g}"].iloc[0])
        assert len(vals) == n_stat, (task, feature, g, len(vals), n_stat)
        ax.boxplot(
            vals, positions=[gi], widths=0.45, patch_artist=True,
            boxprops=dict(facecolor=COLORS[g], alpha=0.35, lw=1.0),
            medianprops=dict(color="black", lw=1.2),
            whiskerprops=dict(color=COLORS[g]), capprops=dict(color=COLORS[g]),
            flierprops=dict(marker=".", markersize=3, alpha=0.5,
                            markeredgecolor=COLORS[g]),
        )
        if len(vals):
            jx = gi + rng.normal(0, 0.055, len(vals))
            ax.scatter(jx, vals, s=5, alpha=0.30, color=COLORS[g],
                       linewidths=0, rasterized=True)
    ax.set_xticks(range(3))
    ns = []
    for g in GROUPS:
        row = stats[(stats["task"] == task) & (stats["feature"] == feature)]
        ns.append(int(row[f"n_{g}"].iloc[0]))
    ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(GROUPS, ns)],
                       fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    # Panel identifier above the axes (structural label, not a title):
    # statistics stay in the report captions.
    ax.set_title(f"Task {task}", loc="left", fontweight="bold", fontsize=9,
                 pad=3)


def make_figure(df, stats, feature, unit):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    for ax, task in zip(axes, TASKS):
        draw_panel(ax, df, stats, task, feature)
    axes[0].set_ylabel(f"{feature} ({unit})")
    fig.tight_layout()
    svg = FIG_DIR / f"fig_{feature}.svg"
    png = FIG_DIR / f"fig_{feature}.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"saved {svg.name} + {png.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=sorted(FEATURES8) + ["all"],
                    default="all")
    args = ap.parse_args()
    df, stats = load_data()
    feats = sorted(FEATURES8) if args.feature == "all" else [args.feature]
    for f in feats:
        make_figure(df, stats, f, FEATURES8[f])
    print(f"done: {len(feats)} figure(s); point counts asserted equal to "
          f"the Kruskal-Wallis group n in {STATS_CSV.name}")


if __name__ == "__main__":
    main()

"""Figure 1 - BEGAD dataset summary.

Panel (a): participant counts per group (HC / MCI / AD).
Panel (b): exported SOB score by group, with Kruskal-Wallis across the
three groups and pairwise Mann-Whitney U (Wilcoxon rank-sum) tests with
Benjamini-Hochberg correction (significance brackets on the plot).

Sources:
  - dataset_v2/participants_20250305.csv (roster; CO relabelled HC)
  - dataset_v2/20250305/<file>.csv (SOB column, constant per participant)

Note: the SOB field is not documented in this repository. Its 0.5-step
distribution is consistent with a CDR-SOB-like scale (unverified), so the
axis uses the neutral export-field label.

Outputs:
  - Figure/outputs/fig1_begad_dataset_summary.svg (+ .png preview, 300 dpi)
  - Figure/outputs/fig1_sob_by_participant.csv (per-participant SOB table)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "dataset_v2" / "20250305"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

GROUP_ORDER = ["HC", "MCI", "AD"]
COLORS = {"HC": "#0072B2", "MCI": "#E69F00", "AD": "#D55E00"}  # Okabe-Ito

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def bh_correct(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    idx = np.argsort(p)
    q = p[idx] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[idx] = q
    return out


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- data: roster + per-participant SOB from the raw exports ---
    roster = pd.read_csv(ROOT / "dataset_v2" / "participants_20250305.csv")
    rows = []
    for rec in roster.itertuples(index=False):
        df = pd.read_csv(RAW_DIR / rec.filename, usecols=["SOB"])
        vals = df["SOB"].dropna().unique()
        assert len(vals) == 1, f"{rec.participant_id}: SOB not constant {vals}"
        rows.append((rec.participant_id, rec.group_label, float(vals[0])))
    sob = pd.DataFrame(rows, columns=["participant_id", "group", "sob"])
    sob["group"] = sob["group"].replace({"CO": "HC"})
    sob.to_csv(OUT_DIR / "fig1_sob_by_participant.csv", index=False)

    # --- statistics ---
    by_group = [sob[sob.group == g]["sob"].to_numpy() for g in GROUP_ORDER]
    counts = [len(x) for x in by_group]
    kw = kruskal(*by_group)
    pairs = [("HC", "MCI"), ("HC", "AD"), ("MCI", "AD")]
    pvals = [
        mannwhitneyu(sob[sob.group == a]["sob"], sob[sob.group == b]["sob"],
                     alternative="two-sided").pvalue
        for a, b in pairs
    ]
    bh = bh_correct(pvals)
    print(f"counts: {dict(zip(GROUP_ORDER, counts))}")
    print(f"Kruskal-Wallis: H={kw.statistic:.3f}, p={kw.pvalue:.3e}")
    for (a, b), p, q in zip(pairs, pvals, bh):
        print(f"  {a} vs {b}: p={p:.3e}, BH={q:.3e} ({stars(q)})")
    for g, x in zip(GROUP_ORDER, by_group):
        print(f"  {g}: median={np.median(x):.2f}, "
              f"IQR=[{np.percentile(x, 25):.2f}, {np.percentile(x, 75):.2f}], "
              f"range=[{x.min():.2f}, {x.max():.2f}]")

    # --- figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2), gridspec_kw={"wspace": 0.35})

    # (a) participant counts
    bars = ax1.bar(GROUP_ORDER, counts, width=0.55,
                   color=[COLORS[g] for g in GROUP_ORDER])
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, c + 2, str(c),
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Number of participants")
    ax1.set_ylim(0, 112)
    ax1.text(-0.12, 1.04, "(a)", transform=ax1.transAxes,
             fontsize=10, fontweight="bold")

    # (b) SOB by group
    positions = np.arange(1, 4)
    ax2.boxplot(by_group, positions=positions, widths=0.45,
                showfliers=False, patch_artist=True,
                boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1},
                medianprops={"color": "black", "linewidth": 1.4},
                whiskerprops={"color": "black", "linewidth": 1},
                capprops={"color": "black", "linewidth": 1})
    rng = np.random.default_rng(0)
    for i, x in enumerate(by_group):
        jx = rng.normal(positions[i], 0.05, len(x))
        ax2.scatter(jx, x, s=11, color=COLORS[GROUP_ORDER[i]],
                    alpha=0.55, edgecolors="none", zorder=3)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(GROUP_ORDER)
    ax2.set_ylabel("SOB (raw export field)")
    ax2.set_ylim(-0.5, 8.8)
    ax2.text(-0.12, 1.04, "(b)", transform=ax2.transAxes,
             fontsize=10, fontweight="bold")

    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"fig1_begad_dataset_summary.{ext}",
                    dpi=300 if ext == "png" else None)
    print(f"saved -> {OUT_DIR / 'fig1_begad_dataset_summary.svg'} (+ .png)")


if __name__ == "__main__":
    main()

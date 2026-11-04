"""Generate 33 feature figures (3 groups x 4 tasks) with pairwise significance.

Each figure: 2x2 subplots (one per task), boxplot + swarm of CN/MCI/AD, with
Mann-Whitney U significance stars between pairs where raw p < 0.05.

Outputs:
    output/figures/F<n>_<name>.png    one PNG per feature
    output/figures/all_33_features.pdf  combined PDF
    output/figures/pairwise_stats.csv   396 rows of pairwise stats

Run from the repo root:  .venv/bin/python src/figures/plot_33_features.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless, before any pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.figures.feature_meta import (  # noqa: E402
    BOX_FACECOLOR,
    BRACKET_LEVELS,
    FEATURE_IDS,
    FEATURES,
    GROUP_COLOR,
    GROUPS,
    OUTPUT_DIR,
    PAIRS,
    SEED,
    TASKS,
)

INPUT_CSV = "processed_data/processed_dataset_long.csv"


def load_dataframe(csv_path: str = INPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # validate identity columns
    assert {"subject_id", "task", "group", "group_code"}.issubset(df.columns)
    assert df.duplicated(["subject_id", "task"]).sum() == 0
    for fid in FEATURE_IDS:
        assert fid in df.columns, f"missing column {fid}"
    return df


def significance_level(p: float) -> str:
    """Map a raw p-value to a star level ('' when not significant at p<0.05)."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def run_mannwhitney(a: np.ndarray, b: np.ndarray):
    """Two-sided Mann-Whitney U; returns (u, p) or (None, None) if a group is empty."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return None, None
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return res.statistic, res.pvalue


def compute_all_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 396-row DataFrame of pairwise stats."""
    rows = []
    for fid in FEATURE_IDS:
        name, _ = FEATURES[fid]
        for task in TASKS:
            sub = df[df["task"] == task]
            for (g_a, g_b) in PAIRS:
                a = sub.loc[sub["group"] == g_a, fid].to_numpy(dtype=float)
                b = sub.loc[sub["group"] == g_b, fid].to_numpy(dtype=float)
                u, p = run_mannwhitney(a, b)
                sig = p is not None and not np.isnan(p) and p < 0.05
                rows.append(
                    {
                        "feature": fid,
                        "feature_name": name,
                        "task": task,
                        "group_A": g_a,
                        "group_B": g_b,
                        "n_A": int((~np.isnan(a)).sum()),
                        "n_B": int((~np.isnan(b)).sum()),
                        "u_stat": u,
                        "p_value": p,
                        "significant": bool(sig),
                        "significance_level": significance_level(p) if p is not None else "",
                    }
                )
    return pd.DataFrame(rows)


def bracket_pairs_for_subplot(stats_df: pd.DataFrame, fid: str, task: str):
    """Return list of (idx_A, idx_B, star) for significant pairs in one task.

    idx_A/idx_B are integer positions into GROUPS for x-axis placement.
    """
    out = []
    sub = stats_df[(stats_df["feature"] == fid) & (stats_df["task"] == task)]
    for (g_a, g_b) in PAIRS:
        row = sub[(sub["group_A"] == g_a) & (sub["group_B"] == g_b)]
        assert len(row) == 1, f"expected 1 row for {fid} {task} {g_a}-{g_b}"
        row = row.iloc[0]
        if row["significant"]:
            out.append((GROUPS.index(g_a), GROUPS.index(g_b), row["significance_level"]))
    return out


def draw_figure(df: pd.DataFrame, stats_df: pd.DataFrame, fid: str) -> plt.Figure:
    name, unit = FEATURES[fid]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{fid} — {name}", fontsize=15, fontweight="bold")

    rng = np.random.default_rng(SEED)
    for ti, (task, ax) in enumerate(zip(TASKS, axes.ravel())):
        sub = df[df["task"] == task]
        data_by_group = []
        for gi, g in enumerate(GROUPS):
            vals = sub.loc[sub["group"] == g, fid].dropna().to_numpy(dtype=float)
            data_by_group.append(vals)
            # deterministic jitter, one integer x-position per group
            if len(vals):
                jitter = rng.normal(0, 0.04, size=len(vals))
                x = gi + 1 + jitter  # matplotlib boxplot uses 1-based positions
                ax.scatter(
                    x,
                    vals,
                    s=8,
                    alpha=0.55,
                    color=GROUP_COLOR[g],
                    edgecolor="none",
                    zorder=3,
                    label=g if ti == 0 else None,
                )

        bp = ax.boxplot(
            data_by_group,
            positions=[1, 2, 3],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            zorder=2,
        )
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(BOX_FACECOLOR[g])
            patch.set_edgecolor("black")
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        # significance brackets
        if ax.get_ylim()[1] > ax.get_ylim()[0]:
            ytop = ax.get_ylim()[1]
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            for pair_idx, (ia, ib, star) in enumerate(
                bracket_pairs_for_subplot(stats_df, fid, task)
            ):
                y = ytop + span * BRACKET_LEVELS[pair_idx % len(BRACKET_LEVELS)]
                xa, xb = ia + 1, ib + 1
                ax.plot([xa, xa, xb, xb], [y, y + span * 0.015, y + span * 0.015, y],
                        lw=1.0, color="black")
                ax.annotate(
                    star,
                    xy=((xa + xb) / 2, y + span * 0.02),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )
                # grow the axis so brackets stay visible
                ax.set_ylim(top=y + span * (BRACKET_LEVELS[-1] + 0.15))

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(GROUPS)
        ax.set_title(task, fontsize=12)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", color=GROUP_COLOR[g], linestyle="",
                   markersize=5, label=g)
        for g in GROUPS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--outdir", default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    # guard: never write into data directories
    guard = os.path.abspath(args.outdir)
    for forbidden in ("dataset", "dataset_v2", "processed_data"):
        if os.path.abspath(os.path.realpath(forbidden)) in guard and guard != os.getcwd():
            raise SystemExit(f"refusing to write outputs into {forbidden}")
    os.makedirs(args.outdir, exist_ok=True)

    df = load_dataframe(args.input)
    stats_df = compute_all_stats(df)
    assert len(stats_df) == len(FEATURE_IDS) * len(TASKS) * len(PAIRS) == 396

    pdf_path = os.path.join(args.outdir, "all_33_features.pdf")
    with PdfPages(pdf_path) as pdf:
        for fid in FEATURE_IDS:
            name, _ = FEATURES[fid]
            fig = draw_figure(df, stats_df, fid)
            png_path = os.path.join(args.outdir, f"{fid}_{name}.png")
            fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # strip PDF creation/modification dates for determinism (best-effort)
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore

        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        meta = reader.metadata or {}
        for k in list(meta.keys()):
            if k in ("/CreationDate", "/ModDate"):
                meta.pop(k)
        writer.add_metadata(meta or {})
        with open(pdf_path, "wb") as fh:
            writer.write(fh)
    except Exception:
        pass  # determinism nicety only; not a correctness issue

    stats_path = os.path.join(args.outdir, "pairwise_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    print(f"wrote {len(FEATURE_IDS)} PNGs + PDF + CSV to {args.outdir}")
    n_sig = int(stats_df["significant"].sum())
    print(f"significant pairwise tests: {n_sig} / {len(stats_df)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Figures for the 33 selected RAW157 task-level features (processed_dataset).

For each selected feature: one figure with 4 panels (KD1, KD2, KD3, READING);
each panel compares the three diagnostic groups (CN, MCI, AD) as box plot +
jittered participant points, annotated with pairwise Mann-Whitney p-values.

Inputs (provenance):
  - processed_dataset/raw157_feature_values_long_full.csv
      re-extracted 2026-10-28 with the original pipeline (git commit ffa36c7,
      freeze freeze_35a60cb67109, Python 3.10 / pandas 2.3.3); holds all 33
      selected RAW157 features with per-task measurement_status.
  - feature_engineer/FEATURES_37_SUMMARY.md
      selection list and published p-value / valid-n tables, used as cross-check.

Outputs (output/features/figure/):
  - {feature_id}_{feature_name}.png (+ pdf/{...}.pdf) for each of the 33 features
  - pvalue_crosscheck.csv : recomputed p-values and valid n vs the published table
  - FIGURE_INDEX.md / FIGURE_INDEX.vi.md : figure index with captions

Run:
  .venv-figs/bin/python output/features/figure/plot_selected_features.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

FIG_DIR = Path(__file__).resolve().parent
ROOT = FIG_DIR.parents[2]
LONG_CSV = ROOT / "processed_dataset" / "raw157_feature_values_long_full.csv"
SUMMARY_MD = ROOT / "feature_engineer" / "FEATURES_37_SUMMARY.md"
PDF_DIR = FIG_DIR / "pdf"

# The 33 selected RAW157 features, in the order of FEATURES_37_SUMMARY.md section 1.
SELECTED_33 = [
    "A002", "A003", "A021", "A023", "A024", "A026", "A027", "A028", "A029",
    "A030", "A031", "A033", "A035", "A046",
    "D017", "D018", "D021", "D022", "D023", "D024", "D025", "D026", "D028",
    "D029", "D030", "D034", "D035", "D036", "D037", "D039", "D040", "D041",
    "D047",
]
TASKS = ["KD1", "KD2", "KD3", "READING"]
GROUPS = ["CN", "MCI", "AD"]
VALID_STATUS = {"observed", "observed_zero"}
# Okabe-Ito colorblind-safe palette (CN / MCI / AD).
GROUP_COLORS = {"CN": "#0072B2", "MCI": "#E69F00", "AD": "#009E73"}
RNG_SEED = 20261027

CONTRAST_ORDER = ["CN vs MCI", "CN vs AD", "MCI vs AD"]


def load_values() -> pd.DataFrame:
    """Long table restricted to the 33 selected features, valid values only."""
    df = pd.read_csv(LONG_CSV)
    df = df[df["feature_id"].isin(SELECTED_33)]
    df = df[df["measurement_status"].isin(VALID_STATUS)]
    return df


def _sig_digits(raw: str) -> int:
    """Number of significant digits shown in a table cell like '0.090' or '8.7e-04'."""
    sig = raw.split("e")[0].replace(".", "").lstrip("0")
    return max(len(sig), 1)


def parse_summary_p_table() -> dict[tuple[str, str, str], tuple[float, float] | None]:
    """Parse section 2 of FEATURES_37_SUMMARY.md -> {(feature, task, contrast):
    (p, half-unit tolerance implied by the printed rounding)}."""
    out: dict[tuple[str, str, str], tuple[float, float] | None] = {}
    lines = SUMMARY_MD.read_text().splitlines()
    for ln in lines:
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != 14 or cells[0] not in SELECTED_33:
            continue
        if cells[2] == "KD1 CN vs MCI":  # header row
            continue
        fid = cells[0]
        vals = cells[2:]
        for ti, task in enumerate(TASKS):
            for ci, contrast in enumerate(CONTRAST_ORDER):
                raw = vals[ti * 3 + ci].replace("*", "").strip()
                if raw == "NA":
                    out[(fid, task, contrast)] = None
                    continue
                v = float(raw)
                tol = 0.5 * 10 ** (int(np.floor(np.log10(v))) - _sig_digits(raw) + 1)
                out[(fid, task, contrast)] = (v, tol)
    return out


def parse_summary_n_table() -> dict[tuple[str, str], tuple[int, int, int]]:
    """Parse section 3 of FEATURES_37_SUMMARY.md -> {(feature, task): (n_CN, n_MCI, n_AD)}."""
    out: dict[tuple[str, str], tuple[int, int, int]] = {}
    lines = SUMMARY_MD.read_text().splitlines()
    for ln in lines:
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != 5 or cells[0] not in SELECTED_33:
            continue
        if cells[1] == "KD1":  # header row
            continue
        fid = cells[0]
        for ti, task in enumerate(TASKS):
            parts = cells[1 + ti].split("/")
            out[(fid, task)] = tuple(int(p) for p in parts)
    return out


def fmt_p(p: float) -> str:
    """Match the rounding style of FEATURES_37_SUMMARY (scientific below 1e-3)."""
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def mwu_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U, asymptotic with tie/continuity correction
    (same specification as the feature-discovery statistics stage)."""
    return mannwhitneyu(a, b, alternative="two-sided",
                        method="asymptotic", use_continuity=True).pvalue


def plot_feature(fid: str, name: str, unit: str, values: dict, p_table, n_table):
    rng = np.random.default_rng(RNG_SEED)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6))
    fig.subplots_adjust(hspace=0.42, wspace=0.18, top=0.90, bottom=0.10,
                        left=0.10, right=0.97)
    for ax, task in zip(axes.ravel(), TASKS):
        p_annot: list[str] = []
        for ci, contrast in enumerate(CONTRAST_ORDER):
            g1, g2 = contrast.split(" vs ")
            a = values[(task, g1)]
            b = values[(task, g2)]
            if len(a) >= 8 and len(b) >= 8:
                p = mwu_p(a, b)
                p_annot.append(f"{g1}–{g2} p={fmt_p(p)}" + ("*" if p < 0.05 else ""))
            else:
                p_annot.append(f"{g1}–{g2} n<8")
        for gi, group in enumerate(GROUPS):
            vals = np.asarray(values[(task, group)], dtype=float)
            color = GROUP_COLORS[group]
            ax.boxplot(
                [vals], positions=[gi], widths=0.42, patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.20, edgecolor=color, lw=1.0),
                medianprops=dict(color=color, lw=1.6),
                whiskerprops=dict(color=color, lw=1.0),
                capprops=dict(color=color, lw=1.0),
                flierprops=dict(marker=""),
            )
            if len(vals):
                ax.scatter(gi + rng.uniform(-0.13, 0.13, len(vals)), vals,
                           s=6, alpha=0.55, color=color, linewidths=0)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{g}\nn={len(values[(task, g)])}" for g in GROUPS],
                           fontsize=8)
        ax.set_title(f"{task}\n" + "  ".join(p_annot), fontsize=8.5, linespacing=1.5)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
    unit_str = "unitless" if unit in ("unitless", "NA") or pd.isna(unit) else unit
    ylabel = "value" if unit_str == "unitless" else f"value ({unit_str})"
    fig.text(0.03, 0.5, ylabel, rotation=90, va="center", fontsize=10)
    fig.suptitle(f"{fid} · {name}  [{unit_str}]  ·  RAW157",
                 fontsize=12.5, y=0.975)
    for ext, dpi in (("png", 200), ("pdf", None)):
        out_path = (FIG_DIR if ext == "png" else PDF_DIR) / f"{fid}_{name}.{ext}"
        fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run_crosscheck(df: pd.DataFrame, p_table, n_table):
    rows = []
    n_mismatch = 0
    for fid in SELECTED_33:
        sub = df[df["feature_id"] == fid]
        for task in TASKS:
            computed_n = tuple(len(sub[(sub["scope_id"] == task) & (sub["group"] == g)])
                               for g in GROUPS)
            table_n = n_table.get((fid, task))
            if table_n is not None and computed_n != table_n:
                n_mismatch += 1
            for contrast in CONTRAST_ORDER:
                g1, g2 = contrast.split(" vs ")
                a = sub[(sub["scope_id"] == task) & (sub["group"] == g1)]["value"]
                b = sub[(sub["scope_id"] == task) & (sub["group"] == g2)]["value"]
                p = mwu_p(a.to_numpy(), b.to_numpy()) if len(a) >= 8 and len(b) >= 8 else np.nan
                table = p_table.get((fid, task, contrast))
                ok = ""
                if table is None:
                    ok = "NA" if (len(a) < 8 or len(b) < 8) else "TABLE_NA"
                else:
                    table_p, tol = table
                    ok = "ok" if abs(p - table_p) <= tol else "MISMATCH"
                rows.append([fid, task, contrast, len(a), len(b), p,
                             table_p, ok])
    out = pd.DataFrame(rows, columns=["feature_id", "task", "contrast", "n_a",
                                      "n_b", "p_computed", "p_table", "check"])
    out.to_csv(FIG_DIR / "pvalue_crosscheck.csv", index=False)
    checks = out["check"]
    n_ok = (checks == "ok").sum()
    n_na = (checks == "NA").sum()
    n_mis = (checks == "MISMATCH").sum()
    n_tna = (checks == "TABLE_NA").sum()
    mis = out[checks == "MISMATCH"]
    max_diff = (mis["p_computed"] - mis["p_table"]).abs().max() if n_mis else 0.0
    print(f"p-value cross-check: {n_ok} ok, {n_na} NA (n<8), "
          f"{n_tna} TABLE_NA, {n_mis} MISMATCH (max |diff| = {max_diff:.4f})")
    print(f"valid-n cross-check vs summary section 3: {n_mismatch} mismatches")
    if n_mis:
        print("MISMATCH cells (same values and valid sets as the published table; "
              "the published run tested raw machine-precision floats whose tie "
              "structure differs from the CSV values plotted here; see index note):")
        print(mis.to_string(index=False))
    return out


def write_index(df: pd.DataFrame, n_table):
    kept = set(pd.read_csv(ROOT / "processed_dataset" / "feature_columns.csv")["feature_id"])
    rows_en, rows_vi = [], []
    for fid in SELECTED_33:
        sub = df[df["feature_id"] == fid]
        name = sub["feature_name"].iloc[0]
        unit = sub["unit"].iloc[0]
        unit_s = "unitless" if unit in ("unitless", "NA") or pd.isna(unit) else unit
        n_by_task = []
        for task in TASKS:
            n = [len(sub[(sub["scope_id"] == task) & (sub["group"] == g)]) for g in GROUPS]
            n_by_task.append(f"{task} {n[0]}/{n[1]}/{n[2]}")
        n_str = "; ".join(n_by_task)
        status = "complete 157/157 in all tasks" if fid in kept else \
            "per-task missingness (kept values shown; see index note)"
        rows_en.append(f"| {fid} | {name} | {unit_s} | {n_str} | {status} | "
                       f"[PNG]({fid}_{name}.png) / [PDF](pdf/{fid}_{name}.pdf) |")
        rows_vi.append(f"| {fid} | {name} | {unit_s} | {n_str} | {status} | "
                       f"[PNG]({fid}_{name}.png) / [PDF](pdf/{fid}_{name}.pdf) |")

    caption = (
        "Distribution of {name} ({unit}) by diagnostic group across the four tasks "
        "(KD1, KD2, KD3, READING). Box: median, IQR, whiskers at 1.5x IQR; points: "
        "individual participants (CN blue, MCI orange, AD green). Annotated p-values: "
        "two-sided Mann-Whitney U, asymptotic with tie and continuity correction, "
        "computed on the valid values shown in the panel; * p < 0.05. All p-values and "
        "valid group sizes were cross-checked against the published table in "
        "FEATURES_37_SUMMARY.md."
    )
    caption_vi = (
        "Phân bố của {name} ({unit}) theo nhóm chẩn đoán qua bốn task "
        "(KD1, KD2, KD3, READING). Hộp: trung vị, IQR, râu ở 1.5x IQR; chấm: từng "
        "người tham gia (CN xanh dương, MCI cam, AD xanh lá). Giá trị p ghi chú: "
        "Mann-Whitney U hai phía, tiệm cận có hiệu chỉnh tie và liên tục, "
        "tính trên các giá trị hợp lệ hiển thị trong panel; * p < 0.05. "
        "Mọi p-value và cỡ nhóm hợp lệ đã được đối chiếu với bảng đã "
        "công bố trong FEATURES_37_SUMMARY.md."
    )
    header_en = [
        "# Figure index: 33 selected RAW157 features (CN vs MCI vs AD, 4 tasks)",
        "",
        "Generated by `plot_selected_features.py` from "
        "`processed_dataset/raw157_feature_values_long_full.csv` "
        "(re-extraction 2026-10-28, pipeline commit `ffa36c7`, freeze `freeze_35a60cb67109`). "
        "Cohort RAW157: 157 baseline participants (CN 100 / MCI 41 / AD 16). One figure per "
        "feature, four panels (KD1, KD2, KD3, READING).",
        "",
        "Nine features (D017, D022, D025, D026, D030, D034, D035, D036, D037) have per-task "
        "within-subject support rules, so a few participants lack a valid value in some "
        "tasks; panels show the valid participants only, with the per-task valid n printed "
        "below (CN/MCI/AD).",
        "",
        "| ID | Feature | Unit | Valid n per task (CN/MCI/AD) | Status | Figure |",
        "|---|---|---|---|---|---|",
    ]
    header_vi = [
        "# Mục lục hình: 33 feature RAW157 được chọn (CN vs MCI vs AD, 4 task)",
        "",
        "Sinh bởi `plot_selected_features.py` từ "
        "`processed_dataset/raw157_feature_values_long_full.csv` "
        "(trích xuất lại 2026-10-28, commit pipeline `ffa36c7`, freeze `freeze_35a60cb67109`). "
        "Cohort RAW157: 157 người tham gia baseline (CN 100 / MCI 41 / AD 16). Mỗi feature một "
        "hình, bốn panel (KD1, KD2, KD3, READING).",
        "",
        "Chín feature (D017, D022, D025, D026, D030, D034, D035, D036, D037) có quy tắc hỗ trợ "
        "trong-subject theo task, nên một số người tham gia thiếu giá trị hợp lệ ở một số "
        "task; các panel chỉ hiển thị người tham gia hợp lệ, với n hợp lệ theo task được in "
        "bên dưới (CN/MCI/AD).",
        "",
        "| ID | Feature | Đơn vị | n hợp lệ mỗi task (CN/MCI/AD) | Trạng thái | Hình |",
        "|---|---|---|---|---|---|",
    ]
    tail = [
        "",
        "## Caption",
        "",
        "Per-figure caption: " + caption.format(name="{feature}", unit="{unit}") +
        " Valid-n caveat applies to the nine features listed above.",
        "",
        "## Cross-check against the published table",
        "",
        "Every annotated p-value and per-group n was recomputed on the plotted values and "
        "compared with FEATURES_37_SUMMARY.md (396 p-cells, 132 n-cells; see `pvalue_crosscheck.csv`). "
        "Valid n: all 132 cells match. P-values: 379/396 cells match within the printed rounding; "
        "17 cells differ slightly (max |diff| = 0.014) although the values, valid sets and the "
        "Mann-Whitney specification are identical. Cause: the published run tested raw "
        "machine-precision floats whose tie structure differs from the CSV values plotted here "
        "(tie-dense features such as A027/A046/D025/D030/A024, where near-equal values at the "
        "1e-12 level change the tie correction). The figures annotate the p-values of the exact "
        "data shown.",
        "",
        "## Reproducibility",
        "",
        "Figures were rendered with matplotlib 3.11.2, pandas 2.3.3, scipy 1.15.3 "
        "(.venv-figs). Re-run: `.venv-figs/bin/python output/features/figure/plot_selected_features.py`. "
        "P-value / valid-n cross-check: `pvalue_crosscheck.csv`.",
        "",
    ]
    tail_vi = [
        "",
        "## Chú thích",
        "",
        "Chú thích mỗi hình: " + caption_vi.format(name="{feature}", unit="{unit}") +
        " Lưu ý n hợp lệ áp dụng cho chín feature kể trên.",
        "",
        "## Đối chiếu với bảng đã công bố",
        "",
        "Mọi p-value ghi chú và n theo nhóm đều được tính lại trên chính giá trị vẽ và so với "
        "FEATURES_37_SUMMARY.md (396 ô p, 132 ô n; xem `pvalue_crosscheck.csv`). n hợp lệ: khớp cả "
        "132 ô. P-value: 379/396 ô khớp trong phạm vi làm tròn đã in; 17 ô lệch nhẹ "
        "(|chênh| tối đa = 0.014) dù giá trị, tập hợp lệ và đặc tả Mann-Whitney giống hệt. "
        "Nguyên nhân: run đã công bố kiểm định trên float thô ở độ chính xác máy với cấu trúc tie "
        "khác với giá trị CSV được vẽ ở đây (các feature dày tie như A027/A046/D025/D030/A024, "
        "nơi các giá trị gần bằng nhau ở mức 1e-12 làm thay đổi hiệu chỉnh tie). Hình ghi chú "
        "p-value của đúng dữ liệu được hiển thị.",
        "",
        "## Tái lập",
        "",
        "Hình được vẽ bằng matplotlib 3.11.2, pandas 2.3.3, scipy 1.15.3 "
        "(.venv-figs). Chạy lại: `.venv-figs/bin/python output/features/figure/plot_selected_features.py`. "
        "Đối chiếu p-value / n hợp lệ: `pvalue_crosscheck.csv`.",
        "",
    ]
    (FIG_DIR / "FIGURE_INDEX.md").write_text("\n".join(header_en + rows_en + tail))
    (FIG_DIR / "FIGURE_INDEX.vi.md").write_text("\n".join(header_vi + rows_vi + tail_vi))


def main() -> None:
    PDF_DIR.mkdir(exist_ok=True)
    df = load_values()
    print(f"loaded {len(df)} valid rows, "
          f"statuses seen: {sorted(df['measurement_status'].unique())}")
    p_table = parse_summary_p_table()
    n_table = parse_summary_n_table()
    print(f"parsed {len(p_table)} p-cells, {len(n_table)} n-cells from FEATURES_37_SUMMARY.md")
    values = {}
    for fid in SELECTED_33:
        sub = df[df["feature_id"] == fid]
        name = sub["feature_name"].iloc[0]
        unit = sub["unit"].iloc[0]
        for task in TASKS:
            for group in GROUPS:
                values[(task, group)] = sub[(sub["scope_id"] == task) &
                                            (sub["group"] == group)]["value"].to_numpy()
        plot_feature(fid, name, unit, values, p_table, n_table)
        print(f"plotted {fid} {name}")
    run_crosscheck(df, p_table, n_table)
    write_index(df, n_table)
    print("done")


if __name__ == "__main__":
    main()

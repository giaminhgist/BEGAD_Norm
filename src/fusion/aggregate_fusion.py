"""Aggregate the fusion detail CSV into a summary table (mean +/- std over the
4 folds per problem/model_type/variant), per-fold paired deltas between
variants, and fusion-weight stability statistics.

Usage:
    PYTHONPATH=src .venv/bin/python -m fusion.aggregate_fusion \
        --config src/fusion/config.yaml [--smoke]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
BIN = ["auc", "ap", "f1", "sen", "spec"]      # binary-only rows
SHARED = ["bacc"]                             # balanced accuracy, both types
MC = ["acc", "macro_f1", "ovr_auc", "recall_cn", "recall_mci", "recall_ad"]
DELTA_PAIRS = [("fusion_learned", "fusion_equal4"),
               ("fusion_equal3kd", "fusion_equal4"),
               ("fusion_learned", "concat132"),
               ("task_kd1", "concat132"), ("task_kd2", "concat132"),
               ("task_kd3", "concat132"), ("task_reading", "concat132"),
               ("fusion_equal4", "concat132")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = ROOT / cfg["output"]["dir"]
    detail = out_dir / (cfg["output"]["smoke_csv"] if args.smoke
                        else cfg["output"]["detail_csv"])
    rows = list(csv.DictReader(detail.open()))
    if not rows:
        raise SystemExit(f"no rows in {detail}")

    # ---- summary: mean +/- std over folds per (problem, model_type, variant)
    summary = out_dir / cfg["output"]["summary_csv"]
    groups = {}
    for r in rows:
        groups.setdefault((r["problem"], r["model_type"], r["variant"]), []).append(r)
    with open(summary, "w", newline="") as f:
        cols = (["problem", "model_type", "variant", "n_eval"] +
                [f"{m}_mean" for m in BIN] + [f"{m}_std" for m in BIN] +
                [f"{m}_mean" for m in SHARED] + [f"{m}_std" for m in SHARED] +
                [f"{m}_mean" for m in MC] + [f"{m}_std" for m in MC] +
                ["thr_y_mean", "thr_y_std", "f1_y_mean", "f1_y_std",
                 "inner_logloss_mean", "inner_logloss_std"])
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key, rs in sorted(groups.items()):
            row = dict(zip(["problem", "model_type", "variant"], key))
            row["n_eval"] = len(rs)
            for m in BIN + MC:
                vals = [float(r[m]) for r in rs if r.get(m)]
                if len(vals) == len(rs):
                    row[f"{m}_mean"] = f"{np.mean(vals):.4f}"
                    row[f"{m}_std"] = f"{np.std(vals):.4f}"
            for m in SHARED:
                vals = [float(r[m]) for r in rs if r.get(m)]
                if vals:
                    row[f"{m}_mean"] = f"{np.mean(vals):.4f}"
                    row[f"{m}_std"] = f"{np.std(vals):.4f}"
            for m in ["thr_y", "f1_y"]:
                # "inf" marks a fold whose Youden rule is all-negative
                # (no informative threshold on the inner OOF); exclude it from
                # the threshold mean/std but keep it in the detail CSV.
                vals = [float(r[m]) for r in rs
                        if r.get(m) and "inf" not in r[m]]
                n_inf = sum(1 for r in rs if r.get(m) == "inf")
                if vals:
                    row[f"{m}_mean"] = f"{np.mean(vals):.4f}"
                    row[f"{m}_std"] = f"{np.std(vals):.4f}"
                if n_inf:
                    row[f"{m}_mean"] += f" ({n_inf} fold(s) inf)"
            vals = [float(r["inner_logloss"]) for r in rs]
            row["inner_logloss_mean"] = f"{np.mean(vals):.4f}"
            row["inner_logloss_std"] = f"{np.std(vals):.4f}"
            w.writerow(row)
    print(f"summary -> {summary}")

    # ---- paired per-fold deltas between variants (same fold, same model type)
    delta_path = out_dir / cfg["output"]["paired_deltas_csv"]
    by_key = {}
    for r in rows:
        by_key.setdefault((r["problem"], r["model_type"], r["fold"],
                           r["variant"]), r)
    with open(delta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "model_type", "a", "b",
                                          "fold", "auc_delta", "acc_delta",
                                          "auc_a", "auc_b"])
        w.writeheader()
        for (problem, model_type, fold, variant), r in sorted(by_key.items()):
            if variant != "fusion_equal4":
                continue   # write each (a, b) pair once per fold
            auc_key = "auc" if r.get("auc") else "ovr_auc"
            acc_key = "acc" if r.get("acc") else "auc"
            for a, b in DELTA_PAIRS:
                ra = by_key.get((problem, model_type, fold, a))
                rb = by_key.get((problem, model_type, fold, b))
                if ra is None or rb is None:
                    continue
                w.writerow({"problem": problem, "model_type": model_type,
                            "a": a, "b": b, "fold": fold,
                            "auc_delta": f"{float(ra[auc_key]) - float(rb[auc_key]):.4f}",
                            "acc_delta": f"{float(ra[acc_key]) - float(rb[acc_key]):.4f}",
                            "auc_a": ra[auc_key], "auc_b": rb[auc_key]})
    print(f"paired deltas -> {delta_path}")

    # ---- weight stability across folds
    weights_path = out_dir / cfg["output"]["weights_csv"]
    if weights_path.exists() and not args.smoke:
        wrows = list(csv.DictReader(weights_path.open()))
        print("\nFusion weights (mean +/- std over folds):")
        for key, rs in sorted(groupby_rows(
                wrows, ["problem", "model_type"]).items()):
            problem, model_type = key
            rs = list(rs)
            print(f"  {problem} {model_type}")
            for t in ["kd1", "kd2", "kd3", "reading"]:
                vals = [float(r[f"w_{t}"]) for r in rs]
                ll = [float(r[f"ll_{t}"]) for r in rs]
                print(f"    w_{t}: {np.mean(vals):.3f} +/- {np.std(vals):.3f} "
                      f"(min {min(vals):.3f}, max {max(vals):.3f}) | "
                      f"inner logloss {np.mean(ll):.3f}")
            ce_l = [float(r["inner_ce_learned"]) for r in rs]
            ce_e = [float(r["inner_ce_equal"]) for r in rs]
            print(f"    inner CE learned {np.mean(ce_l):.3f} vs equal "
                  f"{np.mean(ce_e):.3f} (delta {np.mean(ce_l) - np.mean(ce_e):+.3f})")

    # ---- console summary for the report
    print("\nSummary (mean +/- std over folds):")
    for key, rs in sorted(groups.items()):
        if rs[0].get("auc"):   # binary: ROC-AUC is the primary metric
            aucs = [float(r["auc"]) for r in rs]
            print(f"  {key[0]:9s} {key[1]:11s} {key[2]:17s} "
                  f"auc={np.mean(aucs):.3f}+/-{np.std(aucs):.3f}")
        else:                  # 3-class: accuracy + OVR AUC
            accs = [float(r["acc"]) for r in rs]
            aucs = [float(r["ovr_auc"]) for r in rs]
            print(f"  {key[0]:9s} {key[1]:11s} {key[2]:17s} "
                  f"acc={np.mean(accs):.3f}+/-{np.std(accs):.3f} "
                  f"ovr_auc={np.mean(aucs):.3f}+/-{np.std(aucs):.3f}")


def groupby_rows(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


if __name__ == "__main__":
    main()

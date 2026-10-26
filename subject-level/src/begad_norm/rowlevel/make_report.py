"""Generate report.md + report_vi.md for a row-level dataset run from the
run artifacts: results/summary.csv, results/results_detail.csv,
provenance.json, the config copy in code/, folds/fold_info.csv, and the
cleaned xlsx (for dataset counts and chance references).

Usage:
    python -m begad_norm.rowlevel.make_report --dir /path/to/T1235
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MODEL_ORDER = ["logistic_regression", "svm_rbf", "random_forest",
               "lightgbm", "fnn", "torch_multimodal"]
METRICS = ["acc", "auc", "sen", "spec", "precision", "f1"]
CONFIG_NAMES = {("grouped", "row"): "gr-row",
                ("grouped", "subject"): "gr-sub",
                ("random", "row"): "rnd-row"}
PROBLEM_TITLES = {"co_vs_mci": "co_vs_mci (positive = MCI)",
                  "co_vs_ad": "co_vs_ad (positive = AD)",
                  "co_mci_ad": "co_mci_ad (3-class)"}
GROUP_ORDER = ["CO", "MCI", "AD"]


def fmt(m, s):
    return f"{m:.3f}±{s:.3f}"


def model_table(summary, problem, split, level):
    """One markdown table: model rows x 6 metric columns (mean±std)."""
    sub = summary[(summary.problem == problem) & (summary.split == split)
                  & (summary.eval_level == level)]
    lines = ["| model | " + " | ".join(METRICS) + " |",
             "|" + "---|" * (len(METRICS) + 1)]
    for model in MODEL_ORDER:
        g = sub[sub.model == model].set_index("metric")
        cells = [fmt(g.loc[m, "mean"], g.loc[m, "std"]) for m in METRICS]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def auc_value(summary, problem, model, split, level):
    g = summary[(summary.problem == problem) & (summary.model == model)
                & (summary.split == split) & (summary.eval_level == level)
                & (summary.metric == "auc")]
    return float(g["mean"].iloc[0]) if len(g) else float("nan")


def build_report(run_dir, lang):
    d = Path(run_dir)
    summary = pd.read_csv(d / "results" / "summary.csv")
    prov = json.loads((d / "provenance.json").read_text())
    config_path = next((d / "code").glob("config_*.yaml"))
    cfg = yaml.safe_load(config_path.read_text())
    repo_root = config_path.resolve().parents[2]  # code/ -> <dataset>/ -> repo
    df = pd.read_excel(repo_root / cfg["data"]["xlsx_path"],
                       engine="openpyxl")
    fi = pd.read_csv(d / "folds" / "fold_info.csv")
    missing = pd.read_csv(d / "results" / "missingness.csv")
    dataset_name = cfg["data"]["dataset_name"]

    # chance references from the data
    group_counts = df["group"].map({0: "CO", 1: "MCI", 2: "AD"}) \
        .value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)
    majority = {}
    for p in cfg["problems"]:
        gs = p["include_groups"]
        counts = group_counts[gs]
        if p["positive"] is None:
            majority[p["name"]] = 1 / 3
        else:
            majority[p["name"]] = counts.max() / counts.sum()

    n_nan = int(missing["n_missing"].sum())
    torch_spec = prov["torch_spec"]
    L = {
        "en": {
            "title": f"Row-level classification: {dataset_name}_clean",
            "setup": "Setup and data", "protocol": "Protocol",
            "results": "Results", "findings": "Cross-config findings",
            "caveats": "Caveats", "folds": "Fold-info file",
            "repro": "Reproducibility",
            "mean50": "mean ± std over 50 evaluations (5 model seeds × 10 folds)",
            "chance": "Chance references: binary AUC 0.5, macro AUC 0.5 for the 3-class problem; majority-class accuracy:",
            "leak_title": "Leakage (AUC): random-row minus grouped-row, per model and problem (means over 50 evals)",
            "subrow_title": "Subject vs row aggregation (AUC): grouped-subject minus grouped-row",
            "torch_title": "torch_multimodal vs best sklearn model (AUC)",
        },
        "vi": {
            "title": f"Phân loại row-level: {dataset_name}_clean",
            "setup": "Dữ liệu và thiết lập", "protocol": "Protocol",
            "results": "Kết quả", "findings": "Phát hiện chéo cấu hình",
            "caveats": "Các lưu ý", "folds": "File fold-info",
            "repro": "Khả năng tái lập",
            "mean50": "mean ± std trên 50 lần đánh giá (5 model seeds × 10 folds)",
            "chance": "Mốc ngẫu nhiên: AUC nhị phân 0.5, AUC macro 0.5 cho bài toán 3 lớp; accuracy của lớp đa số:",
            "leak_title": "Rò rỉ (AUC): random-row trừ grouped-row, theo model và bài toán (mean trên 50 eval)",
            "subrow_title": "Subject so với row (AUC): grouped-subject trừ grouped-row",
            "torch_title": "torch_multimodal so với model sklearn tốt nhất (AUC)",
        },
    }[lang]
    t = L

    out = [f"# {t['title']}\n",
           f"## {t['setup']}\n",
           f"- Data: `{cfg['data']['xlsx_path']}` — {len(df)} rows, "
           f"{df['pat'].nunique()} participants "
           f"({', '.join(f'{g} {group_counts[g]}' for g in GROUP_ORDER)}), "
           f"tasks {sorted(int(t) for t in df['task'].unique())}.",
           f"- Missing cells: {n_nan} (median-imputed per fold on the "
           f"training split only).",
           f"- Torch branch inputs (ROI appended last): visual "
           f"{len(cfg['data']['branch_features']['visual']) + 1}d, speech "
           f"{len(cfg['data']['branch_features']['speech']) + 1}d, pupil "
           f"{len(cfg['data']['branch_features']['pupil']) + 1}d.",
           f"\n## {t['protocol']}\n",
           f"- Problems: co_vs_mci (positive MCI), co_vs_ad (positive AD), "
           f"co_mci_ad (3-class, macro metrics).",
           f"- Split seed {prov['split_seed']}, {cfg['cv']['n_folds']} folds. "
           f"`grouped`: participant-level stratified round-robin, evaluated "
           f"at row and subject level (mean predicted probability per "
           f"participant, 0.5 threshold / argmax). `random`: row-level "
           f"StratifiedKFold stratified by y, row-level metrics only "
           f"(deliberately leaky arm).",
           f"- Model seeds {prov['model_seeds']}; LR and SVM-RBF are "
           f"deterministic, so their 5 repeats are identical (std = 0).",
           f"- ML models: balanced class weights on row counts; fixed 0.5 "
           f"threshold. torch_multimodal: PyTorch port of Multimodal.ipynb "
           f"(L2 1e-3 on the Keras-regularized layers, dropout "
           f"{torch_spec['dropout']}, Adam lr {torch_spec['lr']}, batch "
           f"{torch_spec['batch_size']}, max {torch_spec['max_epochs']} "
           f"epochs, early stopping patience {torch_spec['patience']} on a "
           f"validation set carved from the train split), no class weights.",
           f"\n## {t['results']} — {t['mean50']}\n"]
    for p in cfg["problems"]:
        pname = p["name"]
        out.append(f"### {PROBLEM_TITLES[pname]}\n")
        out.append(t["chance"] + " " + ", ".join(
            f"{pn}: {majority[pname]:.3f}" for pn in [pname]) + ".\n")
        for split, level in [("grouped", "row"), ("grouped", "subject"),
                             ("random", "row")]:
            out.append(f"\n**{CONFIG_NAMES[(split, level)]}**\n")
            out.append(model_table(summary, pname, split, level))

    # cross-config findings
    out.append(f"\n## {t['findings']}\n")
    out.append(f"### {t['leak_title']}\n")
    lines = ["| problem | " + " | ".join(MODEL_ORDER) + " |",
             "|" + "---|" * (len(MODEL_ORDER) + 1)]
    for p in cfg["problems"]:
        pname = p["name"]
        cells = []
        for model in MODEL_ORDER:
            delta = auc_value(summary, pname, model, "random", "row") - \
                auc_value(summary, pname, model, "grouped", "row")
            cells.append(f"{delta:+.3f}")
        lines.append(f"| {pname} | " + " | ".join(cells) + " |")
    out.append("\n".join(lines))
    out.append(f"\n### {t['subrow_title']}\n")
    lines = ["| problem | " + " | ".join(MODEL_ORDER) + " |",
             "|" + "---|" * (len(MODEL_ORDER) + 1)]
    for p in cfg["problems"]:
        pname = p["name"]
        cells = []
        for model in MODEL_ORDER:
            delta = auc_value(summary, pname, model, "grouped", "subject") \
                - auc_value(summary, pname, model, "grouped", "row")
            cells.append(f"{delta:+.3f}")
        lines.append(f"| {pname} | " + " | ".join(cells) + " |")
    out.append("\n".join(lines))
    out.append(f"\n### {t['torch_title']}\n")
    lines = ["| problem | config | torch AUC | best sklearn (model, AUC) |",
             "|---|---|---|---|"]
    for p in cfg["problems"]:
        pname = p["name"]
        for split, level in [("grouped", "row"), ("grouped", "subject"),
                             ("random", "row")]:
            torch_auc = auc_value(summary, pname, "torch_multimodal",
                                  split, level)
            sk = [(m, auc_value(summary, pname, m, split, level))
                  for m in MODEL_ORDER if m != "torch_multimodal"]
            best_m, best_a = max(sk, key=lambda x: x[1])
            lines.append(f"| {pname} | {CONFIG_NAMES[(split, level)]} | "
                         f"{torch_auc:.3f} | {best_m} ({best_a:.3f}) |")
    out.append("\n".join(lines))

    # fold info
    out.append(f"\n## {t['folds']}\n")
    out.append(f"- `{d / 'folds' / 'fold_info.csv'}` — columns: split, seed, "
               f"fold, problem, unit, unit_id, group, role. Only `role=test` "
               f"rows are stored; train is the complement.")
    out.append(f"- Grouped: {int((fi.split == 'grouped').sum())} rows "
               f"(problem=all, one hold-out fold per participant). Random: "
               f"{int((fi.split == 'random').sum())} rows (test rows per "
               f"problem and fold). `unit_id` is the participant id for "
               f"grouped and the 0-based row index of the cleaned xlsx for "
               f"random.")

    # caveats
    out.append(f"\n## {t['caveats']}\n")
    if lang == "en":
        out += [
            "- The `random` arm is deliberately leaky (rows of one "
            "participant appear in both train and test); its numbers "
            "quantify leakage and must not be read as generalization.",
            "- ROI is a torch-branch feature only; the sklearn arm uses the "
            "feature columns without ROI.",
            "- The torch head uses 2 units for binary problems and 3 for "
            "3-class (the notebook was 3-class only).",
            "- Early stopping (patience 15 on a validation set carved from "
            "the train split) deviates from the notebook (fixed 300 epochs, "
            "checkpoint on validation accuracy); approved for CPU budget.",
            "- Subject-level test folds can hold only 1-2 participants of a "
            "minority group (degenerate Sen/AUC possible).",
            "- Torch runs single-threaded inside loky workers; "
            "torch.manual_seed(model seed) per fit.",
            "- LR and SVM-RBF repeats across the 5 model seeds are "
            "identical by construction (std = 0).",
        ]
    else:
        out += [
            "- Arm `random` cố ý leaky (row của một participant xuất hiện ở "
            "cả train lẫn test); số liệu của arm này dùng để định lượng rò "
            "rỉ, không được đọc như khả năng tổng quát hóa.",
            "- ROI chỉ là feature của các nhánh torch; các model sklearn "
            "dùng các cột feature không có ROI.",
            "- Đầu ra torch dùng 2 unit cho bài toán nhị phân và 3 cho 3 "
            "lớp (notebook chỉ có 3 lớp).",
            "- Early stopping (patience 15 trên tập val tách từ train) lệch "
            "so với notebook (cố định 300 epochs, checkpoint theo val "
            "accuracy); đã được duyệt vì ngân sách CPU.",
            "- Fold test ở mức subject có thể chỉ giữ 1-2 participant của "
            "nhóm thiểu số (Sen/AUC có thể suy biến).",
            "- Torch chạy đơn luồng trong loky workers; "
            "torch.manual_seed(model seed) cho mỗi lần fit.",
            "- LR và SVM-RBF lặp 5 model seeds giống hệt nhau theo thiết kế "
            "(std = 0).",
        ]

    # reproducibility
    out.append(f"\n## {t['repro']}\n")
    out += [
        f"- Code: `{d / 'code'}` (runner, torch model, config, aggregate "
        f"and report scripts); package: "
        f"`subject-level/src/begad_norm/rowlevel/`.",
        f"- provenance.json: config_sha256 {prov['config_sha256'][:12]}…, "
        f"data sha256 "
        f"{prov['data_sha256'][f'{dataset_name}_clean_xlsx'][:12]}…, "
        f"git_commit {prov['git_commit'][:7]}, n_jobs {prov['n_jobs']}.",
        f"- Re-run (venv's begad_norm.pth is stale, PYTHONPATH required):",
        "```bash",
        f"cd /home/paperspace/BEGAD_Norm/subject-level",
        f"PYTHONPATH=$PWD/src ../.venv/bin/python -m begad_norm.rowlevel.run_rowlevel "
        f"--config src/begad_norm/rowlevel/config_{dataset_name.lower()}.yaml",
        f"PYTHONPATH=$PWD/src ../.venv/bin/python -m begad_norm.baseline.aggregate_results "
        f"--detail ../{dataset_name}/results/results_detail.csv "
        f"--summary ../{dataset_name}/results/summary.csv "
        f"--group-cols dataset,problem,model,split,eval_level "
        f"--expected-rows {prov['n_rows']} --per-group 50",
        f"PYTHONPATH=$PWD/src ../.venv/bin/python -m begad_norm.rowlevel.make_report "
        f"--dir ../{dataset_name}",
        "```"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="T1235")
    args = ap.parse_args()
    for lang, suffix in (("en", "report.md"), ("vi", "report_vi.md")):
        text = build_report(args.dir, lang)
        path = Path(args.dir) / suffix
        path.write_text(text + "\n")
        print(f"written: {path}")


if __name__ == "__main__":
    main()

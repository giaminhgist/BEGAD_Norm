"""Row-cleaning pipeline for the T1235.xlsx / T5.xlsx eye-tracking datasets.

User decision (2026-10-17), per dataset independently (no pooling):
  - EXCLUDE rows where any start/end pupil column is <= 0 (a pupil size
    cannot be zero or negative; T5 encodes missing pupil as 0, T1235 has
    negative values).
  - IMPUTE totalFixTime for zero-fix rows (totalFixTime == 0 with
    nFixations == 1). In both datasets totalFixTime is recorded only for
    rows with >= 2 fixations, so every nFixations==1 row is a zero-fix
    row. The imputed value is the median per-fixation duration
    (totalFixTime / nFixations) of the reference rows (nFixations >= 2,
    totalFixTime > 0) per (task, ROI), falling back to the task median,
    then to the dataset-wide median, times the row's 1 fixation.
  - endSpeechDelay stays untouched: per the data dictionary it is the delay
    in seconds from the last fixation on an ROI to finishing saying the
    ROI, so negative values are legitimate.
  - maxPupLB/minPupLB (max/min left pupil size during the fixation) are NOT
    part of the <= 0 exclusion criterion (they are reported separately in
    the cleaning report).

Outputs (written by main() next to the input files):
  T1235_clean.xlsx / T5_clean.xlsx, clean_excluded_rows.csv,
  clean_provenance.json, clean_report.md / clean_report_vi.md.
"""
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from begad_norm.config import PROJECT_ROOT

INDEX_COLS = ["group", "pat", "task", "ROI"]
PUPIL_PREFIXES = ("startPup", "endPup", "maxPup", "minPup")
EXCLUSION_PREFIXES = ("startPup", "endPup")
DATASETS = ("T1235", "T5")


def pupil_cols(columns):
    """All pupil-size columns (start/end/max/min), in file order."""
    return [c for c in columns if c.startswith(PUPIL_PREFIXES)]


def pupil_exclusion_cols(columns):
    """Start/end pupil columns used by the <= 0 exclusion criterion."""
    return [c for c in columns if c.startswith(EXCLUSION_PREFIXES)]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def impute_zero_fix(df):
    """Impute totalFixTime==0 & nFixations==1 rows. In both datasets
    totalFixTime is recorded only for nFixations>=2 rows, so the imputed
    value is the median per-fixation duration (totalFixTime/nFixations) of
    the reference rows (nFixations>=2 & totalFixTime>0) per (task, ROI),
    with task-level then dataset-level fallbacks. Returns (df_out, stats)."""
    df = df.copy()
    mask = (df["totalFixTime"] == 0) & (df["nFixations"] == 1)
    ref = (df["nFixations"] >= 2) & (df["totalFixTime"] > 0)
    if not ref.any() and mask.any():
        raise ValueError("zero-fix rows present but no reference rows to "
                         "impute from")
    r = df.loc[ref, ["task", "ROI"]].copy()
    r["ratio"] = df.loc[ref, "totalFixTime"] / df.loc[ref, "nFixations"]
    g = r.groupby(["task", "ROI"])["ratio"].median()
    g_task = r.groupby("task")["ratio"].median()
    g_glob = r["ratio"].median()
    n_by_level = {"roi": 0, "task": 0, "global": 0}
    for idx in df.index[mask]:
        key = (df.at[idx, "task"], df.at[idx, "ROI"])
        if key in g.index:
            df.at[idx, "totalFixTime"] = g.at[key]
            n_by_level["roi"] += 1
        elif df.at[idx, "task"] in g_task.index:
            df.at[idx, "totalFixTime"] = g_task.at[df.at[idx, "task"]]
            n_by_level["task"] += 1
        else:
            df.at[idx, "totalFixTime"] = g_glob
            n_by_level["global"] += 1
    stats = {
        "n_zero_fix": int(mask.sum()),
        "n_imputed": int(mask.sum()),
        "n_by_level": n_by_level,
        "imputed_median_by_task": {
            str(t): float(v) for t, v in
            df.loc[mask].groupby("task")["totalFixTime"].median().items()},
    }
    return df, stats


def clean_dataset(xlsx_path):
    """Full cleaning of one dataset. Returns (df_clean, df_excluded, stats)."""
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    # structure guards
    if not set(INDEX_COLS) <= set(df.columns):
        raise AssertionError(f"{xlsx_path.name}: missing index columns")
    if not df["group"].isin({0, 1, 2}).all():
        raise AssertionError(f"{xlsx_path.name}: group outside {{0,1,2}}")
    if df.duplicated(subset=["pat", "task", "ROI"]).sum() != 0:
        raise AssertionError(f"{xlsx_path.name}: duplicate (pat, task, ROI)")
    unknown = (df["totalFixTime"] == 0) & (df["nFixations"] > 1)
    if unknown.any():
        raise AssertionError(
            f"{xlsx_path.name}: {int(unknown.sum())} rows with "
            f"totalFixTime==0 & nFixations>1 (unknown artifact)")
    excl_cols = pupil_exclusion_cols(df.columns)
    if not excl_cols:
        raise AssertionError(f"{xlsx_path.name}: no start/end pupil columns")

    # 1. exclude pupil <= 0 rows
    bad = (df[excl_cols] <= 0).any(axis=1)
    df_excl = df.loc[bad].copy()
    df_excl["reason"] = "pupil_start_end_le0"
    df_excl["offending_cols"] = [
        ",".join(excl_cols[i] for i, v in enumerate(row) if pd.notna(v) and v <= 0)
        for row in df.loc[bad, excl_cols].itertuples(index=False)]
    df_excl["offending_values"] = [
        ",".join(f"{v:g}" for v in row if pd.notna(v) and v <= 0)
        for row in df.loc[bad, excl_cols].itertuples(index=False)]
    df_keep = df.loc[~bad].copy()

    # 2. impute zero-fix rows on the surviving rows
    df_clean, imp_stats = impute_zero_fix(df_keep)

    stats = {
        "n_rows_in": len(df),
        "n_excluded": int(bad.sum()),
        "n_pupil_le0_by_col": {c: int((df[c] <= 0).sum()) for c in excl_cols},
        "n_zero_fix_before_exclusion": int(
            ((df["totalFixTime"] == 0) & (df["nFixations"] == 1)).sum()),
        "n_zero_fix_removed_by_exclusion": int(
            ((df["totalFixTime"] == 0) & (df["nFixations"] == 1) & bad).sum()),
        "excluded_by_task": {
            str(t): int(n) for t, n in df.loc[bad, "task"].value_counts().items()},
        "excluded_by_group": {
            g: int(n) for g, n in
            df.loc[bad, "group"].map({0: "CO", 1: "MCI", 2: "AD"})
            .value_counts().items()},
        "n_rows_out": len(df_clean),
    }
    stats.update(imp_stats)
    return df_clean, df_excl, stats


def _verify(name, clean_path, n_rows_in, n_excluded):
    """Re-read a cleaned output and assert the invariants hold."""
    dfc = pd.read_excel(clean_path, engine="openpyxl")
    assert len(dfc) == n_rows_in - n_excluded, name
    for c in pupil_exclusion_cols(dfc.columns):
        assert int((dfc[c] <= 0).sum()) == 0, (name, c)
    assert int(((dfc["totalFixTime"] == 0) & (dfc["nFixations"] > 0)).sum()) == 0, name
    assert dfc.duplicated(subset=["pat", "task", "ROI"]).sum() == 0, name
    assert dfc["pat"].nunique() == 64, name
    assert int(dfc.groupby("pat")["group"].nunique().max()) == 1, name
    return dfc


def main():
    repo = PROJECT_ROOT.parent
    audit_rows = []
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "script": str(Path(__file__).resolve()),
        "git_commit": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or "unknown",
        "git_dirty": bool(subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True).stdout.strip()),
        "packages": {p: importlib.metadata.version(p)
                     for p in ["numpy", "pandas", "openpyxl"]},
        "criteria": {
            "exclude": "start/end pupil columns <= 0",
            "impute": "totalFixTime==0 & nFixations==1 -> median per-fixation "
                      "duration (totalFixTime/nFixations) of nFixations>=2 & "
                      "totalFixTime>0 rows per (task, ROI), fallback task, "
                      "fallback dataset, times the row's 1 fixation",
        },
        "datasets": {},
    }
    for name in DATASETS:
        xlsx = repo / f"{name}.xlsx"
        dfc, excl, stats = clean_dataset(xlsx)
        prov["datasets"][name] = {"input_sha256": sha256_file(xlsx),
                                  **stats}
        clean_path = repo / f"{name}_clean.xlsx"
        dfc.to_excel(clean_path, index=False)
        prov["datasets"][name]["output_sha256"] = sha256_file(clean_path)
        excl.insert(0, "dataset", name)
        audit_rows.append(excl[["dataset", "pat", "task", "ROI", "group",
                                "reason", "offending_cols", "offending_values"]])
        dfc_back = _verify(name, clean_path, stats["n_rows_in"],
                           stats["n_excluded"])
        print(f"{name}: {stats['n_rows_in']} -> {stats['n_rows_out']} rows "
              f"(-{stats['n_excluded']} pupil<=0; "
              f"imputed {stats['n_imputed']} zero-fix) VERIFIED")

    audit = pd.concat(audit_rows, ignore_index=True)
    audit.to_csv(repo / "clean_excluded_rows.csv", index=False)
    (repo / "clean_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    print(f"excluded rows -> {repo / 'clean_excluded_rows.csv'} "
          f"({len(audit)} rows)")
    print(f"provenance -> {repo / 'clean_provenance.json'}")


if __name__ == "__main__":
    main()

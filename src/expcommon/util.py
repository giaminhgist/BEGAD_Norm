"""Provenance and run-log helpers shared by the two experiments."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(out_dir, cfg, extra):
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "command": " ".join(extra["argv"]),
        "git_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or "unknown",
        "git_dirty": bool(subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True).stdout.strip()),
        "config_sha256": cfg["config_sha256"],
        "data_sha256": {
            "wide": sha256_file(ROOT / cfg["data"]["wide_csv"]),
            "fold_assignments": sha256_file(ROOT / cfg["split"]["assignments_csv"]),
            "heldout_ad": sha256_file(ROOT / cfg["split"]["heldout_csv"]),
        },
        "split": {
            "n_folds": cfg["cv"]["n_folds"],
            "heldout_ad_appended_to_every_val_fold": True,
            "heldout_ad_subjects": sorted(pd.read_csv(
                ROOT / cfg["split"]["heldout_csv"])["subject_id"].astype(str)),
            "inner_seed": cfg["cv"]["inner_seed"],
            "inner_inner_seed": cfg["cv"]["inner_inner_seed"],
            "inner_folds": cfg["cv"]["inner_folds"],
        },
        "protocol": extra["protocol"],
        "packages": {p: importlib.metadata.version(p)
                     for p in ["numpy", "pandas", "scikit-learn", "scipy",
                               "PyYAML"]},
        "n_rows": extra["n_rows"],
    }
    prov.update(extra.get("extra_fields", {}))
    (Path(out_dir) / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")


def append_run_log(out_dir, mode, summary, n_rows, duration_s):
    log_path = Path(out_dir) / "run_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["timestamp_utc", "mode", "summary", "n_rows",
                        "duration_s", "status"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    mode, summary, n_rows, f"{duration_s:.1f}", "ok"])

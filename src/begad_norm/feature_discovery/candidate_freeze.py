"""Freeze the combined catalogue, configuration and extraction code before diagnostic screening."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from begad_norm.feature_discovery.dataset_features import SHORTFALL_NOTE, d_table
from begad_norm.feature_discovery.registry import combined_registry

PKG = Path(__file__).resolve().parent
FROZEN_CODE = ["registry.py", "dataset_features.py", "event_contracts.py", "extract_macro.py",
               "extract_micro.py", "statistics.py", "catalogue/base160.csv"]

LABEL_EXPOSURE = (
    "The proposing agent saw group counts, group-coded participant IDs/filenames and the repository README "
    "listing 25 legacy Kruskal-Wallis-screened features (old p<0.1 task lists). D definitions were authored from "
    "label-blind pooled profiles (workbook column distributions, raw event structure, transition/saccade/pupil "
    "validity) and paper Table II; no diagnostic contrast, p-value or effect size for any base or D candidate "
    "was computed before this freeze. Two candidates resembling legacy screened features were rejected to avoid "
    "reusing old winners (proposal_audit.csv)."
)


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def content_hashes(cfg_path: Path) -> dict:
    reg = combined_registry("unfrozen")
    d = d_table("unfrozen")
    return {
        "registry_sha256": _h(reg.to_csv(index=False)),
        "dataset_definitions_sha256": _h(d.to_csv(index=False)),
        "config_sha256": _h(Path(cfg_path).read_text()),
        "code_sha256": {f: _h((PKG / f).read_text()) for f in FROZEN_CODE},
    }


def write_freeze(run_dir: Path, cfg_path: Path, source_manifest: dict) -> dict:
    reg = combined_registry("unfrozen")
    hashes = content_hashes(cfg_path)
    freeze_id = "freeze_" + hashes["registry_sha256"][:12]
    n_d = int((reg.origin == "dataset100").sum())
    d = d_table(freeze_id)
    rec = {
        "freeze_id": freeze_id,
        "freeze_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_definition_count": int((reg.origin == "base160").sum()),
        "dataset_definition_count": n_d,
        "total_definition_count": len(reg),
        "target_total": 260,
        "extension_status": "complete" if n_d == 100 else "partial",
        "shortfall_note": SHORTFALL_NOTE if n_d < 100 else "",
        "dataset_level_split": d["level"].value_counts().to_dict(),
        "dataset_supported": int((d.feasibility_status == "supported").sum()),
        **hashes,
        "source_sha256": {i["path"]: i["sha256"] for i in source_manifest["items"]
                          if i["cohort"] != "not_used" and i["exists"]},
        "label_exposure_declaration": LABEL_EXPOSURE,
        "amendments": [],
    }
    (run_dir / "candidate_freeze.json").write_text(json.dumps(rec, indent=2))
    return rec


def verify_freeze(run_dir: Path, cfg_path: Path) -> dict:
    f = run_dir / "candidate_freeze.json"
    if not f.exists():
        raise RuntimeError("candidate_freeze.json missing: run the freeze stage before statistics")
    rec = json.loads(f.read_text())
    now = content_hashes(cfg_path)
    changed = [k for k in ("registry_sha256", "dataset_definitions_sha256", "config_sha256") if rec[k] != now[k]]
    changed += [f"code:{k}" for k, v in now["code_sha256"].items() if rec["code_sha256"].get(k) != v]
    if changed:
        raise RuntimeError(f"frozen candidate universe changed since freeze ({changed}); record an amendment and "
                           "re-freeze in a new run directory")
    return rec


def registry_with_freeze(freeze_id: str) -> pd.DataFrame:
    return combined_registry(freeze_id)

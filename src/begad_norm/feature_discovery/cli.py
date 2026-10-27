"""Command line entry point.

    uv run python -m begad_norm.feature_discovery.cli all \
        --config configs/feature_discovery.yaml \
        --run-dir outputs/feature_discovery/2026-10-27_v2_expanded260
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from begad_norm.feature_discovery import pipeline as pl

STAGES = ["audit", "propose", "freeze", "extract", "stats", "stability", "report"]
DEPENDS = {"propose": "audit", "freeze": "propose", "extract": "freeze", "stats": "extract",
           "stability": "stats", "report": "stability"}


def _git(root, *args):
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description="BEGAD feature discovery (no classifier training)")
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", default="configs/feature_discovery.yaml")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--resume", action="store_true",
                    help="skip stages already completed in this run directory with the same frozen inputs")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    ctx = pl.Ctx(root, Path(a.config).resolve(), Path(a.run_dir).resolve())
    state_f = ctx.run / "stage_state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    todo = STAGES if a.stage == "all" else [a.stage]
    if a.resume and state.get("freeze"):
        # resume is only valid for the same frozen universe
        from begad_norm.feature_discovery.candidate_freeze import verify_freeze
        verify_freeze(ctx.run, ctx.cfg_path)
    meta_f = ctx.run / "run_metadata.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {"stages": {}}
    for st in todo:
        dep = DEPENDS.get(st)
        if dep and not state.get(dep):
            raise SystemExit(f"stage {st} needs {dep} to be completed first in {ctx.run}")
        if a.resume and state.get(st):
            print(f"[resume] {st} already complete; skipping")
            continue
        t0 = time.time()
        print(f"[{datetime.now():%H:%M:%S}] stage {st} ...", flush=True)
        getattr(pl, f"stage_{st}")(ctx)
        state[st] = True
        # a rerun of an upstream stage invalidates downstream stages
        for later in STAGES[STAGES.index(st) + 1:]:
            if a.stage != "all":
                state.pop(later, None)
        state_f.write_text(json.dumps(state, indent=1))
        if meta_f.exists():  # a stage may have updated run metadata (e.g. run_status)
            meta = {**json.loads(meta_f.read_text()), "stages": meta["stages"]}
        meta["stages"][st] = dict(completed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  seconds=round(time.time() - t0, 1))
        meta.update(command=" ".join([Path(sys.executable).name, "-m", "begad_norm.feature_discovery.cli",
                                      *(argv or sys.argv[1:])]),
                    git_commit=_git(root, "rev-parse", "HEAD"), git_status_porcelain=_git(root, "status", "--porcelain"),
                    python=platform.python_version(), platform=platform.platform(), debug_only=False,
                    run_status=meta.get("run_status", "in_progress"))
        meta_f.write_text(json.dumps(meta, indent=1))
    print("done")


if __name__ == "__main__":
    main()

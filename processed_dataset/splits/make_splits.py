#!/usr/bin/env python3
"""Generate the training CV split for the processed dataset (stdlib only).

Scheme (full record in split_manifest.json):
- Hold out 4 AD subjects; they never enter any training fold.
- Remaining 153 subjects: stratified-by-group 4-fold CV. Within each group the
  subject ids are shuffled with random.Random(SEED) and assigned round-robin to
  folds 1..4 (same method as subject-level run_baselines.stratified_folds).
- Validation set of a fold = that fold's subjects + the 4 held-out AD subjects.

Deterministic: rerunning with the same SEED rewrites identical files.
"""

import csv
import json
import random
from pathlib import Path

SEED = 42
N_FOLDS = 4
N_HELDOUT_AD = 4

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "processed_dataset_v1_wide.csv"
OUT_ASSIGN = HERE / "cv_fold_assignments.csv"
OUT_HELDOUT = HERE / "heldout_ad_subjects.csv"
OUT_MANIFEST = HERE / "split_manifest.json"


def read_subjects():
    with DATA.open(newline="") as f:
        rows = list(csv.DictReader(f))
    subs = [(r["subject_id"], r["group"]) for r in rows]
    assert len({s for s, _ in subs}) == 157, "expected 157 unique subjects"
    counts = {}
    for _, g in subs:
        counts[g] = counts.get(g, 0) + 1
    assert counts == {"CN": 100, "MCI": 41, "AD": 16}, counts
    return subs


def main():
    rng = random.Random(SEED)
    subs = read_subjects()
    by_group = {}
    for sid, g in subs:
        by_group.setdefault(g, []).append(sid)
    for g in by_group:
        by_group[g].sort()

    ad_ids = sorted(by_group["AD"])
    heldout = sorted(rng.sample(ad_ids, N_HELDOUT_AD))
    heldout_set = set(heldout)

    assignments = []  # (subject_id, group, fold)
    for g, ids in by_group.items():
        pool = [s for s in ids if s not in heldout_set]
        rng.shuffle(pool)
        for i, sid in enumerate(pool):
            assignments.append((sid, g, i % N_FOLDS + 1))

    with OUT_ASSIGN.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "group", "fold"])
        w.writerows(sorted(assignments))

    with OUT_HELDOUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "group", "role"])
        w.writerows((sid, "AD", "heldout_ad") for sid in heldout)

    folds = {f: {"CN": 0, "MCI": 0, "AD": 0} for f in range(1, N_FOLDS + 1)}
    for _, g, fld in assignments:
        folds[fld][g] += 1
    val_folds = {
        f: {g: n + (N_HELDOUT_AD if g == "AD" else 0) for g, n in c.items()}
        for f, c in folds.items()
    }
    train_folds = {
        f: {
            g: sum(folds[k][g] for k in folds if k != f)
            for g in ("CN", "MCI", "AD")
        }
        for f in folds
    }

    manifest = {
        "scheme": "holdout_4_ad_plus_stratified_4fold",
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_heldout_ad": N_HELDOUT_AD,
        "source_dataset": "processed_dataset_v1_wide.csv (157 subjects: CN 100 / MCI 41 / AD 16; "
                          "same subject set as processed_dataset_v2_long.csv)",
        "method": "heldout AD: random.Random(seed).sample of the 16 AD subjects. "
                  "CV folds: stratified by group; within-group seeded shuffle then round-robin "
                  "assignment to folds 1..4 (same method as subject-level "
                  "run_baselines.stratified_folds).",
        "heldout_ad_subjects": heldout,
        "heldout_ad_role": "excluded from every training fold; appended to every validation fold",
        "fold_composition_train": train_folds,
        "fold_composition_validation": val_folds,
        "validation_rule": "validation(f) = subjects of fold f + the 4 held-out AD subjects",
        "caveats": [
            "The same 4 held-out AD subjects appear in every validation fold, so validation folds "
            "are not independent and pooled CV metrics overweight these 4 subjects.",
            "The 4 held-out AD subjects are never seen in training, so the model does not learn "
            "from them.",
            "Report per-fold metrics separately and do not treat the pooled CV mean as an unbiased "
            "generalization estimate. Reporting the 4 held-out AD separately is optional.",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"heldout AD: {heldout}")
    print("fold composition (training) :", train_folds)
    print("fold composition (validation):", val_folds)
    print(f"wrote {OUT_ASSIGN.name}, {OUT_HELDOUT.name}, {OUT_MANIFEST.name}")


if __name__ == "__main__":
    main()

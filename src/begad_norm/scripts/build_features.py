"""Build the per-(participant, task) feature table for normative learning.

Pipeline per roster participant file:
  read CSV -> parse decimal-comma columns -> clean fixations (Final & !HasNulls
  & fixation events, duration window, per-task media bounds, pupil outliers)
  -> extract saccade runs -> compute features per (participant, task):
    - the 45 EMS features (verbatim port) on MCSpx coordinates and binocular
      mean pupil (screen geometry from config.TASK_GEOMETRY per task),
    - pupbl_ (7): same formulas on baseline-corrected binocular pupil,
    - pupdiff_ (7): same formulas on the raw L - R pupil difference,
    - sac_ (13): measured-saccade features (amp/directions attached to
      fixations + saccade-run durations).
Pairs with < MIN_FIX clean fixations yield all-NaN rows (EMS convention).

Outputs (processed_dataset/): features.csv, metadata.csv, feature_names.txt,
quality_report.txt. (No parquet: pyarrow is not a project dependency.)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from begad_norm.config import (
    COHORT_GROUPS,
    DATA_DIR,
    GROUP_LABELS,
    MIN_FIX,
    OUT_DIR,
    PARTICIPANTS_CSV,
    TASK_GEOMETRY,
    TASK_IDS,
)
from begad_norm.features.saccade import compute_saccade_features
from begad_norm.features.stimulus import FEATURE_NAMES, compute_stimulus_features
from begad_norm.rawdata import (
    clean_fixations,
    extract_saccade_runs,
    parse_decimal_columns,
    participant_id_from_filename,
    read_participant_file,
)

PUPBL_NAMES = [n.replace("pup_", "pupbl_") for n in FEATURE_NAMES if n.startswith("pup_")]
PUPDIFF_NAMES = [n.replace("pup_", "pupdiff_") for n in FEATURE_NAMES if n.startswith("pup_")]
SAC_NAMES = [
    "sac_count", "sac_rate", "sac_dur_mean", "sac_dur_std", "sac_dur_total",
    "sac_dur_max", "sac_amp_mean", "sac_amp_std", "sac_amp_max",
    "sac_absdir_mean", "sac_absdir_var", "sac_turn_mean", "sac_turn_std",
]
ALL_FEATURE_NAMES = FEATURE_NAMES + PUPBL_NAMES + PUPDIFF_NAMES + SAC_NAMES
TABLE_COLUMNS = ["participant_id", "task"] + ALL_FEATURE_NAMES


def features_for_pair(g: pd.DataFrame, runs: pd.DataFrame, w: float, h: float) -> dict:
    """Compute all 72 features for one (participant, task) pair."""
    if len(g) < MIN_FIX:
        return {name: np.nan for name in ALL_FEATURE_NAMES}

    x, y = g["x"].to_numpy(), g["y"].to_numpy()
    dur = g["dur"].to_numpy()
    f = compute_stimulus_features(x, y, dur, g["pup"].to_numpy(),
                                  screen_w=w, screen_h=h)
    fbl = compute_stimulus_features(x, y, dur, g["pup_corrected"].to_numpy(),
                                    screen_w=w, screen_h=h)
    fdiff = compute_stimulus_features(x, y, dur, g["pupdiff"].to_numpy(),
                                      screen_w=w, screen_h=h)
    for name in PUPBL_NAMES:
        f[name] = fbl[name.replace("pupbl_", "pup_")]
    for name in PUPDIFF_NAMES:
        f[name] = fdiff[name.replace("pupdiff_", "pup_")]

    total_time = dur.sum() + (f["tem_ifi_mean"] * (len(g) - 1) if len(g) > 1 else 0.0)
    f.update(compute_saccade_features(
        g["amp"].to_numpy(), g["absdir"].to_numpy(), g["reldir"].to_numpy(),
        runs["run_dur"].to_numpy(), total_time,
    ))
    return f


def build_all(participants_subset=None):
    """Build the feature table + metadata + per-file stats for the roster cohort.

    Returns (features, metadata, stats). features has one row per
    (participant, task) pair of the subset (default: full 157-participant
    roster); metadata one row per participant; stats is keyed by participant_id.
    """
    roster = pd.read_csv(PARTICIPANTS_CSV)
    if participants_subset:
        roster = roster[roster["participant_id"].isin(participants_subset)].copy()
    if roster.empty:
        raise ValueError(f"no roster participants match {participants_subset}")

    rows, meta_rows, stats = [], [], {}
    for rec in roster.itertuples(index=False):
        path = DATA_DIR / rec.filename
        df = parse_decimal_columns(read_participant_file(path))
        # cohort guards: roster-driven, assert file consistency
        file_groups = df["Group"].dropna().unique()
        file_followups = df["IsFollowup"].dropna().unique()
        assert file_groups.size == 1 and float(file_groups[0]) == float(rec.group), (
            f"{rec.filename}: Group {file_groups} != roster group {rec.group}")
        assert file_followups.size == 1 and float(file_followups[0]) == 0.0, (
            f"{rec.filename}: IsFollowup {file_followups} != 0")

        clean, per_file_stats = clean_fixations(df, return_stats=True)
        runs = extract_saccade_runs(df, clean)
        stats[rec.participant_id] = per_file_stats

        for task in TASK_IDS:
            g = clean[clean["task"] == task]
            r = runs[runs["task"] == task]
            w, h = TASK_GEOMETRY[task][2], TASK_GEOMETRY[task][3]
            feat = features_for_pair(g, r, w, h)
            feat["participant_id"] = rec.participant_id
            feat["task"] = task
            rows.append(feat)

        meta_rows.append({
            "participant_id": rec.participant_id,
            "group": rec.group,
            "group_label": rec.group_label,
            "file": rec.filename,
            "n_clean_fixations": per_file_stats["clean"],
            "n_tasks_present": int(clean["task"].nunique()),
            "n_missing_baseline": int(
                df["PupilLeftBaseline"].isna().any()
                or df["PupilRightBaseline"].isna().any()
            ),
        })

    features = pd.DataFrame(rows)[TABLE_COLUMNS]
    features["task"] = features["task"].astype("int8")
    metadata = pd.DataFrame(meta_rows)
    return features, metadata, stats


def write_outputs(features, metadata, stats, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "features.csv", index=False)
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    (out_dir / "feature_names.txt").write_text(
        "\n".join(ALL_FEATURE_NAMES) + "\n", encoding="utf-8")

    n_rows, n_feats = features.shape
    nan_frac = features[ALL_FEATURE_NAMES].isna().mean()
    all_nan_rows = features[features[ALL_FEATURE_NAMES].isna().all(axis=1)]
    lines = [
        f"participants: {metadata['participant_id'].nunique()}",
        f"rows: {n_rows} x {n_feats} ({len(ALL_FEATURE_NAMES)} features + 2 id cols)",
        f"all-NaN rows: {len(all_nan_rows)}",
    ]
    if len(all_nan_rows):
        lines.append("all-NaN pairs:")
        for _, r in all_nan_rows.iterrows():
            lines.append(f"  {r['participant_id']} task {r['task']}")
    total = {k: 0 for k in ["raw", "fixation_filter_dropped", "duration_dropped",
                            "bounds_dropped", "pupil_dropped", "clean"]}
    for s in stats.values():
        for k in total:
            total[k] += s[k]
    lines.append(
        "stage drops (all participants): "
        + ", ".join(f"{k}={total[k]}" for k in total))
    lines.append("group counts: "
                 + ", ".join(f"{g}={len(metadata[metadata['group'] == g])}"
                             for g in sorted(GROUP_LABELS)))
    lines.append("participants with NaN pupil baselines (pupbl_* NaN): "
                 + str(int(metadata["n_missing_baseline"].sum())))
    lines.append("per-feature NaN fraction (worst 5):")
    for name, frac in nan_frac.nlargest(5).items():
        lines.append(f"  {name}: {frac:.4%}")
    (out_dir / "quality_report.txt").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the BEGAD feature table.")
    parser.add_argument("--participants", default=None,
                        help="Comma-separated participant_id subset (smoke runs)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    subset = args.participants.split(",") if args.participants else None
    t0 = time.time()
    features, metadata, stats = build_all(participants_subset=subset)
    write_outputs(features, metadata, stats, args.out_dir)
    print(f"built {len(features)} rows x {len(ALL_FEATURE_NAMES)} features "
          f"for {len(metadata)} participants in {time.time() - t0:.1f}s "
          f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

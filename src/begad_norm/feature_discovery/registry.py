"""Combined feature registry: 160 base definitions + dataset-derived D definitions.

The base catalogue text is parsed verbatim from the instruction-pack catalogue
(01_FEATURE_CATALOG_160.md) into ``catalogue/base160.csv`` (committed). The
availability decision for every base ID is made explicitly here, from the
source audit (see semantic_audit.md in a run directory). D definitions live in
``dataset_features.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from begad_norm.feature_discovery import EXTRACTION_VERSION

CATALOGUE_DIR = Path(__file__).resolve().parent / "catalogue"
BASE_CSV = CATALOGUE_DIR / "base160.csv"

COHORTS = ("RAW157", "ROI64_T1235", "ROI64_T5")
TASKS = ("KD1", "KD2", "KD3", "READING")

REGISTRY_COLUMNS = [
    "feature_id", "origin", "level", "family", "name", "formula", "unit",
    "source_requirements", "allowed_tasks", "scope", "within_subject_support",
    "conditioning", "alias_group", "supported_datasets", "status",
    "status_reason", "semantic_flags", "extraction_version", "freeze_id",
]

_ROW = re.compile(r"^\|\s*([MA]\d{3})\s*\|(.*)\|\s*$")


def parse_base_catalogue(md_path: Path) -> pd.DataFrame:
    """Parse the 160 catalogue rows (ID, name, definition, unit, sources, support)."""
    rows, family = [], None
    for line in Path(md_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## Micro") or line.startswith("## Macro"):
            family = line.removeprefix("## ").strip()
        m = _ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        if len(cells) != 5:
            raise ValueError(f"unexpected catalogue row: {line}")
        name, formula, unit, sources, support = cells
        fid = m.group(1)
        rows.append({
            "feature_id": fid,
            "level": "micro" if fid.startswith("M") else "macro",
            "family": family, "name": name, "formula": formula, "unit": unit,
            "source_requirements": sources, "task_support": support,
        })
    df = pd.DataFrame(rows)
    ids = set(df.feature_id)
    expected = {f"M{i:03d}" for i in range(1, 81)} | {f"A{i:03d}" for i in range(1, 81)}
    if ids != expected or len(df) != 160:
        raise ValueError(f"base catalogue must contain exactly M001-M080 and A001-A080; got {len(df)}")
    return df


def load_base_catalogue() -> pd.DataFrame:
    return pd.read_csv(BASE_CSV, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# Availability decisions for base definitions (from the source audit).
# Each supported entry: cohorts, scopes, semantic flags. Everything else is
# blocked with an explicit reason.
# ---------------------------------------------------------------------------
PER_TASK = ",".join(TASKS)

SUPPORTED_BASE: dict[str, dict] = {
    # ROI workbook micro mappings (E, verified against paper Table II)
    "M001": dict(cohorts="ROI64_T1235,ROI64_T5", scope="task_roi",
                 flags="E:nFixations; missing ROI row = unavailable (not zero)"),
    "M051": dict(cohorts="ROI64_T1235,ROI64_T5", scope="task_roi",
                 flags="E:totalSpokenTime (Table II 'total time spent speaking out loud on a ROI'); "
                       "union-vs-sum of repeated utterances not verifiable without TextGrids"),
    "M061": dict(cohorts="ROI64_T1235,ROI64_T5", scope="task_roi",
                 flags="E:speechDelay (Table II 'first fixation on ROI to beginning of saying'); "
                       "fixation onset convention assumed; T1235 export contains no delay >=10 s (right-truncated window), so "
                       "values <=0 or >=10 s are censored in both cohorts"),
    # ROI workbook macro aggregates of M061
    "A061": dict(cohorts="ROI64_T1235,ROI64_T5", scope=PER_TASK,
                 flags="aggregate of E-mapped M061; >=60% expected ROI coverage"),
    "A062": dict(cohorts="ROI64_T1235,ROI64_T5", scope=PER_TASK,
                 flags="aggregate of E-mapped M061; >=3 ROIs and >=60% coverage"),
    "A063": dict(cohorts="ROI64_T1235,ROI64_T5", scope=PER_TASK,
                 flags="aggregate of E-mapped M061; >=10 ROIs and >=60% coverage"),
    "A079": dict(cohorts="ROI64_T1235", scope="CROSS_BOTH",
                 flags="A061@READING minus equal-card mean of A061@KD1..3"),
    # Raw157 fixation/saccade/pupil macro features
    "A002": dict(cohorts="RAW157", scope=PER_TASK, flags="task membership from export Task label; straddling events excluded"),
    "A003": dict(cohorts="RAW157", scope=PER_TASK, flags="task membership from export Task label"),
    "A005": dict(cohorts="RAW157", scope=PER_TASK, flags=""),
    "A006": dict(cohorts="RAW157", scope=PER_TASK, flags=""),
    "A007": dict(cohorts="RAW157", scope=PER_TASK, flags=""),
    "A021": dict(cohorts="RAW157", scope=PER_TASK, flags="I-VT saccade events (may split one fixation-to-fixation movement)"),
    "A023": dict(cohorts="RAW157", scope=PER_TASK,
                 flags="matched saccades only: one saccade event fully covering a gap-free fixation-to-fixation interval; "
                       "amplitude = Tobii fixation-to-fixation angle on landing fixation"),
    "A024": dict(cohorts="RAW157", scope=PER_TASK, flags="matched saccades only"),
    "A025": dict(cohorts="RAW157", scope=PER_TASK, flags="matched saccades only"),
    "A026": dict(cohorts="RAW157", scope=PER_TASK, flags="matched saccades only"),
    "A027": dict(cohorts="RAW157", scope=PER_TASK, flags="matched saccades only"),
    "A028": dict(cohorts="RAW157", scope=PER_TASK, flags="matched saccades only; mean speed, not peak velocity"),
    "A029": dict(cohorts="RAW157", scope=PER_TASK, flags="matched saccade population"),
    "A030": dict(cohorts="RAW157", scope=PER_TASK, flags="matched saccade population"),
    "A031": dict(cohorts="RAW157", scope=PER_TASK, flags="sum over gap-free fixation-to-fixation transitions"),
    "A033": dict(cohorts="RAW157", scope=PER_TASK, flags="in-media fixations only (Tobii clips MCS coordinates off-media)"),
    "A034": dict(cohorts="RAW157", scope=PER_TASK, flags="in-media fixations only"),
    "A035": dict(cohorts="RAW157", scope=PER_TASK, flags="in-media fixations only"),
    "A036": dict(cohorts="RAW157", scope=PER_TASK, flags="gap-free transitions between in-media fixations"),
    "A042": dict(cohorts="RAW157", scope=PER_TASK,
                 flags="event-boundary pupil samples (export keeps onset/offset rows only), not a uniform time series"),
    "A043": dict(cohorts="RAW157", scope=PER_TASK, flags="event-boundary pupil samples"),
    "A046": dict(cohorts="RAW157", scope=PER_TASK, flags="event-boundary pupil samples; illumination not controlled"),
    "A076": dict(cohorts="RAW157", scope="CROSS_KD", flags="A005@KD3 - A005@KD1"),
}

R_NO_ROI = "blocked_source: no ROI bounding boxes, stimulus layout, reading order or line metadata in the repository"
R_NO_EVT_ROI = ("blocked_source: ROI workbooks hold per-ROI summaries only; no fixation/visit chronology per ROI, "
                "and raw157 events have no ROI geometry")
R_NO_V = "blocked_source: no audio, TextGrid, utterance intervals or error annotations; only workbook speech summaries"
R_NO_L = "blocked_source: no word text, letter or syllable metadata"
R_NO_T = "blocked_source: media onset/offset not exported (rows exist only at gaze events)"
R_NO_P = ("blocked_source: raw exports contain pupil only at event onset/offset rows; "
          "no continuous samples for the required time bins/windows")
R_TFT = ("blocked_semantics: totalFixTime is 0 for every single-fixation row and ~0.0017 s per fixation for ~45% "
         "of multi-fixation rows; it is not a sum of fixation durations")
R_SACC = ("blocked_semantics: tiny/long/regressive counts exceed nTobiiSaccades in thousands of rows; "
          "event populations unreconciled")
R_BASE = ("blocked_semantics: raw PupilLeft/RightBaseline does not reproduce the protocol's first-10-instruction-"
          "fixation rule; workbook B fields use a different unidentified constant with inverted sign")

BLOCKED_BASE: dict[str, str] = {}
for i in (2, 4, 5, 6, 7, 9, 10):
    BLOCKED_BASE[f"M{i:03d}"] = R_NO_EVT_ROI
for i in (3, 8):
    BLOCKED_BASE[f"M{i:03d}"] = R_TFT
for i in range(11, 21):
    BLOCKED_BASE[f"M{i:03d}"] = R_NO_EVT_ROI + "; ROI ID order not verified as reading order"
for i in range(21, 29):
    BLOCKED_BASE[f"M{i:03d}"] = R_NO_EVT_ROI
BLOCKED_BASE["M029"] = R_SACC
BLOCKED_BASE["M030"] = R_SACC
for i in range(31, 41):
    BLOCKED_BASE[f"M{i:03d}"] = R_NO_ROI
for i in (41, 42, 43, 47, 48, 49, 50):
    BLOCKED_BASE[f"M{i:03d}"] = R_NO_EVT_ROI + "; no pupil samples per ROI"
BLOCKED_BASE["M044"] = ("blocked_semantics: export 'pupil at first fixation' is a single exported value whose "
                        "fixation-average convention is undocumented; T1235 values are sign-inverted relative to an "
                        "unidentified constant")
BLOCKED_BASE["M045"] = ("blocked_semantics: Table II 'last fixation' is the landing fixation after the last saccade "
                        "leaving the ROI, not the last fixation inside r")
BLOCKED_BASE["M046"] = "blocked_semantics: depends on M045 endpoint convention (see D001 for the export convention)"
for i in range(52, 61):
    BLOCKED_BASE[f"M{i:03d}"] = R_NO_V
BLOCKED_BASE["M062"] = ("blocked_semantics: endSpeechDelay references the post-exit landing fixation (Table II caption) "
                        "with unspecified fixation time point; 10 s export cap")
for i in range(63, 71):
    BLOCKED_BASE[f"M{i:03d}"] = R_NO_V
for i in range(71, 76):
    BLOCKED_BASE[f"M{i:03d}"] = R_NO_L
BLOCKED_BASE["M076"] = R_NO_V
BLOCKED_BASE["M077"] = R_NO_EVT_ROI
BLOCKED_BASE["M078"] = R_NO_EVT_ROI
BLOCKED_BASE["M079"] = R_NO_ROI
BLOCKED_BASE["M080"] = R_NO_V
BLOCKED_BASE["A001"] = R_NO_T
BLOCKED_BASE["A004"] = R_NO_T + " (needs A001)"
BLOCKED_BASE["A008"] = R_NO_EVT_ROI + " (needs M006)"
BLOCKED_BASE["A009"] = R_NO_EVT_ROI + " (needs M006)"
BLOCKED_BASE["A010"] = R_TFT + " (needs M008)"
for i in range(11, 21):
    BLOCKED_BASE[f"A{i:03d}"] = R_NO_ROI
BLOCKED_BASE["A022"] = R_NO_T + " (needs A001)"
BLOCKED_BASE["A032"] = R_NO_ROI
BLOCKED_BASE["A037"] = R_TFT + " (needs complete D(r))"
for i in (38, 39, 40):
    BLOCKED_BASE[f"A{i:03d}"] = R_NO_ROI
BLOCKED_BASE["A041"] = R_BASE
for i in (44, 45, 48, 49):
    BLOCKED_BASE[f"A{i:03d}"] = R_NO_P
BLOCKED_BASE["A047"] = R_BASE + "; " + R_NO_P
BLOCKED_BASE["A050"] = R_BASE
for i in range(51, 61):
    BLOCKED_BASE[f"A{i:03d}"] = R_NO_V
for i in (64, 65, 66):
    BLOCKED_BASE[f"A{i:03d}"] = BLOCKED_BASE["M062"]
for i in range(67, 71):
    BLOCKED_BASE[f"A{i:03d}"] = R_NO_V
BLOCKED_BASE["A071"] = R_NO_L
BLOCKED_BASE["A072"] = R_NO_L
BLOCKED_BASE["A073"] = R_NO_L
BLOCKED_BASE["A074"] = R_NO_ROI
BLOCKED_BASE["A075"] = R_NO_T + " (elapsed-time thirds need task onset/offset)"
BLOCKED_BASE["A077"] = R_NO_T + " (needs A001)"
BLOCKED_BASE["A078"] = R_NO_ROI + " (needs A013)"
BLOCKED_BASE["A080"] = R_NO_V


def base_registry(freeze_id: str = "unfrozen") -> pd.DataFrame:
    cat = load_base_catalogue()
    overlap = set(SUPPORTED_BASE) & set(BLOCKED_BASE)
    missing = set(cat.feature_id) - set(SUPPORTED_BASE) - set(BLOCKED_BASE)
    if overlap or missing:
        raise ValueError(f"base availability incomplete: overlap={overlap} missing={missing}")
    rows = []
    for r in cat.itertuples(index=False):
        sup = SUPPORTED_BASE.get(r.feature_id)
        if sup:
            status, reason, cohorts, scope, flags = (
                "supported", "", sup["cohorts"], sup["scope"], sup["flags"])
        else:
            reason = BLOCKED_BASE[r.feature_id]
            status = reason.split(":")[0]
            cohorts, scope, flags = "", "", ""
        rows.append({
            "feature_id": r.feature_id, "origin": "base160", "level": r.level,
            "family": r.family, "name": r.name, "formula": r.formula, "unit": r.unit,
            "source_requirements": r.source_requirements,
            "allowed_tasks": "READING" if r.task_support.startswith("R;") or r.task_support.startswith("READING") else (
                "CROSS" if r.task_support.startswith("CROSS") else (
                    "GLOBAL" if r.task_support.startswith("GLOBAL") else PER_TASK)),
            "scope": scope or ("task_roi" if r.level == "micro" else r.task_support.split(";")[0]),
            "within_subject_support": r.task_support,
            "conditioning": "conditional" if "conditional" in (r.task_support + r.formula).lower() else "",
            "alias_group": "", "supported_datasets": cohorts, "status": status,
            "status_reason": reason, "semantic_flags": flags,
            "extraction_version": EXTRACTION_VERSION, "freeze_id": freeze_id,
        })
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def combined_registry(freeze_id: str = "unfrozen") -> pd.DataFrame:
    from begad_norm.feature_discovery.dataset_features import dataset_registry

    reg = pd.concat([base_registry(freeze_id), dataset_registry(freeze_id)], ignore_index=True)
    if reg.feature_id.duplicated().any():
        raise ValueError("duplicate feature IDs in combined registry")
    if not set(reg.level) <= {"micro", "macro"}:
        raise ValueError("every definition needs level micro or macro")
    return reg

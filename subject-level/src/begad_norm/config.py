"""BEGAD_Norm constants: task media geometry, cleaning parameters, paths."""

from pathlib import Path

# Cohort protocol (notebook Analysisbase_for_Ali.ipynb)
COHORT_GROUPS = (0, 1, 2)  # 0=CO, 1=MCI, 2=AD; excludes NPH (3) and -11
GROUP_LABELS = {0: "CO", 1: "MCI", 2: "AD"}

# EMS cleaning parameters (kept for fidelity)
DUR_MIN = 40.0
DUR_MAX = 2000.0
MIN_FIX = 2
PUPIL_SD = 4.0
PUPIL_EPS = 1e-6
EPS = 1e-6

# Per-task media geometry on the 1920x1080 screen: (pos_x, pos_y, width, height).
# Verified constant across all 205 raw files (see tests/test_config.py).
TASK_GEOMETRY = {
    0: (0.0, 0.0, 1920.0, 1080.0),
    1: (586.0, 258.0, 748.0, 564.0),
    2: (0.0, 0.0, 1920.0, 1080.0),
    3: (587.0, 257.0, 746.0, 566.0),
    4: (588.0, 278.0, 744.0, 525.0),
    5: (592.0, 286.0, 737.0, 508.0),
    6: (0.0, 0.0, 1920.0, 1080.0),
    7: (485.0, 332.0, 950.0, 417.0),
    8: (0.0, 0.0, 1920.0, 1080.0),
    9: (501.0, 82.0, 918.0, 917.0),
    10: (0.0, 0.0, 1920.0, 1080.0),
}
TASK_IDS = tuple(range(11))

# Columns stored with European decimal commas (pandas 3 reads them as str dtype).
DECIMAL_COMMA_COLS = (
    "SaccadicAmplitude",
    "AbsoluteSaccadicDirection",
    "RelativeSaccadicDirection",
    "PupilLeft",
    "PupilRight",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dataset_v2" / "20250305"
PARTICIPANTS_CSV = PROJECT_ROOT / "dataset_v2" / "participants_20250305.csv"
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "20250305"
OUT_DIR = PROJECT_ROOT / "processed_dataset"

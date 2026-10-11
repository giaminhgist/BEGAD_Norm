"""TASK_GEOMETRY in config must match the per-task media geometry of the raw files."""

import unittest

import pandas as pd

from begad_norm.config import RAW_DATA_DIR, TASK_GEOMETRY, TASK_IDS


class TaskGeometryTests(unittest.TestCase):
    def test_matches_raw_files(self):
        if not RAW_DATA_DIR.exists():
            self.skipTest("raw dataset/ not present")
        files = sorted(RAW_DATA_DIR.glob("*.csv"))
        self.assertGreater(len(files), 0)
        seen = {t: set() for t in TASK_IDS}
        for f in files:
            df = pd.read_csv(
                f,
                usecols=[
                    "MediaPosX (ADCSpx)",
                    "MediaPosY (ADCSpx)",
                    "MediaWidth",
                    "MediaHeight",
                    "Task",
                ],
            )
            for task, g in df[df["Task"].isin(TASK_IDS)].groupby("Task"):
                for row in g.drop_duplicates().itertuples(index=False):
                    seen[task].add((row[0], row[1], row[2], row[3]))
        for task in TASK_IDS:
            with self.subTest(task=task):
                self.assertEqual(seen[task], {TASK_GEOMETRY[task]}, f"task {task}")


if __name__ == "__main__":
    unittest.main()

import numpy as np
import pandas as pd
import pytest

from begad_norm.feature_discovery.event_contracts import build_fixations, build_saccades, build_transitions
from begad_norm.feature_discovery.extract_macro import task_features
from begad_norm.feature_discovery.extract_micro import macro_values, micro_values
from begad_norm.feature_discovery.statistics import build_matrix

COLS = ["RecordingTimestamp", "GazeEventType", "GazeEventDuration", "FixationIndex", "SaccadeIndex", "Final", "Task",
        "FixationPointX (MCSpx)", "FixationPointY (MCSpx)", "SaccadicAmplitude", "ValidityLeft", "ValidityRight",
        "PupilLeft", "PupilRight", "GazePointX (ADCSpx)", "GazePointY (ADCSpx)", "MediaPosX (ADCSpx)",
        "MediaPosY (ADCSpx)", "MediaWidth", "MediaHeight", "DistanceLeft", "DistanceRight"]


def _fix(i, on, dur, x, amp, task=3, pup=(3.0, 3.1), task_fin=None):
    base = dict(GazeEventType="Fixation", GazeEventDuration=dur, FixationIndex=i, SaccadeIndex=np.nan,
                FixationPointX=x, SaccadicAmplitude=amp)
    rows = []
    for ts, fin, p, t in ((on, False, pup[0], task), (on + dur - 3, True, pup[1], task_fin or task)):
        rows.append({"RecordingTimestamp": ts, "GazeEventType": "Fixation", "GazeEventDuration": dur,
                     "FixationIndex": i, "SaccadeIndex": np.nan, "Final": fin, "Task": t,
                     "FixationPointX (MCSpx)": x, "FixationPointY (MCSpx)": 100.0, "SaccadicAmplitude": amp,
                     "ValidityLeft": 0, "ValidityRight": 0, "PupilLeft": p, "PupilRight": p,
                     "GazePointX (ADCSpx)": x + 500, "GazePointY (ADCSpx)": 400, "MediaPosX (ADCSpx)": 500.0,
                     "MediaPosY (ADCSpx)": 300.0, "MediaWidth": 700.0, "MediaHeight": 500.0,
                     "DistanceLeft": 600.0, "DistanceRight": 600.0})
    del base
    return rows


def _sac(i, on, dur, task=3):
    return [{"RecordingTimestamp": ts, "GazeEventType": "Saccade", "GazeEventDuration": dur, "FixationIndex": np.nan,
             "SaccadeIndex": i, "Final": fin, "Task": task, "ValidityLeft": 0, "ValidityRight": 0,
             "PupilLeft": 3.0, "PupilRight": 3.0} for ts, fin in ((on, False), (on + dur - 3, True))]


@pytest.fixture
def raw():
    rows = []
    rows += _fix(1, 0, 200, 100, 9.0)          # first fixation: incoming amplitude refers to previous task
    rows += _sac(1, 200, 30)                    # matched: one saccade covers 200..230
    rows += _fix(2, 230, 300, 200, 2.0)
    rows += _sac(2, 530, 15) + _sac(3, 547, 15)  # split saccade: two events, gap-free
    rows += _fix(3, 562, 250, 50, 3.0)
    rows += _sac(4, 812, 20)                    # then 500 ms uncovered -> tracking gap
    rows += _fix(4, 1332, 220, 150, 4.0)
    rows += _fix(5, 1600, 100, 160, 1.0, task_fin=4)  # straddles task boundary
    return pd.DataFrame(rows, columns=COLS)


def test_event_reconstruction_matching_and_no_bridging(raw):
    fix = build_fixations(raw)
    sac = build_saccades(raw)
    tr = build_transitions(fix, sac)
    assert fix.set_index("fixation_id").loc[5, "valid"] == False  # noqa: E712 straddling excluded
    assert len(sac) == 4 and (sac.n_rows == 2).all()
    t = tr.set_index("to_id")
    assert t.loc[2, "status"] == "gap_free" and t.loc[2, "matched"] and t.loc[2, "sacc_dur_ms"] == 30
    assert t.loc[2, "amp_deg"] == 2.0                    # amplitude from the landing fixation
    assert t.loc[3, "status"] == "gap_free" and not t.loc[3, "matched"] and t.loc[3, "n_saccades"] == 2
    assert t.loc[4, "status"] == "gap" and np.isnan(t.loc[4, "amp_deg"])  # never bridged
    assert 5 not in t.index


def test_speed_uses_matched_saccade_duration_not_fixation_dwell(raw):
    fix = build_fixations(raw)
    sac = build_saccades(raw)
    tr = build_transitions(fix, sac)
    f = task_features(fix, sac, tr, raw, 3)
    assert f["A023"][0] == pytest.approx(2.0)            # only the matched saccade
    assert f["A028"][0] == pytest.approx(2.0 / 0.030)    # amplitude / matched saccade duration
    assert f["A031"][0] == pytest.approx(5.0)            # gap-free transitions 2 + 3; the gap is excluded
    assert f["A021"][0] == 4
    assert f["A003"][0] == 4                              # straddling fixation not counted


def _wb(rows):
    base = dict(group=0, totalFixTime=0.0, nTobiiSaccades=1, longSaccades=0, tinySaccades=0, endSpeechDelay=0.5,
                startPupLB=0.3, startPupRB=0.3, endPupLB=0.2, endPupRB=0.2, diffPupL=0.1, diffPupR=0.1,
                maxPupLB=0.1, minPupLB=0.4, diffXpupL=0.3, regSaccades=0, saccadeTotLength=3.0)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_workbook_statuses_missing_censored_and_guards():
    w = _wb([dict(pat="P1", task=1, ROI=0, nFixations=2, totalSpokenTime=0.4, speechDelay=0.8),
             dict(pat="P1", task=1, ROI=1, nFixations=1, totalSpokenTime=0.04, speechDelay=10.2),
             dict(pat="P1", task=1, ROI=2, nFixations=3, totalSpokenTime=0.5, speechDelay=0.9, startPupLB=0.5)])
    m = micro_values(w, "ROI64_T1235").set_index(["roi_id", "feature_id"])
    assert m.loc[(1, "M061"), "measurement_status"] == "censored" and np.isnan(m.loc[(1, "M061"), "value"])
    assert m.loc[(1, "D005"), "measurement_status"] == "insufficient_within_subject_support"
    assert m.loc[(0, "D003"), "measurement_status"] == "observed_zero"
    assert m.loc[(2, "D002"), "measurement_status"] == "invalid_measurement"   # start outside [max, min] range
    # ROI 3..39 have no rows: the macro gate counts only observed ROIs, never zeros
    mac = macro_values(micro_values(w, "ROI64_T1235"), "ROI64_T1235", ["P1"], 0.60).set_index(["scope_id", "feature_id"])
    assert mac.loc[("KD1", "D007"), "measurement_status"] == "insufficient_within_subject_support"


def test_t5_zero_pupil_is_invalid_not_value():
    w = _wb([dict(pat="P1", task=5, ROI=0, nFixations=2, totalSpokenTime=0.4, speechDelay=0.8)])
    w = w.drop(columns=[c for c in w.columns if "B" in c and c.endswith("B") or c in ("maxPupLB", "minPupLB")])
    w["startPupL"], w["startPupR"], w["endPupL"], w["endPupR"] = 3.0, 0.0, 3.1, 3.2
    m = micro_values(w, "ROI64_T5").set_index("feature_id")
    assert m.loc["D001", "measurement_status"] == "invalid_measurement"


def test_duplicate_keys_raise():
    long = pd.DataFrame(dict(subject_id=["a", "a"], scope_id=["KD1", "KD1"], feature_id=["A005", "A005"],
                             value=[1.0, 2.0], measurement_status=["observed", "observed"]))
    with pytest.raises(ValueError):
        build_matrix(long, pd.DataFrame(dict(subject_id=["a"])), "scope_id")
    w = _wb([dict(pat="P1", task=1, ROI=0, nFixations=2, totalSpokenTime=0.4, speechDelay=0.8)] * 2)
    with pytest.raises(ValueError):
        micro_values(w, "ROI64_T1235")


def test_extraction_is_label_free(raw):
    """Extraction functions never receive diagnosis; shuffling labels cannot change values."""
    import inspect
    from begad_norm.feature_discovery import extract_macro, extract_micro
    for fn in (extract_macro.task_features, extract_micro.micro_values, extract_micro.macro_values):
        assert "group" not in inspect.signature(fn).parameters
    w = _wb([dict(pat="P1", task=1, ROI=0, nFixations=2, totalSpokenTime=0.4, speechDelay=0.8, group=0)])
    w2 = w.assign(group=2)
    pd.testing.assert_frame_equal(micro_values(w, "ROI64_T1235"), micro_values(w2, "ROI64_T1235"))

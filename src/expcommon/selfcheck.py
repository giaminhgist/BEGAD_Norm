"""Small deterministic self-checks for the shared experiment components.

Validates shapes, split boundaries, metric functions, the contrast builder, the
fusion weight learner and determinism of the tuning loop before any full run.
Run from the repo root:

    PYTHONPATH=src .venv/bin/python -m expcommon.selfcheck
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from expcommon.data import (LABEL_MAP, TASKS, build_problem, fold_masks,
                            load_wide, load_split, task_columns)  # noqa: E402
from expcommon.metrics import (binary_metrics, multiclass_metrics,  # noqa: E402
                               youden_threshold)
from expcommon.models import (en_grid, inner_oof, logloss, svm_grid,  # noqa: E402
                              tune_params)

ROOT = Path(__file__).resolve().parents[2]


def check(name, cond, detail=""):
    if not cond:
        print(f"FAIL {name}: {detail}")
        sys.exit(1)
    print(f"ok   {name}" + (f" ({detail})" if detail else ""))


def main():
    import yaml

    cfg = yaml.safe_load((ROOT / "src/expcommon/selfcheck.yaml").read_text())

    # 1. Data shapes, task columns, split boundaries (real data)
    X, pid, groups, feats = load_wide(cfg)
    check("wide X shape", X.shape == (157, 132), f"X.shape={X.shape}")
    check("4 tasks x 33 features", [len(task_columns(feats, t)) for t in TASKS]
          == [33] * 4)
    fold_of, held = load_split(cfg)
    assert len(held) == 4 and len(fold_of) == 153
    for problem in cfg["problems"]:
        y = build_problem(groups, problem)
        for fold in range(1, 5):
            tr, va = fold_masks(fold_of, held, groups, problem, fold, pid)
            check("train/val disjoint",
                  not (set(tr) & set(va)), f"{problem['name']} fold {fold}")
            if "AD" in problem["include_groups"]:
                check("heldout AD in every val, never in train",
                      set(held) <= {pid[i] for i in va if groups[i] == "AD"}
                      and not any(pid[i] in held for i in tr))
            else:
                check("AD excluded from CN-vs-MCI",
                      not any(groups[i] == "AD" for i in tr)
                      and not any(groups[i] == "AD" for i in va))
            exp_train = {"co_mci_ad": {114, 115}, "co_vs_mci": {105, 106},
                         "co_vs_ad": {84}}   # 75 CN + 9 AD per fold
            check("train subject counts",
                  len(tr) in exp_train[problem["name"]],
                  f"{problem['name']} fold {fold} n_train={len(tr)}")
    # problem-specific invariants from the split manifest
    prob = cfg["problems"][0]
    y = build_problem(groups, prob)
    for fold in range(1, 5):
        tr, va = fold_masks(fold_of, held, groups, prob, fold, pid)
        check("3-class AD counts", (y[va] == 2).sum() == 7 and (y[tr] == 2).sum() == 9,
              f"fold {fold}: AD val={(y[va]==2).sum()} train={(y[tr]==2).sum()}")

    # 2. Metrics on hand-checked fixtures
    y = np.array([0, 0, 1, 1])
    p1 = np.array([0.1, 0.6, 0.4, 0.9])
    m = binary_metrics(y, p1, 0.5)
    check("binary metrics", abs(m["auc"] - 0.75) < 1e-9 and
          abs(m["sen"] - 0.5) < 1e-9 and abs(m["spec"] - 0.5) < 1e-9
          and abs(m["bacc"] - 0.5) < 1e-9, str(m))
    # J = tpr - fpr peaks at 0.5 both for "p1 >= 0.9" and "p1 >= 0.4";
    # argmax keeps the first (most conservative) rule -> threshold 0.9.
    check("youden picks 0.9", abs(youden_threshold(y, p1) - 0.9) < 1e-9,
          f"thr={youden_threshold(y, p1)}")
    y3 = np.array([0, 0, 1, 1, 2, 2])
    P3 = np.array([[0.8, 0.1, 0.1]] * 2 + [[0.1, 0.8, 0.1]] * 2
                  + [[0.1, 0.1, 0.8]] * 2)
    m3 = multiclass_metrics(y3, P3, P3.argmax(1))
    check("multiclass perfect", abs(m3["acc"] - 1.0) < 1e-9 and
          abs(m3["bacc"] - 1.0) < 1e-9 and abs(m3["macro_f1"] - 1.0) < 1e-9
          and abs(m3["ovr_auc"] - 1.0) < 1e-9 and
          m3["recall_cn"] == m3["recall_mci"] == m3["recall_ad"] == 1.0)

    # 3. Contrast builder semantics (synthetic) + real-data contrast matrix
    from contrasts.run_contrasts import build_contrast_matrix
    small_cfg = {"ratio_log": ["A002"], "difference": ["D035"],
                 "pairs": {"21": ["KD2", "KD1"], "31": ["KD3", "KD1"]},
                 "expected_n": 4}
    feats_f = ["A002__KD1", "A002__KD2", "A002__KD3", "D035__KD1", "D035__KD2",
               "D035__KD3"]
    Xf = np.array([[2.0, 4.0, 8.0, 0.1, 0.3, 0.5],
                   [1.0, np.nan, 3.0, -0.2, 0.4, 0.0],
                   [0.0, 2.0, 2.0, 0.5, 0.5, 0.5]])
    cols, V = build_contrast_matrix(Xf, feats_f, small_cfg)
    check("contrast names", cols == ["A002__contrast_21", "A002__contrast_31",
                                     "D035__contrast_21", "D035__contrast_31"])
    check("ratio log + missing propagation + nonpositive guard",
          np.allclose(V[0], [np.log(2), np.log(4), 0.2, 0.4]) and
          np.isnan(V[1, 0]) and np.isclose(V[1, 1], np.log(3)) and
          np.isclose(V[1, 2], 0.6) and np.isclose(V[1, 3], 0.2) and
          np.isnan(V[2, 0]) and np.isnan(V[2, 1]) and
          V[2, 2] == 0.0 and V[2, 3] == 0.0)
    colsR, VR = build_contrast_matrix(X, feats, cfg["contrasts"])
    check("18 contrasts on real data",
          len(colsR) == 18 and VR.shape == (157, 18),
          f"{VR.shape}")
    check("ratio contrasts finite everywhere (all components positive)",
          np.isfinite(VR[:, :6]).all())
    n_nan_diff = int(np.isnan(VR[:, 6:]).sum())
    check("difference contrasts NaN only from missing components",
          0 < n_nan_diff < 40, f"n_nan={n_nan_diff}")

    # 4. Fusion weight learner semantics (synthetic)
    from fusion.run_fusion import learn_weights
    rng = np.random.default_rng(0)
    n, K = 400, 2
    yf = rng.integers(0, 2, n)
    Pt = [np.tile([0.5, 0.5], (n, 1))] * 3  # three uninformative tasks
    Pt.append(np.column_stack([1 - yf, yf]) * 0.9 + 0.05)  # near-perfect task
    w, obj, status = learn_weights(Pt, yf, lam=1e-4, n_starts=4, seed=0)
    check("learned weight favors informative task", w[3] > 0.9,
          f"w={np.round(w, 3)}")
    w0, _, _ = learn_weights(Pt, yf, lam=1e9, n_starts=4, seed=0)
    check("strong regularization pulls to uniform", np.allclose(w0, 0.25, atol=1e-3),
          f"w0={np.round(w0, 3)}")
    check("simplex respected", abs(w.sum() - 1.0) < 1e-9 and (w >= 0).all())

    # 5. Determinism of the tuning loop on a small real slice
    Xs = X[:60, :33]
    ys = build_problem(groups[:60], cfg["problems"][1])  # co_vs_mci
    ys = ys[ys >= 0]
    Xs = Xs[ys >= 0]
    from expcommon.data import fit_prep
    tr = np.arange(len(ys) // 2)
    prep = fit_prep(Xs, tr)
    Z = prep(Xs)
    p1, s1 = tune_params(Z[tr], ys[tr], "elasticnet", 33, 42, 3)
    p2, s2 = tune_params(Z[tr], ys[tr], "elasticnet", 33, 42, 3)
    check("tuning deterministic", p1 == p2 and abs(s1 - s2) < 1e-12,
          f"{p1} {s1:.6f}")
    Po, bf = inner_oof(Z[tr], ys[tr], "elasticnet", 33, 42, 7, 3)
    check("inner OOF shapes", Po.shape == (len(tr), 2) and len(bf) == 3
          and not np.isnan(Po).any())
    check("logloss bounds", 0.0 < logloss(ys[tr], Po) < 2.0,
          f"{logloss(ys[tr], Po):.4f}")

    print("selfcheck: all checks passed")


if __name__ == "__main__":
    main()

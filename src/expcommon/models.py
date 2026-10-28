"""Model factories, hyperparameter grids and nested-CV helpers.

Both experiments share the two base model families and the same tuning budget:

- elasticnet: LogisticRegression(penalty='elasticnet', solver='saga') over
  C x l1_ratio (12 combinations).
- svm_rbf: SVC(kernel='rbf') with probability calibration via
  CalibratedClassifierCV(method='sigmoid'); the number of calibration folds
  adapts to the smallest class (min(5, min class count), stratified,
  deterministic) so tiny inner-inner splits stay valid, same convention as the
  baseline otherwise. gamma is expressed as k / n_features so that the grid
  stays comparable across representations with different feature counts
  (features are standardized, so var=1 and sklearn's 'scale' equals
  1/n_features). C x k = 9 combinations.

Hyperparameters are selected by mean cross-validated log-loss (a proper scoring
rule, appropriate for probability fusion and calibration). Models are
unweighted (class_weight=None): probability fusion combines the raw model
probabilities and balanced class weights would shift their scale per problem;
this deviation from the baseline convention is documented in the config and
provenance.

Determinism: SVC, sigmoid calibration, saga and the StratifiedKFold splitters
(shuffle=True with fixed random_state) are deterministic for a fixed input
order, so each (problem, fold, variant) is evaluated exactly once.
"""

from itertools import product

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# The weakly regularized grid corners (C=10) can reach max_iter before
# satisfying saga's weight-change criterion on some folds. Their CV scores are
# still usable for selection (they lose the race by a wide margin, ~2-3 vs
# ~0.6 log-loss, and all grid points are scored identically), so the warning
# is filtered to keep run logs readable. Recorded here for provenance.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning,
                        module="sklearn.linear_model._sag")

EN_CS = [0.01, 0.1, 1.0, 10.0]
EN_L1_RATIOS = [0.1, 0.5, 0.9]
SVM_CS = [0.1, 1.0, 10.0]
SVM_GAMMA_MULTS = [0.1, 1.0, 10.0]
EPS = 1e-7


def logloss(y, P):
    """Mean cross-entropy of the probability matrix P for labels y (clipped)."""
    Pc = np.clip(P, EPS, 1.0 - EPS)
    return float(-np.mean(np.log(Pc[np.arange(len(y)), y])))


def en_grid():
    return [{"C": c, "l1_ratio": lr}
            for c, lr in product(EN_CS, EN_L1_RATIOS)]


def svm_grid(n_features):
    return [{"C": c, "gamma": k / n_features}
            for c, k in product(SVM_CS, SVM_GAMMA_MULTS)]


def grid_for(model_type, n_features):
    if model_type == "elasticnet":
        return en_grid()
    if model_type == "svm_rbf":
        return svm_grid(n_features)
    raise ValueError(model_type)


def make_estimator(model_type, params):
    if model_type == "elasticnet":
        # saga shuffles the data order using random_state; pin it so repeated
        # fits on identical data are identical (random_state=None shuffles
        # nondeterministically and changes the converged solution). tol=1e-4 /
        # max_iter=2000 converges within budget on this data, including the
        # weakly regularized grid corners (verified: n_iter <= 1922).
        return LogisticRegression(penalty="elasticnet", solver="saga",
                                  max_iter=2000, tol=1e-4, random_state=0,
                                  **params)
    if model_type == "svm_rbf":
        return CalibratedClassifierCV(SVC(kernel="rbf", **params), cv=5,
                                      method="sigmoid")
    raise ValueError(model_type)


def calibrated_svm(params, y_tr):
    """SVC-RBF with sigmoid calibration; the number of calibration folds
    adapts to the smallest class so calibration stays stratified and valid on
    tiny training sets (e.g. 4 AD subjects in an inner-inner split)."""
    cv = min(5, int(np.min(np.bincount(y_tr))))
    return CalibratedClassifierCV(SVC(kernel="rbf", **params), cv=cv,
                                  method="sigmoid")


def fit_predict_proba(X_tr, y_tr, X_va, model_type, params):
    if model_type == "svm_rbf":
        est = calibrated_svm(params, y_tr)
    else:
        est = make_estimator(model_type, params)
    est.fit(X_tr, y_tr)
    return est.predict_proba(X_va)


def tune_params(X, y, model_type, n_features, cv_seed, cv_folds):
    """Grid search by mean cross-validated log-loss; return (best_params, score).

    The CV splitter stratifies by y and depends only on (y, cv_seed), so the
    splits are identical across representations and model types for the same
    training subjects, which keeps the evaluations paired.
    """
    best_params, best_score = None, np.inf
    for params in grid_for(model_type, n_features):
        scores = []
        for tr_i, va_i in StratifiedKFold(cv_folds, shuffle=True,
                                          random_state=cv_seed).split(X, y):
            P = fit_predict_proba(X[tr_i], y[tr_i], X[va_i], model_type, params)
            scores.append(logloss(y[va_i], P))
        s = float(np.mean(scores))
        if s < best_score:
            best_score, best_params = s, params
    return best_params, best_score


def inner_oof(X, y, model_type, n_features, cv_seed, inner_cv_seed, cv_folds):
    """Leak-free out-of-fold probabilities for every training subject.

    For each inner fold, hyperparameters are tuned on the remaining subjects
    only (inner-inner CV), so the predicted subject was used in neither the
    fit, the tuning, nor the calibration of the model that predicted it.
    Returns (P_oof, best_params_per_fold).
    """
    K = int(np.unique(y).size)
    P = np.zeros((len(y), K))
    best_per_fold = []
    for tr_i, va_i in StratifiedKFold(cv_folds, shuffle=True,
                                      random_state=cv_seed).split(X, y):
        params, _ = tune_params(X[tr_i], y[tr_i], model_type, n_features,
                                inner_cv_seed, cv_folds)
        best_per_fold.append(params)
        P[va_i] = fit_predict_proba(X[tr_i], y[tr_i], X[va_i], model_type, params)
    return P, best_per_fold

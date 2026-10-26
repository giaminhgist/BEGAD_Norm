"""PyTorch port of the multimodal model in Multimodal.ipynb.

Notebook architecture (Keras functional, ported faithfully):
  - visual branch: Dense(128, relu, L2 1e-3) -> Dropout(0.2) ->
    Dense(64, relu, L2 1e-3); input = 6 gaze features + ROI.
  - speech branch: identical structure; input = 4 speech features + ROI.
  - pupil branch: Dense(128, relu, NO regularizer, notebook quirk) ->
    Dropout(0.2) -> Dense(64, relu, L2 1e-3); input = pupil features + ROI
    (7d for T5, 10d for T1235 with the 3 extra pupil columns).
  - fusion: concat(3 x 64) -> Dense(128, relu, L2) -> Dropout(0.2) ->
    Dense(64, relu, L2) -> Dense(n_classes, softmax).
  - Adam (lr 1e-3), cross-entropy, batch 32. Output units: 2 for binary
    problems, 3 for 3-class (adaptation; the notebook was 3-class only).
  - L2 is added to the loss as reg * sum(w^2) over exactly the layers that
    carried kernel_regularizer in Keras (biases and the output layer are
    unregularized), mirroring Keras more faithfully than weight decay.
  - No class weights (faithful to the notebook).
  - Early stopping: patience on validation loss, best state restored.
    Deviation from the notebook (fixed 300 epochs, checkpoint on
    validation accuracy): user-approved for CPU budget.
"""
import copy
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


class MultimodalNet(torch.nn.Module):
    def __init__(self, visual_dim, speech_dim, pupil_dim, n_classes,
                 l2=0.001, dropout=0.2):
        super().__init__()
        self.visual_net = torch.nn.Sequential(
            torch.nn.Linear(visual_dim, 128), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(128, 64),
            torch.nn.ReLU())
        self.speech_net = torch.nn.Sequential(
            torch.nn.Linear(speech_dim, 128), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(128, 64),
            torch.nn.ReLU())
        self.pupil_net = torch.nn.Sequential(
            torch.nn.Linear(pupil_dim, 128), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(128, 64),
            torch.nn.ReLU())
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(192, 128), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(128, 64),
            torch.nn.ReLU(), torch.nn.Linear(64, n_classes))
        self.l2 = l2
        # Layers with kernel_regularizer in the Keras notebook: visual and
        # speech Dense(128) + Dense(64); pupil Dense(64) ONLY; fusion
        # Dense(128) + Dense(64). Output layer and biases unregularized.
        self._l2_layers = [self.visual_net[0], self.visual_net[3],
                           self.speech_net[0], self.speech_net[3],
                           self.pupil_net[3],
                           self.fusion[0], self.fusion[3]]

    def forward(self, x_vis, x_sp, x_pup):
        v = self.visual_net(x_vis)
        s = self.speech_net(x_sp)
        p = self.pupil_net(x_pup)
        return self.fusion(torch.cat([v, s, p], dim=1))

    def l2_penalty(self):
        out = torch.zeros(())
        for layer in self._l2_layers:
            out = out + (layer.weight ** 2).sum()
        return out


def torch_branch_inputs(X, cols, branch_features, roi_column):
    """Split the scaled feature matrix into the three branch inputs, ROI
    appended as the last column of every branch (as in the notebook).
    X: (n, m) numpy array; cols: the m column names."""
    idx = {c: i for i, c in enumerate(cols)}
    out = []
    for name in ("visual", "speech", "pupil"):
        cols_sel = branch_features[name] + [roi_column]
        out.append(X[:, [idx[c] for c in cols_sel]].astype(np.float32))
    return tuple(out)


def carve_val_mask(train_mask, pat, pid_group, y, split, seed,
                   val_fraction=0.2):
    """Validation mask carved from the training rows (early-stopping set).
    grouped: ~val_fraction of training participants, stratified by group
    (fallback to unstratified sampling when a group has < 2 participants);
    participant-disjoint from the test rows.
    random: row-level StratifiedShuffleSplit over the training rows (this
    arm is deliberately leaky, so row-level validation is consistent).
    Returns a boolean mask over all rows."""
    if split == "grouped":
        train_pats = np.unique(pat[train_mask])
        rng = np.random.default_rng(seed)
        groups = np.array([pid_group[str(p)] for p in train_pats])
        uniq, counts = np.unique(groups, return_counts=True)
        if (counts < 2).any():
            k = max(1, int(round(val_fraction * len(train_pats))))
            val_pats = set(rng.choice(train_pats, size=k, replace=False))
        else:
            val_pats = set()
            for g in uniq:
                g_pats = train_pats[groups == g]
                k = max(1, int(round(val_fraction * len(g_pats))))
                val_pats |= set(rng.choice(g_pats, size=k, replace=False))
        return train_mask & np.isin(pat, list(val_pats))
    tr_idx = np.flatnonzero(train_mask)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                 random_state=seed)
    _, val_rel = next(sss.split(np.zeros(len(tr_idx)), y[tr_idx]))
    val = np.zeros(len(pat), dtype=bool)
    val[tr_idx[val_rel]] = True
    return val


def fit_torch_model(X_vis, X_sp, X_pup, y, train_mask, val_mask, test_mask,
                    seed, cfg):
    """Fit MultimodalNet on the training rows, early-stopping on the
    validation rows, predict softmax probabilities on the test rows.
    Returns dict(proba_test, fit_s, n_epochs, best_epoch, val_loss)."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    n_classes = len(np.unique(y))
    model = MultimodalNet(X_vis.shape[1], X_sp.shape[1], X_pup.shape[1],
                          n_classes, l2=cfg["l2"], dropout=cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)

    def tensors(X):
        return torch.from_numpy(np.ascontiguousarray(X))

    t_v, t_s, t_p = tensors(X_vis), tensors(X_sp), tensors(X_pup)
    t_y = torch.from_numpy(y.astype(np.int64))
    tr_idx = np.flatnonzero(train_mask & ~val_mask)
    va_idx = np.flatnonzero(val_mask)
    te_idx = np.flatnonzero(test_mask)

    def loss_fn(logits, yb):
        return (torch.nn.functional.cross_entropy(logits, yb)
                + model.l2 * model.l2_penalty())

    t0 = time.time()
    max_epochs = cfg["max_epochs"]
    patience = cfg["patience"]
    use_early_stop = len(va_idx) > 0
    if not use_early_stop:
        max_epochs = min(50, max_epochs)
    best_val = np.inf
    best_state = None
    best_epoch = 0
    waited = 0
    n_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(tr_idx))
        for start in range(0, len(order), cfg["batch_size"]):
            batch = tr_idx[order[start:start + cfg["batch_size"]]]
            optimizer.zero_grad()
            loss = loss_fn(model(t_v[batch], t_s[batch], t_p[batch]),
                           t_y[batch])
            loss.backward()
            optimizer.step()
        n_epochs = epoch
        if not use_early_stop:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(t_v[va_idx], t_s[va_idx],
                                          t_p[va_idx]), t_y[va_idx]))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(t_v[te_idx], t_s[te_idx], t_p[te_idx])
        proba = torch.softmax(logits, dim=1).numpy().astype(np.float64)
    proba = np.nan_to_num(proba, nan=0.0)
    if proba.sum(axis=1).min() == 0:  # guard degenerate rows
        proba = np.where(proba.sum(axis=1, keepdims=True) > 0, proba,
                         1.0 / n_classes)
    return {"proba_test": proba, "fit_s": time.time() - t0,
            "n_epochs": n_epochs, "best_epoch": best_epoch,
            "val_loss": best_val if use_early_stop else float("nan")}

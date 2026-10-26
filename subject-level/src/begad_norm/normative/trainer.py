"""Seeded training loop for the normative model (port of LNGModel trainer).

- Seeds BEFORE model construction (model passed as a factory), so initial
  parameters depend only on cfg.seed, never on a previous fold's RNG state.
- Per-epoch CO bank refresh (lagged by one epoch); train/val metrics are
  computed at eval() over ALL subjects of each split.
- Early stopping on validation AUC (patience, earliest epoch wins ties);
  a NaN val AUC aborts the run with its reason.
- best.pt / last.pt include the bank buffers; restoring the best checkpoint
  must reproduce its validation predictions exactly (asserted).
- Artifacts per run dir: metrics.jsonl, best.pt, last.pt, predictions.csv,
  summary.json, run_info.json.
"""
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from begad_norm.config import PROJECT_ROOT
from begad_norm.normative.metrics import compute_metrics
from begad_norm.normative.seeding import code_hash, determinism_env_info, seed_all

HEADLINE = ["acc", "auc", "sen", "spec", "f1"]


@dataclass
class NormativeConfig:
    """Run configuration; every run saves its full config to config.json."""
    experiment: str = "normative"
    problem: str = "co_vs_mci"
    model: str = "normative_mlp_deepset"
    seed: int = 42
    fold: int = 1
    epochs: int = 150
    patience: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    dropout: float = 0.3
    n_classes: int = 1
    out_root: str = ""

    def run_dir(self, create=True):
        d = (Path(self.out_root) / "runs" / self.problem / self.model /
             f"seed{self.seed}" / f"fold{self.fold}")
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def save(self, run_dir):
        (Path(run_dir) / "config.json").write_text(
            json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, run_dir):
        return cls.from_dict(
            json.loads((Path(run_dir) / "config.json").read_text()))


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=PROJECT_ROOT, check=False).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _class_weights(labels):
    """Per-class inverse frequencies (matches baseline class_weight=balanced)."""
    counts = np.bincount(labels)
    return len(labels) / (len(counts) * counts)


@torch.no_grad()
def _evaluate(model, ds):
    """One-shot eval over ALL subjects of the dataset (P <= 157)."""
    model.eval()
    prob, _ = model((ds.X, ds.mask))
    return prob.numpy()


def _probs_for_metrics(probs, n_classes):
    if n_classes == 1:
        return probs.ravel()
    return probs


def train_model(model_factory, train_ds, val_ds, cfg, device="cpu",
                run_info=None):
    """Full training loop. Returns a dict run summary.

    model_factory() is called AFTER seed_all(cfg.seed).
    """
    seed_all(cfg.seed)
    model = model_factory()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    n_classes = cfg.n_classes
    run_dir = cfg.run_dir(create=True)

    w = _class_weights(train_ds.labels)
    w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
    if n_classes == 1:
        loss_fn = nn.BCELoss(reduction="none")  # per-sample, then weighted
        train_w = torch.tensor([w[int(y)] for y in train_ds.labels],
                               dtype=torch.float32, device=device)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=w_tensor)

    info = {
        "git_commit": _git_commit(),
        "code_hash": code_hash(),
        "env": determinism_env_info(),
        "train_subject_ids": train_ds.subjects,
        "val_subject_ids": val_ds.subjects,
        "n_train": len(train_ds.subjects), "n_val": len(val_ds.subjects),
        "device": device, "status": "failed", "started": None,
        "finished": None,
    }
    info.update(run_info or {})
    overlap = set(info["train_subject_ids"]) & set(info["val_subject_ids"])
    assert not overlap, "train/val subject overlap"
    cfg.save(run_dir)
    info["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    log_path = run_dir / "metrics.jsonl"
    log_path.write_text("")  # a re-run must start clean

    best_auc, best_state, best_epoch, wait = -1.0, None, -1, 0
    best_val_metrics = None
    best_val_probs, best_val_sids, best_val_ys = None, None, None
    history = []
    stopping_reason = "max_epochs"
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        model.refresh_bank(train_ds)  # rebuild the CO latent bank (lagged)
        order = np.random.permutation(len(train_ds))
        on_loss, on_n = 0.0, 0
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            if len(idx) == 1:
                # Merge a singleton tail batch (n_train = k*batch + 1) into
                # the previous batch: BatchNorm1d in the head needs >= 2
                # samples in training mode, and no subject may be dropped.
                idx = order[i - cfg.batch_size:i + 1]
            xb = (train_ds.X[idx].to(device), train_ds.mask[idx].to(device))
            yb = torch.tensor(train_ds.labels[idx], dtype=torch.long,
                              device=device)
            opt.zero_grad()
            prob, aux = model(xb, y=yb)
            if n_classes == 1:
                loss = loss_fn(prob.view(-1), yb.float().view(-1))
                loss = (loss * train_w[idx].to(device)).mean()
            else:
                loss = loss_fn(aux["logits"], yb)
            assert torch.isfinite(loss), f"non-finite loss at epoch {epoch}"
            loss.backward()
            opt.step()
            on_loss += loss.item() * len(idx)
            on_n += len(idx)

        # epoch-end metrics at eval() over ALL subjects of each split
        tr_probs = _evaluate(model, train_ds)
        val_probs = _evaluate(model, val_ds)
        tr_metrics = compute_metrics(
            train_ds.labels, _probs_for_metrics(tr_probs, n_classes), n_classes)
        val_metrics = compute_metrics(
            val_ds.labels, _probs_for_metrics(val_probs, n_classes), n_classes)

        auc = val_metrics["auc"]
        if np.isnan(auc):
            raise RuntimeError(
                f"val AUC is NaN at epoch {epoch} "
                f"(reason: {val_metrics['auc_reason']}) — abort run")
        best_so_far = False
        if auc > best_auc:
            best_auc, best_epoch, wait = auc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_metrics, best_val_ys = val_metrics, val_ds.labels.copy()
            best_val_probs = _probs_for_metrics(val_probs, n_classes).copy()
            best_val_sids = list(val_ds.subjects)
            best_so_far = True
        else:
            wait += 1
            if wait >= cfg.patience:
                stopping_reason = "early_stopping"

        tr_loss, val_loss = float("nan"), float("nan")
        row = {
            "epoch": epoch, "lr": float(opt.param_groups[0]["lr"]),
            "elapsed_s": round(time.time() - t0, 3),
            "best_so_far": best_so_far,
            "train": {**{k: tr_metrics[k] for k in HEADLINE},
                      "loss": tr_loss},
            "train_online": {"loss": on_loss / max(on_n, 1)},
            "val": {**{k: val_metrics[k] for k in HEADLINE},
                    "loss": val_loss},
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        history.append(row)
        if stopping_reason == "early_stopping":
            break

    epochs_trained = epoch + 1

    # last checkpoint: state at the end of the final executed epoch
    torch.save({"model_state": {k: v.clone() for k, v in model.state_dict().items()},
                "epoch": epoch, "config": cfg.to_dict()},
               run_dir / "last.pt")

    # restore best + verify its predictions reproduce exactly
    model.load_state_dict(best_state)
    verify_probs = _evaluate(model, val_ds)
    verify_probs = _probs_for_metrics(verify_probs, n_classes)
    verify_metrics = compute_metrics(val_ds.labels, verify_probs, n_classes)
    assert abs(verify_metrics["auc"] - best_auc) < 1e-9 and \
        np.allclose(np.asarray(verify_probs, dtype=np.float64),
                    np.asarray(best_val_probs, dtype=np.float64)), \
        "best-checkpoint restore does not reproduce its validation predictions"
    torch.save({"model_state": best_state, "epoch": best_epoch,
                "val_auc": best_auc, "val_metrics": best_val_metrics,
                "val_subject_ids": best_val_sids,
                "val_probs": [float(p) for p in np.asarray(best_val_probs).ravel()],
                "config": cfg.to_dict()}, run_dir / "best.pt")

    if n_classes == 1:
        preds = pd.DataFrame({"subject_id": best_val_sids,
                              "label": best_val_ys,
                              "prob": np.asarray(best_val_probs).ravel()})
    else:
        preds = pd.DataFrame({"subject_id": best_val_sids,
                              "label": best_val_ys})
        for c in range(n_classes):
            preds[f"prob_{c}"] = np.asarray(best_val_probs)[:, c]
    preds.to_csv(run_dir / "predictions.csv", index=False)

    summary = {
        "run_dir": str(run_dir), "best_epoch": best_epoch,
        "epochs_trained": epochs_trained, "stopping_reason": stopping_reason,
        "best_val_metrics": best_val_metrics, "best_val_auc": best_auc,
        "train_subject_ids": info["train_subject_ids"],
        "val_subject_ids": info["val_subject_ids"],
        "history": history,
    }
    (run_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "history"},
                   indent=2, default=str))
    info["status"] = "completed"
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    return summary

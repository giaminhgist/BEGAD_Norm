"""Learned normative-deviation model for BEGAD (CO/MCI/AD).

Direct port of LNGModel src/models/proposal.py (EXP-PROP-001, mlp_deepset),
adapted to BEGAD: 25 features, 11 tasks, CO as the normative reference
group, and an n_classes head (sigmoid for binary, softmax for 3-class).

Encoder f_theta maps each task feature vector to a latent z; a CO latent
normative bank (per-task mu^z, sigma^z) is recomputed EVERY EPOCH from the
train-fold CO encodings (model.refresh_bank hook, called by the trainer);
a comparator produces per-task deviation d_{i,t}; deepset pooling (masked
mean || masked max) aggregates the T deviations into the subject embedding;
an MLP head classifies.
"""
import torch
import torch.nn as nn

EPS = 1e-6


class NormativeModel(nn.Module):
    """Encoder + CO latent bank + comparator + set pooling + head.

    comparator: 'mlp' (g_phi([z, mu, z-mu, z*mu])) — the locked proposal.
    pool: 'deepset' (mean||max) — the locked proposal.
    """

    def __init__(self, d_in=25, latent_dim=128, dev_dim=64, comparator="mlp",
                 pool="deepset", dropout=0.3, n_stim=11, n_classes=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.comparator = comparator
        self.dev_dim = dev_dim
        self.n_classes = n_classes
        self.pool = pool
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, latent_dim))
        self.comp = nn.Sequential(
            nn.Linear(latent_dim * 4, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, dev_dim))
        # CO latent normative bank (per task) — refreshed every epoch
        self.register_buffer("bank_mu", torch.zeros(n_stim, latent_dim))
        self.register_buffer("bank_sigma", torch.ones(n_stim, latent_dim))
        self.head = nn.Sequential(
            nn.Linear(self.dev_dim * 2, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes))

    # ------------------------------------------------------------------ #
    # Bank refresh hook (trainer calls it once per epoch, before training).
    # Same lagged-by-one-epoch schedule as LNGModel: the bank used during
    # epoch e's training AND for the epoch-e train/val evaluation is
    # computed at the START of epoch e from the pre-epoch-e weights. The
    # best checkpoint therefore contains exactly the bank its validation
    # metrics were computed with.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_bank(self, train_ds):
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        co = train_ds.co_indices()
        D = train_ds.X[co].to(device)
        mask = train_ds.mask[co].to(device)
        z, _ = self._encode(D, mask)
        valid = mask[..., None]
        mu = (z * valid).sum(0) / valid.sum(0).clamp(min=1)          # (T, L)
        var = ((z - mu[None]) ** 2 * valid).sum(0) / valid.sum(0).clamp(min=1)
        self.bank_mu.copy_(mu)
        self.bank_sigma.copy_(torch.sqrt(var.clamp(min=EPS)) + EPS)
        self.train(was_training)

    def _encode(self, D, mask):
        B, T, _ = D.shape
        z = self.encoder(D.view(B * T, -1)).view(B, T, self.latent_dim)
        return z, mask

    def _deviate(self, z, mask):
        B, T, L = z.shape
        mu = self.bank_mu[None].expand(B, T, L)
        sig = self.bank_sigma[None].expand(B, T, L)
        feat = torch.cat([z, mu, z - mu, z * mu], dim=-1)
        d = self.comp(feat)
        d = d * mask[..., None]
        return d

    def _pool(self, d, mask):
        """Deepset: masked mean || masked max -> (B, 2 * dev_dim)."""
        valid = mask[..., None]
        mean = (d * valid).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        maxv = (d * valid + (1 - valid) * (-1e9)).max(1).values
        return torch.cat([mean, maxv], dim=1)

    def forward(self, x, y=None):
        D, mask = x
        z, mask = self._encode(D, mask)
        d = self._deviate(z, mask)
        h = self._pool(d, mask)
        out = self.head(h)
        if self.n_classes == 1:
            prob = torch.sigmoid(out)
        else:
            prob = torch.softmax(out, dim=1)
        return prob, {"emb": h, "logits": out}

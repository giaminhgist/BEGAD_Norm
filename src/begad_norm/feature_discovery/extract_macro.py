"""Raw157 participant-level macro features (base A-IDs and dataset D-IDs).

Each function returns (value, status, n_events). Statuses follow file 02:
observed, observed_zero, insufficient_within_subject_support, unavailable,
invalid_measurement.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

from begad_norm.feature_discovery.event_contracts import (
    RAW_TASK_TO_CANONICAL, SCREEN_H, SCREEN_W, px_to_deg,
)

OBS, ZERO, INSUF, UNAV, INVAL = (
    "observed", "observed_zero", "insufficient_within_subject_support", "unavailable", "invalid_measurement")

RAW_MACRO_IDS = [
    "A002", "A003", "A005", "A006", "A007", "A021", "A023", "A024", "A025", "A026", "A027", "A028",
    "A029", "A030", "A031", "A033", "A034", "A035", "A036", "A042", "A043", "A046",
    *[f"D{i:03d}" for i in range(17, 48)],
]
RAW_CROSS_IDS = {"A076": "CROSS_KD", "D048": "CROSS_BOTH", "D049": "CROSS_KD"}


def _res(v, n, minimum=1, zero_ok=False):
    if n < minimum:
        return (np.nan, INSUF, n)
    if v is None or not np.isfinite(v):
        return (np.nan, INSUF, n)
    if zero_ok and v == 0:
        return (0.0, ZERO, n)
    return (float(v), OBS, n)


def _q(a, q):
    return float(np.quantile(a, q))


def _spearman(a, b, minimum):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = int(len(a))
    if n < minimum or np.ptp(a) == 0 or np.ptp(b) == 0:
        return (np.nan, INSUF, n)
    return (float(stats.spearmanr(a, b).statistic), OBS, n)


def _entropy(counts, k):
    c = np.asarray(counts, float)
    p = c[c > 0] / c.sum()
    h = -(p * np.log(p)).sum() / np.log(k)
    return float(max(h, 0.0))


def _consecutive(tr):
    """Pairs of gap-free transitions sharing a fixation (t1.to_id == t2.from_id)."""
    g = tr[tr["status"] == "gap_free"]
    m = g.merge(g, left_on="to_id", right_on="from_id", suffixes=("_1", "_2"))
    return m


def task_features(fix, sac, tr, rows, task_code) -> dict:
    """All RAW157 macro features for one participant-task."""
    out = {}
    f = fix[fix["valid"] & (fix["task"] == task_code)].sort_values("onset_ms")
    s = sac[sac["task"] == task_code]
    t = tr[tr["task"] == task_code] if len(tr) else tr
    if len(f) == 0:
        return {fid: (np.nan, UNAV, 0) for fid in RAW_MACRO_IDS}
    dur = f["dur_ms"].to_numpy(float) / 1000.0
    n = len(dur)
    out["A002"] = _res(dur.sum(), n)
    out["A003"] = _res(float(n), n, zero_ok=True)
    out["A005"] = _res(np.median(dur), n)
    out["A006"] = _res(_q(dur, .75) - _q(dur, .25) if n >= 3 else np.nan, n, 3, zero_ok=True)
    out["A007"] = _res(_q(dur, .9) if n >= 10 else np.nan, n, 10)
    out["A021"] = (float(len(s)), ZERO if len(s) == 0 else OBS, len(s))

    gf = t[t["status"] == "gap_free"] if len(t) else t
    m = gf[gf["matched"]] if len(gf) else gf
    amp = m["amp_deg"].to_numpy(float) if len(m) else np.array([])
    sd = m["sacc_dur_ms"].to_numpy(float) / 1000.0 if len(m) else np.array([])
    ok = np.isfinite(amp) & np.isfinite(sd) & (sd > 0)
    amp, sd = amp[ok], sd[ok]
    nm = len(amp)
    out["A023"] = _res(np.median(amp) if nm else np.nan, nm)
    out["A024"] = _res(_q(amp, .75) - _q(amp, .25) if nm >= 3 else np.nan, nm, 3, zero_ok=True)
    out["A025"] = _res(_q(amp, .9) if nm >= 10 else np.nan, nm, 10)
    out["A026"] = _res(np.median(sd) if nm else np.nan, nm)
    out["A027"] = _res(_q(sd, .75) - _q(sd, .25) if nm >= 3 else np.nan, nm, 3, zero_ok=True)
    out["A028"] = _res(np.median(amp / sd) if nm else np.nan, nm)
    out["A029"] = _res((amp < 1).mean() if nm else np.nan, nm, 1, zero_ok=True)
    out["A030"] = _res((amp > 2).mean() if nm else np.nan, nm, 1, zero_ok=True)
    gamp = gf["amp_deg"].to_numpy(float) if len(gf) else np.array([])
    gamp = gamp[np.isfinite(gamp)]
    out["A031"] = _res(gamp.sum() if len(gamp) else np.nan, len(gamp), 1, zero_ok=True)

    im = f[f["in_media"]]
    xs = im["screen_x"].to_numpy(float) / SCREEN_W
    ys = im["screen_y"].to_numpy(float) / SCREEN_H
    ni = len(xs)
    if ni >= 3:
        out["A033"] = _res(np.sqrt(np.mean((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)), ni, 3, zero_ok=True)
        pts = np.column_stack([xs, ys])
        try:
            area = ConvexHull(pts).volume if np.linalg.matrix_rank(pts - pts.mean(0)) == 2 else 0.0
        except QhullError:
            area = 0.0
        out["A034"] = _res(area, ni, 3, zero_ok=True)
    else:
        out["A033"] = out["A034"] = (np.nan, INSUF, ni)
    if ni >= 10:
        bx = np.clip((xs * 8).astype(int), 0, 7)
        by = np.clip((ys * 6).astype(int), 0, 5)
        out["A035"] = _res(_entropy(np.bincount(bx * 6 + by, minlength=48), 48), ni, 10, zero_ok=True)
    else:
        out["A035"] = (np.nan, INSUF, ni)
    gt = gf[np.isfinite(gf["from_x"]) & np.isfinite(gf["to_x"])] if len(gf) else gf
    ntr = len(gt)
    if ntr >= 30:
        def st(x, y):
            return (np.clip((x / SCREEN_W * 4).astype(int), 0, 3) * 3
                    + np.clip((y / SCREEN_H * 3).astype(int), 0, 2))
        a = st(gt["from_x"].to_numpy(float), gt["from_y"].to_numpy(float))
        b = st(gt["to_x"].to_numpy(float), gt["to_y"].to_numpy(float))
        mat = np.zeros((12, 12))
        np.add.at(mat, (a, b), 1)
        pi = mat.sum(1) / mat.sum()
        h = 0.0
        for i in range(12):
            if mat[i].sum() > 0:
                p = mat[i][mat[i] > 0] / mat[i].sum()
                h += pi[i] * -(p * np.log(p)).sum()
        out["A036"] = _res(h / np.log(12), ntr, 30, zero_ok=True)
    else:
        out["A036"] = (np.nan, INSUF, ntr)

    # pupil: all event-boundary rows in the task with both eyes valid
    pr = rows[(rows["Task"] == task_code) & (rows["ValidityLeft"] == 0) & (rows["ValidityRight"] == 0)]
    pv = ((pr["PupilLeft"] + pr["PupilRight"]) / 2.0).to_numpy(float)
    pv = pv[np.isfinite(pv) & (pv > 0)]
    npv = len(pv)
    out["A042"] = _res(_q(pv, .75) - _q(pv, .25) if npv >= 3 else np.nan, npv, 3, zero_ok=True)
    out["A043"] = _res(_q(pv, .9) - _q(pv, .1) if npv >= 10 else np.nan, npv, 10, zero_ok=True)
    out["A046"] = _res(np.median(pv) if npv else np.nan, npv)

    # ---------------- D features ----------------
    pairs = _consecutive(t) if len(t) else None
    gfd = gf.copy() if len(gf) else gf
    d_from = gfd["from_dur_ms"].to_numpy(float) / 1000 if len(gfd) else np.array([])
    d_to = gfd["to_dur_ms"].to_numpy(float) / 1000 if len(gfd) else np.array([])
    out["D017"] = _spearman(d_from, d_to, 30)
    q1, q2, q3 = (_q(dur, .25), _q(dur, .5), _q(dur, .75)) if n >= 3 else (np.nan,) * 3
    out["D018"] = _res((q3 - q1) / (q3 + q1) if n >= 3 and (q3 + q1) > 0 else np.nan, n, 3, zero_ok=True)
    out["D019"] = _res((q3 + q1 - 2 * q2) / (q3 - q1) if n >= 10 and q3 > q1 else np.nan, n, 10, zero_ok=True)
    out["D020"] = _res(np.median(np.abs(d_to - d_from)) if len(d_from) >= 10 else np.nan, len(d_from), 10,
                       zero_ok=True)
    iv = np.clip(gfd["interval_ms"].to_numpy(float), 0, None) / 1000 if len(gfd) else np.array([])
    out["D021"] = _res(np.median(iv) if len(iv) else np.nan, len(iv), 1, zero_ok=True)
    ioi = d_from + iv
    if len(ioi) >= 30 and (ioi.std() + ioi.mean()) > 0:
        out["D022"] = _res((ioi.std() - ioi.mean()) / (ioi.std() + ioi.mean()), len(ioi), 30, zero_ok=True)
    else:
        out["D022"] = (np.nan, INSUF, len(ioi))
    out["D023"] = _res((f["offset_ms"].max() - f["onset_ms"].min()) / 1000 if n >= 2 else np.nan, n, 2)
    # saccadic time fraction within gap-free transitions
    if len(gfd) >= 10:
        joined = np.unique(np.concatenate([gfd["from_id"].to_numpy(), gfd["to_id"].to_numpy()]))
        fd = f.set_index("fixation_id").loc[joined, "dur_ms"].sum() / 1000
        out["D024"] = _res(iv.sum() / (iv.sum() + fd), len(gfd), 10, zero_ok=True)
    else:
        out["D024"] = (np.nan, INSUF, len(gfd))
    if nm >= 10 and len(np.unique(amp)) >= 3:
        out["D025"] = _res(stats.theilslopes(sd, amp).slope, nm, 10, zero_ok=True)
    else:
        out["D025"] = (np.nan, INSUF, nm)
    out["D026"] = _spearman(amp, sd, 10)
    withsac = gfd[gfd["n_saccades"] >= 1] if len(gfd) else gfd
    ns = len(withsac)
    out["D027"] = _res((withsac["n_saccades"] >= 2).mean() if ns >= 10 else np.nan, ns, 10, zero_ok=True)

    # direction features (in-media displacement, amplitude >= 1 deg)
    dx = gfd["dx_px"].to_numpy(float) if len(gfd) else np.array([])
    dy = gfd["dy_px"].to_numpy(float) if len(gfd) else np.array([])
    ga = gfd["amp_deg"].to_numpy(float) if len(gfd) else np.array([])
    big = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(ga) & (ga >= 1)
    left = big & (dx < 0) & (np.abs(dx) > np.abs(dy))
    right = big & (dx > 0) & (np.abs(dx) > np.abs(dy))
    out["D028"] = _res(left.sum() / big.sum() if big.sum() >= 10 else np.nan, int(big.sum()), 10, zero_ok=True)
    mm = gfd["matched"].to_numpy(bool) if len(gfd) else np.array([], bool)
    ra = ga[right & mm]
    la = ga[left & mm]
    out["D029"] = _res(np.median(ra) if len(ra) else np.nan, len(ra))
    out["D030"] = _res(np.median(la) if len(la) else np.nan, len(la))
    if big.sum() >= 30:
        ang = np.arctan2(-dy[big], dx[big])  # screen y grows downward
        sec = ((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        out["D031"] = _res(_entropy(np.bincount(sec, minlength=8), 8), int(big.sum()), 30, zero_ok=True)
    else:
        out["D031"] = (np.nan, INSUF, int(big.sum()))
    if pairs is not None and len(pairs):
        v1 = np.column_stack([pairs["dx_px_1"], pairs["dy_px_1"]]).astype(float)
        v2 = np.column_stack([pairs["dx_px_2"], pairs["dy_px_2"]]).astype(float)
        a1 = pairs["amp_deg_1"].to_numpy(float)
        a2 = pairs["amp_deg_2"].to_numpy(float)
        okp = np.isfinite(v1).all(1) & np.isfinite(v2).all(1) & (a1 >= 1) & (a2 >= 1)
        cosang = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
        turn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        out["D032"] = _res((turn[okp] > 150).mean() if okp.sum() >= 30 else np.nan, int(okp.sum()), 30,
                           zero_ok=True)
        aa1 = pairs["amp_deg_1"].to_numpy(float)
        aa2 = pairs["amp_deg_2"].to_numpy(float)
        mid = pairs["to_dur_ms_1"].to_numpy(float) / 1000
        large = np.isfinite(aa1) & np.isfinite(aa2) & (aa1 >= 5)
        corr = large & (mid < 0.25) & (aa2 < 1)
        out["D033"] = _res(corr.sum() / large.sum() if large.sum() >= 5 else np.nan, int(large.sum()), 5,
                           zero_ok=True)
        out["D034"] = _spearman(aa1, aa2, 30)
    else:
        out["D032"] = out["D033"] = out["D034"] = (np.nan, INSUF, 0)
    out["D035"] = _spearman(d_from, ga, 30)
    out["D036"] = _spearman(ga, d_to, 30)
    dl = d_to[left]
    dr = d_to[right]
    if len(dl) >= 5 and len(dr) >= 5 and np.median(dl) > 0 and np.median(dr) > 0:
        out["D037"] = _res(np.log(np.median(dl) / np.median(dr)), min(len(dl), len(dr)), 5, zero_ok=True)
    else:
        out["D037"] = (np.nan, INSUF, int(min(len(dl), len(dr))))
    # immediate return A->B->A within 1 deg
    if pairs is not None and len(pairs):
        fxi = f.set_index("fixation_id")
        x0 = pairs["from_x_1"].to_numpy(float)
        y0 = pairs["from_y_1"].to_numpy(float)
        x2 = pairs["to_x_2"].to_numpy(float)
        y2 = pairs["to_y_2"].to_numpy(float)
        eye = fxi.loc[pairs["to_id_2"], ["dist_on", "dist_fin"]].mean(axis=1).to_numpy(float)
        a1 = pairs["amp_deg_1"].to_numpy(float)
        okr = np.isfinite(x0) & np.isfinite(x2) & np.isfinite(eye) & np.isfinite(a1) & (a1 >= 1)
        dd = px_to_deg(np.hypot(x2 - x0, y2 - y0), eye)
        out["D038"] = _res((dd[okr] < 1).mean() if okr.sum() >= 30 else np.nan, int(okr.sum()), 30, zero_ok=True)
    else:
        out["D038"] = (np.nan, INSUF, 0)
    sx, sy = im["screen_x"].to_numpy(float), im["screen_y"].to_numpy(float)
    if ni >= 10 and sy.std() > 0 and sx.std() > 0:
        out["D039"] = _res(np.log(sx.std() / sy.std()), ni, 10, zero_ok=True)
    else:
        out["D039"] = (np.nan, INSUF, ni)
    out["D040"] = _spearman(im["onset_ms"].to_numpy(float), sy, 10)
    loc = f[f["located"]]
    nl = len(loc)
    out["D041"] = _res((loc.loc[~loc["in_media"], "dur_ms"].sum() / loc["dur_ms"].sum())
                       if nl >= 10 else np.nan, nl, 10, zero_ok=True)
    pc = (f["pup_fin"] - f["pup_on"]).to_numpy(float)
    okc = np.isfinite(pc)
    out["D042"] = _res(np.median(pc[okc]) if okc.sum() >= 10 else np.nan, int(okc.sum()), 10, zero_ok=True)
    out["D043"] = _spearman(dur, pc, 20)
    pm = f[["pup_on", "pup_fin"]].mean(axis=1, skipna=True)
    pmap = dict(zip(f["fixation_id"], pm))
    if len(gfd):
        p1 = np.array([pmap.get(i, np.nan) for i in gfd["from_id"]], float)
        p2 = np.array([pmap.get(i, np.nan) for i in gfd["to_id"]], float)
        dp = np.abs(p2 - p1)
        dp = dp[np.isfinite(dp)]
    else:
        dp = np.array([])
    out["D044"] = _res(np.median(dp) if len(dp) >= 10 else np.nan, len(dp), 10, zero_ok=True)
    tt = f["onset_ms"].to_numpy(float) / 1000
    okt = np.isfinite(pm.to_numpy(float))
    if okt.sum() >= 10 and np.ptp(tt[okt]) >= 2:
        out["D045"] = _res(stats.theilslopes(pm.to_numpy(float)[okt], tt[okt]).slope, int(okt.sum()), 10,
                           zero_ok=True)
    else:
        out["D045"] = (np.nan, INSUF, int(okt.sum()))
    disp = np.concatenate([f["disp_on"].to_numpy(float), f["disp_fin"].to_numpy(float)])
    disp = disp[np.isfinite(disp)]
    out["D046"] = _res(_q(disp, .75) - _q(disp, .25) if len(disp) >= 10 else np.nan, len(disp), 10, zero_ok=True)
    dist = np.concatenate([f["dist_on"].to_numpy(float), f["dist_fin"].to_numpy(float)])
    dist = dist[np.isfinite(dist)]
    out["D047"] = _res(_q(dist, .75) - _q(dist, .25) if len(dist) >= 10 else np.nan, len(dist), 10, zero_ok=True)
    missing = set(RAW_MACRO_IDS) - set(out)
    if missing:
        raise RuntimeError(f"raw macro features not computed: {missing}")
    return out


def cross_features(per_task: dict) -> dict:
    """per_task[canonical_task][feature_id] -> (value, status, n)."""
    def val(t, fid):
        v = per_task.get(t, {}).get(fid)
        return v[0] if v and v[1] in (OBS, ZERO) else np.nan

    out = {}
    a = val("KD3", "A005") - val("KD1", "A005")
    out["A076"] = (a, OBS, 2) if np.isfinite(a) else (np.nan, UNAV, 0)
    kd = [val(t, "A005") for t in ("KD1", "KD2", "KD3")]
    r = val("READING", "A005")
    d = r - np.mean(kd) if np.all(np.isfinite(kd)) and np.isfinite(r) else np.nan
    out["D048"] = (d, OBS, 4) if np.isfinite(d) else (np.nan, UNAV, 0)
    c1, c3 = val("KD1", "A021"), val("KD3", "A021")
    lr = np.log(c3) - np.log(c1) if np.isfinite(c1) and np.isfinite(c3) and c1 > 0 and c3 > 0 else np.nan
    out["D049"] = (lr, OBS, 2) if np.isfinite(lr) else (np.nan, UNAV, 0)
    return out


def participant_macro(fix, sac, tr, rows) -> dict:
    per_task = {}
    for code, canon in RAW_TASK_TO_CANONICAL.items():
        per_task[canon] = task_features(fix, sac, tr, rows, code)
    cross = cross_features(per_task)
    return per_task, cross

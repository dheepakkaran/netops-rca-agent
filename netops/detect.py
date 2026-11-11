"""Anomaly detection: PyTorch LSTM forecaster vs a rolling z-score baseline.

Both score every router-step, flag runs of >= MIN_RUN steps above k, and group
flags into incidents. k is tuned on the dev faults only; test is scored once.
"""
from __future__ import annotations

import io
import warnings
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .sim import KPIS, STEPS, TRAIN_END

W, MIN_RUN, GAP = 30, 2, 12          # input window, debounce (steps), incident gap (2 min)
warnings.filterwarnings("ignore", category=DeprecationWarning)  # torch.ao dynamic quantization: deprecated, still works on 2.15
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
torch.manual_seed(0)
np.random.seed(0)


def normalize(X):
    mu = X[:, :TRAIN_END].mean(1, keepdims=True)
    sd = X[:, :TRAIN_END].std(1, keepdims=True) + 1e-3
    return (X - mu) / sd


class Forecaster(nn.Module):
    def __init__(self, k=len(KPIS), h=32):
        super().__init__()
        self.lstm = nn.LSTM(k, h, batch_first=True)
        self.head = nn.Linear(h, k)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])


def windows(Z, lo, hi):
    """Local normalization: each window is centred on its own mean, so the model
    learns the shape of normal behaviour, not the absolute level (the daily
    traffic curve drifts outside the 2-hour training range; FAILURES.md #2)."""
    xs, ys = [], []
    for i in range(Z.shape[0]):
        for t in range(max(lo, W), hi):
            w = Z[i, t - W:t]; c = w.mean(0)
            xs.append(w - c); ys.append(Z[i, t] - c)
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)


def train(Z, epochs=6):
    m = Forecaster()
    x, y = windows(Z, 0, TRAIN_END)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for b in range(0, len(x), 512):
            idx = perm[b:b + 512]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(x[idx]), y[idx])
            loss.backward(); opt.step()
    return m, loss.item()


@torch.no_grad()
def lstm_scores(m, Z):
    """|forecast error| / train residual std, max over KPIs → [router, step]."""
    m.eval()
    R, T, K = Z.shape
    err = np.zeros((R, T, K))
    for i in range(R):
        win = np.stack([Z[i, t - W:t] for t in range(W, T)]); c = win.mean(1)
        x = torch.tensor(win - c[:, None, :], dtype=torch.float32)
        err[i, W:] = np.abs(m(x).numpy() - (Z[i, W:] - c))
    sd = err[:, W:TRAIN_END].std(1, keepdims=True) + 1e-3
    return (err / sd).max(2)


def zscore_scores(Z, win=60):
    R, T, K = Z.shape
    s = np.zeros((R, T))
    for t in range(win, T):
        hist = Z[:, t - win:t]
        s[:, t] = (np.abs(Z[:, t] - hist.mean(1)) / (hist.std(1) + 1e-3)).max(1)
    return s


def flags(S, k):
    F = S > k
    out = np.zeros_like(F)
    for i in range(F.shape[0]):
        run = 0
        for t in range(F.shape[1]):
            run = run + 1 if F[i, t] else 0
            if run >= MIN_RUN:
                out[i, t - run + 1:t + 1] = True
    out[:, :TRAIN_END] = False                     # training period is not scored
    return out


def incidents(Fl, S, routers):
    pts = sorted((int(t), int(i)) for i, t in zip(*np.nonzero(Fl)))
    groups, cur = [], []
    for t, i in pts:
        if cur and t - cur[-1][0] > GAP:
            groups.append(cur); cur = []
        cur.append((t, i))
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        first = {}
        for t, i in g:
            first.setdefault(routers[i], t)
        out.append({"start": g[0][0], "end": g[-1][0], "routers": first,
                    "peak": {routers[i]: round(float(S[i, g[0][0]:g[-1][0] + 1].max()), 1) for i in {i for _, i in g}}})
    return out


def score(incs, faults, split):
    fs = [f for f in faults if f["split"] == split]
    lo, hi = min(f["start"] for f in fs) - 60, max(f["end"] for f in fs) + 60
    incs = [x for x in incs if lo <= x["start"] <= hi]
    # A fault counts as found only if an incident STARTS inside it: one giant
    # incident spanning several faults finds only the first (FAILURES.md #1).
    onset = lambda x, f: f["start"] - 2 <= x["start"] <= f["end"]
    tp_inc = [x for x in incs if any(onset(x, f) for f in fs)]
    found = [f for f in fs if any(onset(x, f) for x in incs)]
    delays = [min(x["start"] for x in incs if onset(x, f)) - f["start"] for f in found]
    p = len(tp_inc) / len(incs) if incs else 0.0
    r = len(found) / len(fs)
    return {"faults": len(fs), "found": len(found), "incidents": len(incs), "false_incidents": len(incs) - len(tp_inc),
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "mean_delay_s": round(10 * float(np.mean(delays)), 1) if delays else None}


def tune(S, faults, routers):
    best = None
    for k in (3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40):
        m = score(incidents(flags(S, k), S, routers), faults, "dev")
        if best is None or (m["f1"], k) > (best[1]["f1"], best[0]):
            best = (k, m)
    return best


def size_kb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); return round(buf.tell() / 1024, 1)


def main(X=None):
    truth = json.loads(Path("data/truth.json").read_text())
    routers = sorted(n["id"] for n in truth["topology"]["nodes"])
    if X is None:
        from .store import load_matrix, session
        X = load_matrix(session(), routers)
    Z = normalize(X)
    t0 = time.perf_counter(); m, loss = train(Z); t_train = time.perf_counter() - t0
    res = {"train": {"seconds": round(t_train, 1), "final_mse": round(loss, 4), "samples": int(30 * (TRAIN_END - W))}}
    t0 = time.perf_counter(); S_l = lstm_scores(m, Z); t_fp32 = time.perf_counter() - t0
    q = torch.ao.quantization.quantize_dynamic(m, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
    t0 = time.perf_counter(); S_q = lstm_scores(q, Z); t_int8 = time.perf_counter() - t0
    S_z = zscore_scores(Z)
    out_inc = {}
    for name, S in (("lstm", S_l), ("lstm_int8", S_q), ("zscore", S_z)):
        k, dev = tune(S, truth["faults"], routers)
        incs = incidents(flags(S, k), S, routers)
        res[name] = {"k": k, "dev": dev, "test": score(incs, truth["faults"], "test")}
        out_inc[name] = incs
    res["optimization"] = {"fp32_kb": size_kb(m), "int8_kb": size_kb(q), "fp32_score_s": round(t_fp32, 2), "int8_score_s": round(t_int8, 2),
                           "max_score_diff": round(float(np.abs(S_l - S_q).max()), 3)}
    Path("results").mkdir(exist_ok=True)
    Path("results/detection.json").write_text(json.dumps(res, indent=2))
    Path("results/incidents.json").write_text(json.dumps(out_inc["lstm"], indent=1))
    torch.save(m.state_dict(), "results/forecaster_fp32.pt")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

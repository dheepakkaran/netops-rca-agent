"""Seeded network telemetry simulator with labelled faults.

30 routers (2 core, 8 aggregation, 20 access), 5 KPIs every 10 s for 6 h.
12 faults in hours 2-6 spread from a root router to the routers that depend
on it; 6 decoy spikes are noise that must not be called incidents.

Realism that makes root-cause analysis non-trivial (each is in the README):
- every router reports with its own 0-2 step lag, so a child can look first;
- a BGP misconfiguration is weak on the root (flaps only) and loud on the
  children (packet loss), so "loudest" and "earliest" are both traps.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

STEP_S = 10
STEPS = 6 * 3600 // STEP_S          # 2160
TRAIN_END = 2 * 3600 // STEP_S      # 720: first 2 h are fault-free (model training)
KPIS = ["traffic_mbps", "latency_ms", "loss_pct", "cpu_pct", "bgp_flaps"]
T0 = 1793491200                      # 2026-11-01T00:00:00Z


@dataclass
class Fault:
    id: str
    type: str          # link_down | cpu_overload | bgp_misconfig
    root: str
    start: int         # step
    end: int
    affected: list[str] = field(default_factory=list)
    split: str = ""    # dev | test


@dataclass
class Decoy:
    id: str
    router: str
    kpi: str
    start: int
    end: int


def topology() -> nx.DiGraph:
    """Edges point child -> parent (the uplink a router depends on)."""
    g = nx.DiGraph()
    cores = ["core-1", "core-2"]
    aggs = [f"agg-{i}" for i in range(1, 9)]
    acc = [f"acc-{i:02d}" for i in range(1, 21)]
    for n, tier in [(c, "core") for c in cores] + [(a, "agg") for a in aggs] + [(x, "access") for x in acc]:
        g.add_node(n, tier=tier)
    for i, a in enumerate(aggs):
        g.add_edge(a, cores[i % 2])               # primary core
    for i, x in enumerate(acc):
        g.add_edge(x, aggs[i % 8])                # 20 access over 8 aggs
    return g


def dependents(g: nx.DiGraph, root: str) -> list[str]:
    """Routers whose uplink path goes through root (its subtree)."""
    return sorted(nx.ancestors(g, root))


def plan_faults(g: nx.DiGraph, rng: np.random.Generator) -> tuple[list[Fault], list[Decoy]]:
    roots = {
        "link_down": ["agg-1", "acc-05", "agg-6", "acc-13"],
        "cpu_overload": ["core-1", "agg-3", "agg-7", "core-2"],
        "bgp_misconfig": ["agg-2", "agg-5", "core-1", "agg-8"],
    }
    seq = [(t, r) for t in roots for r in roots[t]]
    rng.shuffle(seq)
    faults, t = [], TRAIN_END + 60
    for i, (typ, root) in enumerate(seq):
        dur = int(rng.integers(20, 41))
        affected = [root] + dependents(g, root)
        faults.append(Fault(f"F{i + 1:02d}", typ, root, t, t + dur, affected, "dev" if i < 4 else "test"))
        t += dur + int(rng.integers(70, 100))      # >= ~12 min of calm between faults
    decoys = []
    for j in range(6):
        f = faults[2 * j]
        nxt = faults[2 * j + 1] if 2 * j + 1 < len(faults) else None
        gap_start = f.end + 25
        gap_end = (nxt.start - 25) if nxt else STEPS - 10
        s = int(rng.integers(gap_start, max(gap_start + 1, gap_end - 4)))
        r = str(rng.choice(sorted(g.nodes)))
        decoys.append(Decoy(f"D{j + 1}", r, str(rng.choice(["cpu_pct", "traffic_mbps", "latency_ms"])), s, s + int(rng.integers(1, 3))))
    return faults, decoys


def simulate(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    g = topology()
    routers = sorted(g.nodes)
    faults, decoys = plan_faults(g, rng)
    t = np.arange(STEPS)
    diurnal = 1 + 0.25 * np.sin(2 * math.pi * (t * STEP_S) / (6 * 3600) - 1.0)
    base = {}
    for r in routers:
        tier = g.nodes[r]["tier"]
        scale = {"core": 4000, "agg": 900, "access": 180}[tier]
        base[r] = {
            "traffic_mbps": scale * diurnal * (1 + rng.normal(0, 0.04, STEPS)),
            "latency_ms": {"core": 1.2, "agg": 2.5, "access": 4.0}[tier] + rng.normal(0, 0.15, STEPS),
            "loss_pct": np.abs(rng.normal(0.02, 0.01, STEPS)),
            "cpu_pct": rng.uniform(20, 35) + 5 * diurnal + rng.normal(0, 1.5, STEPS),
            "bgp_flaps": rng.poisson(0.03, STEPS).astype(float),
        }
    m = {r: {k: v.copy() for k, v in base[r].items()} for r in routers}

    for f in faults:
        for r in f.affected:
            root = r == f.root
            d = 0 if root else int(rng.integers(1, 3))           # symptoms trail the root
            s, e = f.start + d, f.end + d
            if f.type == "link_down":
                if root:
                    m[r]["traffic_mbps"][s:e] *= 0.05
                    m[r]["loss_pct"][s:e] += rng.uniform(35, 60)
                else:
                    m[r]["loss_pct"][s:e] += rng.uniform(15, 30)
                    m[r]["latency_ms"][s:e] += rng.uniform(8, 15)
                    m[r]["traffic_mbps"][s:e] *= 0.4
            elif f.type == "cpu_overload":
                if root:
                    m[r]["cpu_pct"][s:e] = rng.uniform(94, 99, e - s)
                    m[r]["latency_ms"][s:e] += rng.uniform(6, 10)
                else:
                    m[r]["latency_ms"][s:e] += rng.uniform(3, 6)
            elif f.type == "bgp_misconfig":
                if root:                                            # weak at the root
                    m[r]["bgp_flaps"][s:e] += rng.poisson(3, e - s)
                else:                                               # loud downstream
                    m[r]["loss_pct"][s:e] += rng.uniform(8, 20)
                    m[r]["latency_ms"][s:e] += rng.uniform(2, 5)
    for dc in decoys:
        bump = {"cpu_pct": 55, "traffic_mbps": base[dc.router]["traffic_mbps"].mean() * 1.2, "latency_ms": 9}[dc.kpi]
        m[dc.router][dc.kpi][dc.start:dc.end] += bump

    lag = {r: int(rng.integers(0, 3)) for r in routers}       # per-router polling lag
    events = []
    for r in routers:
        for k in KPIS:
            np.clip(m[r][k], 0, None, out=m[r][k])
        np.clip(m[r]["cpu_pct"], 0, 100, out=m[r]["cpu_pct"])
        np.clip(m[r]["loss_pct"], 0, 100, out=m[r]["loss_pct"])
        L = lag[r]
        for step in range(STEPS):
            src = max(0, step - L)
            events.append({"ts": T0 + step * STEP_S, "step": step, "router": r,
                           **{k: round(float(m[r][k][src]), 4) for k in KPIS}})
    events.sort(key=lambda e: (e["step"], e["router"]))
    alarms = alarm_log(g, faults, decoys, lag, rng)
    return {"topology": nx.node_link_data(g, edges="links"), "faults": [asdict(f) for f in faults],
            "decoys": [asdict(d) for d in decoys], "lag": lag, "events": events, "alarms": alarms}


def alarm_log(g, faults, decoys, lag, rng) -> list[dict]:
    """Syslog-style lines: symptom alarms plus background chatter."""
    def ip():
        return f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
    out = []
    for f in faults:
        for r in f.affected:
            root = r == f.root
            s = f.start + (0 if root else 1) + lag[r]
            parent = next(iter(g.successors(r)), "core")
            if f.type == "link_down":
                msgs = [f"%LINK-3-UPDOWN: Interface ge-0/0/{rng.integers(0, 4)} to {parent}, changed state to down"] if root else \
                       [f"%QOS-4-DROP: uplink packet loss {rng.integers(15, 30)}% exceeds threshold", f"%PING-4-SLA: probe to {parent} latency {rng.integers(9, 16)} ms"]
            elif f.type == "cpu_overload":
                msgs = [f"%SYS-4-CPUHOG: process routing-daemon used {rng.integers(94, 99)}% CPU for {rng.integers(20, 60)} s"] if root else \
                       [f"%PING-4-SLA: probe to {parent} latency {rng.integers(5, 9)} ms"]
            else:
                msgs = [f"%BGP-5-ADJCHANGE: neighbor {ip()} Down - hold timer expired", f"%BGP-5-ADJCHANGE: neighbor {ip()} Up"] if root else \
                       [f"%QOS-4-DROP: uplink packet loss {rng.integers(8, 20)}% exceeds threshold"]
            for msg in msgs:
                out.append({"step": int(s + rng.integers(0, 3)), "router": r, "msg": msg, "fault": f.id})
    for d in decoys:
        msg = {"cpu_pct": f"%SYS-6-INFO: scheduled job backup-config started", "traffic_mbps": f"%SNMP-6-INFO: interface counters reset",
               "latency_ms": f"%PING-6-SLA: probe jitter {rng.integers(2, 5)} ms"}[d.kpi]
        out.append({"step": d.start, "router": d.router, "msg": msg, "fault": None})
    chatter = ["%SYS-6-LOGIN: user noc-bot logged in via ssh", "%NTP-6-SYNC: clock synchronized to {ip}", "%SNMP-6-INFO: trap sent to collector {ip}",
               "%SYS-6-CONFIG: configuration saved by noc-bot"]
    for _ in range(900):
        out.append({"step": int(rng.integers(0, STEPS)), "router": str(rng.choice(sorted(g.nodes))), "msg": str(rng.choice(chatter)).replace("{ip}", ip()), "fault": None})
    out.sort(key=lambda a: (a["step"], a["router"]))
    return out


def main() -> None:
    d = simulate()
    out = Path("data")
    out.mkdir(exist_ok=True)
    (out / "truth.json").write_text(json.dumps({k: d[k] for k in ("topology", "faults", "decoys", "lag")}, indent=1))
    with open(out / "events.jsonl", "w") as f:
        for e in d["events"]:
            f.write(json.dumps(e) + "\n")
    with open(out / "alarms.jsonl", "w") as f:
        for a in d["alarms"]:
            f.write(json.dumps(a) + "\n")
    print(f"{len(d['events']):,} KPI events · {len(d['alarms']):,} alarm lines · {len(d['faults'])} faults · {len(d['decoys'])} decoys")
    for f in d["faults"]:
        print(f"  {f['id']} {f['split']:<4} {f['type']:<14} root={f['root']:<7} steps {f['start']}-{f['end']}  affects {len(f['affected'])}")


if __name__ == "__main__":
    main()

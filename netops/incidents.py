"""Shared helpers: load truth, match detected incidents to faults."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def truth():
    t = json.loads(Path("data/truth.json").read_text())
    g = nx.node_link_graph(t["topology"], directed=True, edges="links")
    return t, g


def match(inc: dict, faults: list[dict]) -> dict | None:
    """The fault this incident's onset falls in (same rule as detection scoring)."""
    for f in faults:
        if f["start"] - 2 <= inc["start"] <= f["end"]:
            return f
    return None


def split_incidents(split: str):
    t, g = truth()
    incs = json.loads(Path("results/incidents.json").read_text())
    fs = [f for f in t["faults"] if f["split"] == split]
    lo, hi = min(f["start"] for f in fs) - 60, max(f["end"] for f in fs) + 60
    out = []
    for i, x in enumerate(incs):
        if lo <= x["start"] <= hi:
            f = match(x, fs)
            out.append({"id": f"{split}-{len(out) + 1:02d}", **x, "fault": f["id"] if f else None,
                        "true_root": f["root"] if f else None, "true_type": f["type"] if f else None})
    return out, t, g

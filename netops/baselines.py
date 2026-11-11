"""Root-cause baselines the agent has to beat."""
from __future__ import annotations

import networkx as nx


def earliest(inc: dict) -> str:
    """The router that went anomalous first (ties: alphabetical)."""
    return min(inc["routers"].items(), key=lambda kv: (kv[1], kv[0]))[0]


def topology_rule(inc: dict, g: nx.DiGraph) -> str:
    """Highest anomalous router whose own uplink is healthy; most anomalous
    dependents wins, then earliest. Classic upstream-most heuristic."""
    bad = set(inc["routers"])
    cands = [r for r in bad if not any(p in bad for p in g.successors(r))]
    def key(r):
        return (-len(nx.ancestors(g, r) & bad), inc["routers"][r], r)
    return min(cands or bad, key=key)

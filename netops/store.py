"""Cassandra schema and access. Every later step reads from here, so the
pipeline is the system of record (a file mode exists for CI only)."""
from __future__ import annotations

import os

import numpy as np

from .sim import KPIS, STEPS

KEYSPACE = "netops"
SCHEMA = [
    f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}",
    f"""CREATE TABLE IF NOT EXISTS {KEYSPACE}.kpi_raw (router text, step int, ts bigint,
        {", ".join(f"{k} double" for k in KPIS)}, PRIMARY KEY ((router), step))""",
    f"""CREATE TABLE IF NOT EXISTS {KEYSPACE}.kpi_minute (router text, minute int,
        {", ".join(f"{k}_mean double, {k}_max double" for k in KPIS)}, n int, PRIMARY KEY ((router), minute))""",
    f"""CREATE TABLE IF NOT EXISTS {KEYSPACE}.alarms (router text, step int, seq int, msg text,
        PRIMARY KEY ((router), step, seq))""",
]


def session():
    from cassandra.cluster import Cluster
    cluster = Cluster([os.environ.get("CASSANDRA_HOST", "127.0.0.1")])
    s = cluster.connect()
    for q in SCHEMA:
        s.execute(q)
    s.set_keyspace(KEYSPACE)
    return s


def load_matrix(s, routers: list[str]) -> np.ndarray:
    """kpi_raw → array [router, step, kpi]."""
    X = np.full((len(routers), STEPS, len(KPIS)), np.nan)
    q = s.prepare(f"SELECT step, {', '.join(KPIS)} FROM kpi_raw WHERE router = ?")
    for i, r in enumerate(routers):
        for row in s.execute(q, [r]):
            X[i, row.step] = [getattr(row, k) for k in KPIS]
    return X


def load_alarms(s, routers: list[str]) -> list[dict]:
    out = []
    q = s.prepare("SELECT step, seq, msg FROM alarms WHERE router = ?")
    for r in routers:
        out += [{"router": r, "step": row.step, "msg": row.msg} for row in s.execute(q, [r])]
    return sorted(out, key=lambda a: (a["step"], a["router"]))

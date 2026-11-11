"""Root-cause agent on the Claude API.

Routing: Haiku labels the alarm clusters once (checked, Sonnet fallback);
Opus makes the root-cause decision per incident with read-only tools.
Context engineering: one compact evidence pack per incident (anomalous
routers, their parents, top KPI deviation, symptom alarms in the window,
including routers whose metrics looked calm) instead of raw series.
Every call is written to results/ledger.jsonl; the run stops at the budget.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anthropic
import numpy as np

from . import rag
from .incidents import split_incidents
from .sim import KPIS, TRAIN_END

OPUS, SONNET, HAIKU = "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"
PRICE = {OPUS: (4.0, 20.0, 0.20, 5.0), SONNET: (2.0, 10.0, 0.10, 2.5), HAIKU: (0.10, 0.50, 0.01, 0.125)}  # $/MTok in, out, cache read, cache write
BUDGET = float(os.environ.get("RCA_BUDGET", "1.00"))
LEDGER = Path("results/ledger.jsonl")
MAX_TURNS = 6
FAULT_TYPES = ["link_down", "cpu_overload", "bgp_misconfig", "transient_noise"]


def load_env():
    for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
        k, _, v = line.partition("=")
        if k and v and not os.environ.get(k):
            os.environ[k] = v.strip()


def spent() -> float:
    return sum(json.loads(l)["cost_usd"] for l in LEDGER.read_text().splitlines()) if LEDGER.exists() else 0.0


def call(client, run, role, model, effort, fmt=None, **kw):
    if spent() >= BUDGET:
        raise RuntimeError(f"budget ${BUDGET:.2f} reached")
    t0 = time.perf_counter()
    oc = {"effort": effort} | ({"format": {"type": "json_schema", "schema": fmt}} if fmt else {})
    r = client.messages.create(model=model, max_tokens=8000, output_config=oc, **kw)
    u = r.usage
    cr, cw = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
    p = PRICE[model]
    row = {"run": run, "role": role, "model": model, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
           "cache_read_tokens": cr, "cache_write_tokens": cw, "stop": r.stop_reason, "seconds": round(time.perf_counter() - t0, 2),
           "cost_usd": (u.input_tokens * p[0] + u.output_tokens * p[1] + cr * p[2] + cw * p[3]) / 1e6}
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return r


# ---------- step 1: Haiku labels the alarm clusters (once) ----------
LABEL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["labels"], "properties": {"labels": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "label", "kind"],
    "properties": {"id": {"type": "integer"}, "label": {"type": "string"}, "kind": {"type": "string", "enum": ["symptom", "noise"]}}}}}}


def label_clusters(client) -> dict:
    out = Path("results/cluster_labels.json")
    if out.exists():
        return json.loads(out.read_text())
    cl = json.loads(Path("results/alarm_clusters.json").read_text())["clusters"]
    brief = [{"id": c["id"], "size": c["size"], "top_terms": c["top_terms"], "examples": c["examples"][:2]} for c in cl]
    prompt = ("Label each cluster of network syslog lines with a short name (3-6 words) and say whether it is a fault "
              "symptom or routine noise. Severity 3-5 lines are usually symptoms, severity 6 lines are informational.\n" + json.dumps(brief))
    for model in (HAIKU, SONNET):                                     # cascade: check, then one model up
        r = call(client, "labels", "labeler", model, "low", fmt=LABEL_SCHEMA, messages=[{"role": "user", "content": prompt}])
        try:
            labels = json.loads(next(b.text for b in r.content if b.type == "text"))["labels"]
        except (StopIteration, json.JSONDecodeError, KeyError):
            labels = []
        if sorted(x["id"] for x in labels) == sorted(c["id"] for c in cl) and all(x["label"].strip() for x in labels):
            res = {"model": model, "labels": {str(x["id"]): x for x in labels}}
            out.write_text(json.dumps(res, indent=1))
            return res
    raise RuntimeError("cluster labels failed the check on both models")


# ---------- step 2: evidence pack + tools ----------
class Evidence:
    def __init__(self, X, alarms, labels, cluster_of, g, routers):
        self.X, self.alarms, self.labels, self.cluster_of, self.g, self.routers = X, alarms, labels, cluster_of, g, routers
        self.mu = X[:, :TRAIN_END].mean(1); self.sd = X[:, :TRAIN_END].std(1) + 1e-3

    def z(self, r, lo, hi):
        i = self.routers.index(r)
        return (self.X[i, lo:hi] - self.mu[i]) / self.sd[i]

    def symptom_alarms(self, lo, hi, router=None):
        out = []
        for j, a in enumerate(self.alarms):
            if lo <= a["step"] <= hi and (router is None or a["router"] == router):
                lab = self.labels[str(self.cluster_of[j])]
                if lab["kind"] == "symptom" or router is not None:
                    out.append({"router": a["router"], "t": a["step"], "theme": lab["label"], "line": a["msg"]})
        return out

    def pack(self, inc):
        lo, hi = inc["start"] - 6, inc["end"] + 6
        rows = []
        for r, first in sorted(inc["routers"].items(), key=lambda kv: kv[1]):
            zz = self.z(r, lo, hi); k = int(np.abs(zz).max(0).argmax())
            rows.append({"router": r, "tier": self.g.nodes[r]["tier"], "uplink": next(iter(self.g.successors(r)), None),
                         "first_anomaly_s": 10 * (first - inc["start"]), "top_kpi": KPIS[k], "top_kpi_z": round(float(zz[:, k][np.abs(zz[:, k]).argmax()]), 1)})
        al = self.symptom_alarms(lo, hi)
        by = {}
        for a in al:
            by.setdefault(a["router"], {}).setdefault(a["theme"], 0)
            by[a["router"]][a["theme"]] += 1
        return {"incident": inc["id"], "duration_s": 10 * (inc["end"] - inc["start"] + 1), "anomalous_routers": rows,
                "symptom_alarms_by_router": by, "note": "first_anomaly_s is when monitoring saw it; routers report with a 0-20 s lag."}

    def tool(self, name, args, inc):
        lo, hi = inc["start"] - 6, inc["end"] + 6
        r = args.get("router", "")
        if name != "search_runbooks" and r not in self.routers:
            return {"error": f"unknown router {r!r}"}
        if name == "neighbors":
            return {"router": r, "tier": self.g.nodes[r]["tier"], "uplink": next(iter(self.g.successors(r)), None),
                    "direct_children": sorted(self.g.predecessors(r)), "all_dependents": len(__import__("networkx").ancestors(self.g, r))}
        if name == "kpi_window":
            zz = self.z(r, lo, hi)
            return {"router": r, "z_vs_normal": {k: {"min": round(float(zz[:, i].min()), 1), "max": round(float(zz[:, i].max()), 1)} for i, k in enumerate(KPIS)}}
        if name == "alarms":
            return {"router": r, "lines": [a["line"] for a in self.symptom_alarms(lo, hi, router=r)][:10]}
        if name == "search_runbooks":
            return {"results": rag.search(args.get("query", ""), 2)}
        return {"error": "unknown tool"}


def obj(props, req=None):
    return {"type": "object", "additionalProperties": False, "required": req or list(props), "properties": props}


TOOLS = [
    {"name": "neighbors", "description": "Topology around one router: its uplink, direct children and how many routers depend on it.", "strict": True,
     "input_schema": obj({"router": {"type": "string"}})},
    {"name": "kpi_window", "description": "How far each KPI of one router moved from normal during the incident (z-scores, min and max).", "strict": True,
     "input_schema": obj({"router": {"type": "string"}})},
    {"name": "alarms", "description": "Alarm lines one router logged during the incident window (any router, also ones whose metrics looked normal).", "strict": True,
     "input_schema": obj({"router": {"type": "string"}})},
    {"name": "search_runbooks", "description": "Semantic search over the network operations runbooks; returns the 2 closest.", "strict": True,
     "input_schema": obj({"query": {"type": "string"}})},
    {"name": "submit_diagnosis", "description": "Submit the final diagnosis. Call exactly once, as the last step.", "strict": True,
     "input_schema": obj({"root_router": {"type": "string", "description": "router id, or 'none' if this is transient noise, not a fault"},
                          "fault_type": {"type": "string", "enum": FAULT_TYPES},
                          "confidence": {"type": "number"},
                          "evidence": {"type": "array", "items": {"type": "string"}},
                          "runbook_id": {"type": "string"},
                          "next_step": {"type": "string"}})},
]
TOOLS[-1]["cache_control"] = {"type": "ephemeral"}        # system + tools are identical for every incident

SYSTEM = """You are the root-cause analyst for a network operations center. You only read data and recommend; you never change the network.
An incident is a burst of KPI anomalies across routers. Routers form a tree: access → aggregation → core; each router depends on its uplink.
Find the single router whose fault explains the incident, or answer 'none' with fault_type transient_noise if it is a short single-router blip with only informational alarms.
Rules of thumb that matter here:
- Symptoms spread to the routers behind the faulty one. The loudest or earliest-seen router is often a victim: monitoring lags 0-20 s per router.
- If several anomalous routers share one uplink, check that uplink router even if its metrics look calm: its alarm lines can show the cause (for example BGP sessions going down).
- Fault types: link_down (traffic to ~0 and loss on the root, link-down alarm), cpu_overload (CPU >90% and CPUHOG alarm on the root, latency behind it), bgp_misconfig (BGP Down/Up alarms on the root, loss behind it), transient_noise.
Use the tools only when the evidence pack is not enough; usually 1-3 calls. Cite the runbook that fits. Finish by calling submit_diagnosis."""


def diagnose(client, ev: Evidence, inc: dict, model: str, run: str) -> dict:
    msgs = [{"role": "user", "content": "Evidence pack:\n" + json.dumps(ev.pack(inc), separators=(",", ":"))}]
    calls = 0
    for turn in range(MAX_TURNS):
        r = call(client, run, "rca", model, "medium", system=[{"type": "text", "text": SYSTEM}], tools=TOOLS, messages=msgs)
        msgs.append({"role": "assistant", "content": r.content})
        uses = [b for b in r.content if b.type == "tool_use"]
        sub = next((b for b in uses if b.name == "submit_diagnosis"), None)
        if sub:
            return {**sub.input, "tool_calls": calls, "turns": turn + 1}
        if not uses:
            msgs.append({"role": "user", "content": "Call submit_diagnosis now with your answer."})
            continue
        res = []
        for b in uses:
            calls += 1
            res.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(ev.tool(b.name, b.input, inc), separators=(",", ":"))})
        msgs.append({"role": "user", "content": res})
    return {"root_router": "unanswered", "fault_type": "transient_noise", "confidence": 0, "evidence": [], "runbook_id": "", "next_step": "", "tool_calls": calls, "turns": MAX_TURNS}


def main():
    load_env()
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    model = {"opus": OPUS, "sonnet": SONNET}[sys.argv[2] if len(sys.argv) > 2 else "opus"]
    client = anthropic.Anthropic()
    from .store import load_alarms, load_matrix, session
    incs, t, g = split_incidents(split)
    routers = sorted(g.nodes)
    s = session()
    X, alarms = load_matrix(s, routers), load_alarms(s, routers)
    from .textmine import cluster
    cluster_of, _, _ = cluster(alarms)                         # same clustering as results/alarm_clusters.json (seeded)
    labels = label_clusters(client)["labels"]
    ev = Evidence(X, alarms, labels, cluster_of, g, routers)
    out = []
    for inc in incs:
        d = diagnose(client, ev, inc, model, f"{split}-{model.split('-')[1]}-{inc['id']}")
        ok_root = (d["root_router"] == inc["true_root"]) if inc["fault"] else (d["root_router"] == "none")
        ok_type = (d["fault_type"] == inc["true_type"]) if inc["fault"] else (d["fault_type"] == "transient_noise")
        out.append({"incident": inc["id"], "fault": inc["fault"], "true_root": inc["true_root"], "true_type": inc["true_type"], **d,
                    "root_correct": ok_root, "type_correct": ok_type})
        print(f"  {inc['id']} true={inc['true_root'] or 'noise':<7} → {d['root_router']:<7} {d['fault_type']:<15} {'✓' if ok_root else '✗'} type {'✓' if ok_type else '✗'} tools={d['tool_calls']} · spent ${spent():.3f}")
    Path(f"results/rca_{split}_{model.split('-')[1]}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

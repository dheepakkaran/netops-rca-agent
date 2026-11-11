import json

import numpy as np

from netops import agent, sim


class FakeEv(agent.Evidence):
    pass


def make_ev():
    g = sim.topology()
    routers = sorted(g.nodes)
    X = np.random.default_rng(0).normal(size=(30, sim.STEPS, len(sim.KPIS)))
    alarms = [{"router": "agg-2", "step": 1000, "msg": "%BGP-5-ADJCHANGE: neighbor 10.0.0.1 Down - hold timer expired"}]
    labels = {"0": {"id": 0, "label": "BGP neighbor down", "kind": "symptom"}}
    return agent.Evidence(X, alarms, labels, [0], g, routers)


def test_tools_are_read_only_and_reject_unknown_routers():
    ev = make_ev()
    inc = {"id": "t", "start": 995, "end": 1010, "routers": {"acc-02": 997}}
    before = ev.X.copy()
    assert ev.tool("neighbors", {"router": "agg-2"}, inc)["direct_children"] == ["acc-02", "acc-10", "acc-18"]
    assert "error" in ev.tool("kpi_window", {"router": "rm -rf"}, inc)
    assert ev.tool("alarms", {"router": "agg-2"}, inc)["lines"][0].startswith("%BGP")
    assert np.array_equal(before, ev.X)


def test_evidence_pack_includes_calm_router_alarms():
    ev = make_ev()
    p = ev.pack({"id": "t", "start": 995, "end": 1010, "routers": {"acc-02": 997}})
    assert "agg-2" in p["symptom_alarms_by_router"], "alarms on a router whose KPIs looked calm must reach the agent"
    assert len(json.dumps(p)) < 4000, "evidence pack must stay compact"


def test_tool_schemas_are_strict():
    for t in agent.TOOLS:
        s = t["input_schema"]
        assert t["strict"] and s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    assert agent.TOOLS[-1]["name"] == "submit_diagnosis"

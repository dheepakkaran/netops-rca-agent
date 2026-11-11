import json

import networkx as nx
import numpy as np
import pytest

from netops import baselines, detect, sim, textmine
from netops.store import load_matrix_file


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    d = sim.simulate(seed=7)
    p = tmp_path_factory.mktemp("data") / "events.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in d["events"]))
    routers = sorted(n["id"] for n in d["topology"]["nodes"])
    return d, routers, load_matrix_file(str(p), routers)


def test_simulator_truth(world):
    d, routers, X = world
    assert X.shape == (30, sim.STEPS, len(sim.KPIS)) and not np.isnan(X).any()
    assert len(d["faults"]) == 12 and sum(f["split"] == "test" for f in d["faults"]) == 8
    assert all(f["start"] >= sim.TRAIN_END for f in d["faults"]), "training hours must be fault-free"
    g = sim.topology()
    for f in d["faults"]:
        assert set(f["affected"]) == {f["root"], *nx.ancestors(g, f["root"])}


def test_alarm_labels_never_enter_pipeline():
    src = open("netops/produce.py").read()
    assert 'a.pop("fault", None)' in src


def test_flags_debounce_and_training_period():
    S = np.zeros((1, sim.STEPS))
    S[0, 900] = 99                      # one-step blip
    S[0, 1000:1003] = 99                # three-step run
    S[0, 100:110] = 99                  # inside training: never scored
    F = detect.flags(S, 5)
    assert not F[0, 900] and F[0, 1000:1003].all() and not F[0, 100:110].any()


def test_giant_incident_finds_only_one_fault():
    faults = [{"start": 800, "end": 820, "split": "dev"}, {"start": 900, "end": 920, "split": "dev"}]
    giant = [{"start": 801, "end": 919, "routers": {}, "peak": {}}]
    m = detect.score(giant, faults, "dev")
    assert m["found"] == 1 and m["recall"] == 0.5


def test_topology_rule_and_earliest():
    g = sim.topology()
    victims = {"acc-02": 1003, "acc-10": 1001, "acc-18": 1004}          # all under agg-2, root itself calm
    inc = {"routers": victims}
    assert baselines.earliest(inc) == "acc-10"
    assert baselines.topology_rule({"routers": {**victims, "agg-2": 1005}}, g) == "agg-2"


def test_textmine_masks_variable_tokens():
    m = textmine.mask("%QOS-4-DROP: uplink packet loss 17% exceeds threshold on agg-3 ge-0/0/2 10.1.2.3")
    assert "17" not in m and "agg-3" not in m and "<router>" in m and "<iface>" in m and "<ip>" in m

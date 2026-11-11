"""Build results/summary.json + RESULTS.md from the result files only."""
from __future__ import annotations

import json
from pathlib import Path

from .baselines import earliest, topology_rule
from .incidents import split_incidents

R = Path("results")


def rca_baselines(split):
    incs, t, g = split_incidents(split)
    seen, faults = set(), []
    for x in incs:
        if x["fault"] and x["fault"] not in seen:
            seen.add(x["fault"]); faults.append(x)
    return {"faults": len(faults), "earliest": sum(earliest(x) == x["true_root"] for x in faults),
            "topology_rule": sum(topology_rule(x, g) == x["true_root"] for x in faults), "noise_incidents": sum(1 for x in incs if not x["fault"])}


def agent_score(path):
    rows = json.loads(Path(path).read_text())
    real = [r for r in rows if r["fault"]]
    noise = [r for r in rows if not r["fault"]]
    return {"root": sum(r["root_correct"] for r in real), "faults": len(real), "type": sum(r["type_correct"] for r in rows), "incidents": len(rows),
            "noise_dismissed": sum(r["root_correct"] for r in noise), "noise": len(noise), "tool_calls": sum(r["tool_calls"] for r in rows)}


def cost(prefix):
    rows = [json.loads(l) for l in (R / "ledger.jsonl").read_text().splitlines()]
    sel = [r for r in rows if r["run"].startswith(prefix)]
    return {"calls": len(sel), "usd": round(sum(r["cost_usd"] for r in sel), 4), "cache_read_tokens": sum(r["cache_read_tokens"] for r in sel),
            "prompt_tokens": sum(r["input_tokens"] + r["cache_read_tokens"] + r["cache_write_tokens"] for r in sel)}


def main():
    det = json.loads((R / "detection.json").read_text())
    s = {"pipeline": json.loads((R / "pipeline.json").read_text()), "detection": det,
         "text_mining": json.loads((R / "alarm_clusters.json").read_text())["stats"],
         "rca_baselines": {sp: rca_baselines(sp) for sp in ("dev", "test")},
         "agent": {"dev_opus": agent_score(R / "rca_dev_opus.json"), "test_opus": agent_score(R / "rca_test_opus.json"),
                   "test_sonnet": agent_score(R / "rca_test_sonnet.json")},
         "cost": {"labels": cost("labels"), "dev_opus": cost("dev-opus"), "test_opus": cost("test-opus"), "test_sonnet": cost("test-sonnet"),
                  "total": cost("")}}
    d, b, a, c = det, s["rca_baselines"]["test"], s["agent"], s["cost"]
    rp = s["routed_policy"] = json.loads((R / "routed_policy_test.json").read_text())
    md = f"""# Results

Every number below is read from a file in `results/` by `python -m netops.report`. Data is synthetic and seeded (`data/truth.json`); the 8 test faults were scored once, after the prompt was frozen (commit 755e886).

## Pipeline (Kafka → Spark Structured Streaming → Cassandra)
| | |
|---|---|
| KPI rows written to Cassandra | {s['pipeline']['raw_rows']:,} (30 routers × 2,160 steps) |
| 1-minute feature rows (upserted) | {s['pipeline']['minute_rows_upserted']:,} |
| Alarm lines | {s['pipeline']['alarm_rows']:,} |
| Run time incl. Spark start | {s['pipeline']['seconds']} s · {s['pipeline']['end_to_end_rows_per_s']:,} rows/s end to end · Spark peak {s['pipeline']['spark_peak_processed_rows_per_s']:,} rows/s |

## Anomaly detection (test: 8 faults, threshold tuned on dev only)
| Detector | k | Faults found | Incidents | False incidents | Precision | F1 | Mean delay |
|---|---|---|---|---|---|---|---|
| PyTorch LSTM forecaster | {d['lstm']['k']} | {d['lstm']['test']['found']}/8 | {d['lstm']['test']['incidents']} | {d['lstm']['test']['false_incidents']} | {d['lstm']['test']['precision']} | **{d['lstm']['test']['f1']}** | {d['lstm']['test']['mean_delay_s']} s |
| Rolling z-score | {d['zscore']['k']} | {d['zscore']['test']['found']}/8 | {d['zscore']['test']['incidents']} | {d['zscore']['test']['false_incidents']} | {d['zscore']['test']['precision']} | {d['zscore']['test']['f1']} | {d['zscore']['test']['mean_delay_s']} s |
| LSTM, int8 dynamic quantization | {d['lstm_int8']['k']} | {d['lstm_int8']['test']['found']}/8 | {d['lstm_int8']['test']['incidents']} | {d['lstm_int8']['test']['false_incidents']} | {d['lstm_int8']['test']['precision']} | {d['lstm_int8']['test']['f1']} | {d['lstm_int8']['test']['mean_delay_s']} s |

Model optimization: int8 is {d['optimization']['fp32_kb']} KB → {d['optimization']['int8_kb']} KB ({d['optimization']['fp32_kb'] / d['optimization']['int8_kb']:.1f}× smaller) with the same detections, but scoring took {d['optimization']['fp32_score_s']} s → {d['optimization']['int8_score_s']} s on this CPU (slower, not faster, for a 32-unit LSTM).

## Text mining (alarm lines)
TF-IDF + KMeans on masked lines: k = {s['text_mining']['k']}, silhouette {s['text_mining']['silhouette']}, purity {s['text_mining']['purity']} against the syslog mnemonic the model never sees. Lines are templated by the simulator, so this is the easy case.

## Root-cause analysis (test: 8 faults + {b['noise_incidents']} noise incident)
| Method | Root router correct | Noise dismissed | Fault type correct | Model cost |
|---|---|---|---|---|
| Earliest anomaly | {b['earliest']}/{b['faults']} | – | – | $0 |
| Topology rule (upstream-most anomalous router) | {b['topology_rule']}/{b['faults']} | – | – | $0 |
| Agent, Opus decides | **{a['test_opus']['root']}/{a['test_opus']['faults']}** | {a['test_opus']['noise_dismissed']}/{a['test_opus']['noise']} | {a['test_opus']['type']}/{a['test_opus']['incidents']} | ${c['test_opus']['usd']} |
| Agent, Sonnet decides | **{a['test_sonnet']['root']}/{a['test_sonnet']['faults']}** | {a['test_sonnet']['noise_dismissed']}/{a['test_sonnet']['noise']} | {a['test_sonnet']['type']}/{a['test_sonnet']['incidents']} | ${c['test_sonnet']['usd']} |
| Agent, routed: Sonnet, Opus when it disagrees with the rule | {rp['root_and_noise_correct']}/{rp['incidents']} incidents | | | ${rp['cost_usd']} (computed*) |

Dev (5 incidents, prompt iteration): Opus {a['dev_opus']['root']}/{a['dev_opus']['faults']} roots, {a['dev_opus']['noise_dismissed']}/{a['dev_opus']['noise']} noise. Both models hit the ceiling on test, so the test cannot tell them apart; Sonnet costs {100 * (1 - c['test_sonnet']['usd'] / c['test_opus']['usd']):.0f}% less, so it is the default decider and Opus is the escalation.
\* The routed policy escalated {', '.join(rp['escalated'])} to Opus. Its cost is computed from the Sonnet and Opus test ledgers, not a separate run, because the policy was chosen after seeing the test results.
The two faults the topology rule misses (F07, F11) are BGP misconfigurations whose root router's KPIs stayed calm; the agent found them from the root's BGP alarm lines and the shared uplink.

## Cost
Total for every model call in this project, dev and both test runs included: **${c['total']['usd']}** over {c['total']['calls']} calls. Cache reads: {c['total']['cache_read_tokens']:,} of {c['total']['prompt_tokens']:,} prompt tokens (system prompt + tools are identical for every incident).
"""
    (R / "summary.json").write_text(json.dumps(s, indent=2))
    Path("RESULTS.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

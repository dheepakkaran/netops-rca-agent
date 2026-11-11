# netops-rca-agent

Network telemetry → anomaly detection → an AI agent that names the root-cause router. A seeded simulator streams router KPIs and syslog alarms through **Kafka → Spark Structured Streaming → Cassandra**; a **PyTorch LSTM forecaster** flags anomalies and groups them into incidents; **TF-IDF + KMeans** mines the alarm text; and a **Claude agent** with read-only tools (topology, KPI windows, alarm lines, **Chroma** runbook search) picks the root cause, the fault type and the runbook. It recommends; it never changes the network.

**Project page:** [dheepakkaran.github.io/netops-rca-agent](https://dheepakkaran.github.io/netops-rca-agent/) · **All numbers:** [RESULTS.md](RESULTS.md) · **What broke:** [FAILURES.md](FAILURES.md)

## Results (synthetic network, test set scored once)
| | |
|---|---|
| Anomaly detection, 8 test faults | LSTM forecaster F1 **0.941** (8/8 found, 1 false incident) vs rolling z-score 0.800 (5 false), ~14 s to detect |
| Root cause, 8 faults + 1 noise incident | agent **8/8** roots and 1/1 noise dismissed, vs topology rule 6/8 and "earliest anomaly" 5/8 |
| Model routing for the decision | Sonnet matched Opus (9/9) at **60%** lower cost; routed policy (Sonnet, Opus when it disagrees with the rule) 9/9 at $0.130 vs $0.223 for Opus alone (computed from the ledgers) |
| Pipeline | 64,800 KPI rows + 10,800 one-minute features + 986 alarm lines into Cassandra in 54 s |
| Model optimization | int8 dynamic quantization: 2.4× smaller, same detections, slower on CPU (reported, not hidden) |
| Total model spend | **$0.48** over 50 calls, dev and both test runs included |

The simulator makes root-cause analysis hard on purpose: each router reports with its own 0–20 s lag (so a victim can look first), and a BGP misconfiguration is quiet at its root and loud at its victims. Limits: synthetic data, templated alarm text, a tree topology, 8 test faults.

## How the agent spends tokens
- **Routing:** Haiku labels the 12 alarm clusters once (checked by code, Sonnet fallback); the root-cause decision runs on Sonnet and goes to Opus only when Sonnet disagrees with a deterministic baseline.
- **Context packing:** each incident becomes one compact evidence pack (anomalous routers in seen-order, their uplinks, top KPI deviation, symptom alarms from every router in the window). The raw series stay in Cassandra; tools fetch only what is asked.
- **Prompt caching:** the system prompt and tools are identical for every incident: 37.5% of prompt tokens were cache reads.
- **Strict tools:** every tool has a strict schema; the answer is a `submit_diagnosis` call (root router or `none`, fault type, evidence, runbook, next step).

## Run
```
pixi install                      # Java 17 (no sudo)
uv sync                           # Python deps (PyTorch CPU)
scripts/services.sh up            # Kafka (KRaft) + Cassandra from .services/ (see scripts/services.sh for the downloads)
uv run python -m netops.sim       # data/: events, alarms, truth
uv run python -m netops.produce   # → Kafka topics kpi, alarms
uv run python -m netops.stream    # Spark: Kafka → Cassandra
uv run python -m netops.detect    # LSTM vs z-score, int8, incidents
uv run python -m netops.textmine  # alarm clusters
uv run python -m netops.agent test routed   # .env: ANTHROPIC_API_KEY · also: opus | sonnet
uv run python -m netops.report    # RESULTS.md
```
`python -m netops.detect --file` runs detection without Kafka/Cassandra (used in CI).

## Layout
| Path | What |
|---|---|
| `netops/sim.py` | seeded topology, KPIs, faults, decoys, alarm lines |
| `netops/produce.py`, `netops/stream.py`, `netops/store.py` | Kafka producer, Spark job, Cassandra schema + reads |
| `netops/detect.py` | LSTM forecaster, z-score baseline, int8, incidents, scoring |
| `netops/textmine.py` | TF-IDF + KMeans on masked alarm lines |
| `netops/baselines.py` | earliest-anomaly and topology-rule root cause |
| `netops/agent.py`, `netops/rag.py`, `runbooks/` | Claude agent, tools, Chroma runbook search |
| `results/` | every number: `ledger.jsonl` (each model call), detection, RCA runs, summary |

# Results

Every number below is read from a file in `results/` by `python -m netops.report`. Data is synthetic and seeded (`data/truth.json`); the 8 test faults were scored once, after the prompt was frozen (commit 755e886).

## Pipeline (Kafka → Spark Structured Streaming → Cassandra)
| | |
|---|---|
| KPI rows written to Cassandra | 64,800 (30 routers × 2,160 steps) |
| 1-minute feature rows (upserted) | 10,800 |
| Alarm lines | 986 |
| Run time incl. Spark start | 54.2 s · 1,196 rows/s end to end · Spark peak 3,087 rows/s |

## Anomaly detection (test: 8 faults, threshold tuned on dev only)
| Detector | k | Faults found | Incidents | False incidents | Precision | F1 | Mean delay |
|---|---|---|---|---|---|---|---|
| PyTorch LSTM forecaster | 30 | 8/8 | 9 | 1 | 0.889 | **0.941** | 13.8 s |
| Rolling z-score | 6 | 8/8 | 15 | 5 | 0.667 | 0.8 | 13.8 s |
| LSTM, int8 dynamic quantization | 30 | 8/8 | 9 | 1 | 0.889 | 0.941 | 13.8 s |

Model optimization: int8 is 22.8 KB → 9.5 KB (2.4× smaller) with the same detections, but scoring took 0.84 s → 1.41 s on this CPU (slower, not faster, for a 32-unit LSTM).

## Text mining (alarm lines)
TF-IDF + KMeans on masked lines: k = 12, silhouette 0.998, purity 0.999 against the syslog mnemonic the model never sees. Lines are templated by the simulator, so this is the easy case.

## Root-cause analysis (test: 8 faults + 1 noise incident)
| Method | Root router correct | Noise dismissed | Fault type correct | Model cost |
|---|---|---|---|---|
| Earliest anomaly | 5/8 | – | – | $0 |
| Topology rule (upstream-most anomalous router) | 6/8 | – | – | $0 |
| Agent, Opus decides | **8/8** | 1/1 | 9/9 | $0.2234 |
| Agent, Sonnet decides | **8/8** | 1/1 | 9/9 | $0.0887 |
| Agent, routed: Sonnet, Opus when it disagrees with the rule | 9/9 incidents | | | $0.1303 (computed*) |

Dev (5 incidents, prompt iteration): Opus 4/4 roots, 1/1 noise. Both models hit the ceiling on test, so the test cannot tell them apart; Sonnet costs 60% less, so it is the default decider and Opus is the escalation.
\* The routed policy escalated test-03, test-08 to Opus. Its cost is computed from the Sonnet and Opus test ledgers, not a separate run, because the policy was chosen after seeing the test results.
The two faults the topology rule misses (F07, F11) are BGP misconfigurations whose root router's KPIs stayed calm; the agent found them from the root's BGP alarm lines and the shared uplink.

## Cost
Total for every model call in this project, dev and both test runs included: **$0.4759** over 50 calls. Cache reads: 42,018 of 111,934 prompt tokens (system prompt + tools are identical for every incident).

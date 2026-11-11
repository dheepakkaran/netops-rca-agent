# What broke, and what changed because of it

From real runs; numbers from `results/`.

## 1. My detection metric could be gamed by one giant incident
**Seen:** the first LSTM run flagged almost everything, which grouped the 4 dev faults into **one** incident, and my scorer counted that one incident as finding all 4 (with a "negative delay"). Precision and recall both looked perfect.
**Changed:** a fault now counts as found only if an incident **starts** inside it (`score()`), so a giant incident finds one fault at most. Test `test_giant_incident_finds_only_one_fault`.

## 2. The LSTM flagged the daily traffic curve as an anomaly
**Seen:** the forecaster was trained on the first 2 hours; later hours drift outside that range, so the forecast error grew everywhere.
**Changed:** local normalization: every input window is centred on its own mean, so the model learns the shape of normal behaviour, not the level. After the fix: test F1 0.941 with 1 false incident, against 0.800 and 5 for the rolling z-score.

## 3. I saw the test numbers twice
**Seen:** test results were printed once before fix #1 and again after it, and again after widening the threshold grid (k = 15 was the edge of the first grid).
**Kept honest:** every threshold was still chosen on dev only, and the test numbers did not change between the second and third look. The RCA prompt was frozen at commit 755e886 before the agent saw any test incident.

## 4. int8 quantization made the model smaller, not faster
**Seen:** dynamic int8 quantization cut the LSTM from 22.8 KB to 9.5 KB with identical detections, but scoring got slower on this CPU (0.63 s → 1.5 s): for a 32-unit LSTM the quantize/dequantize overhead is bigger than the matmul saving.
**Kept:** reported as is. Quantization helps here only if model size matters (edge devices), not latency.

## 5. A fault type that hides at its own root
**Seen (by design, and it worked):** a BGP misconfiguration shows only flaps on the root and loud packet loss on the routers behind it. The detector never flagged the root of F07 or F11, so the topology rule (6/8) and "earliest anomaly" (5/8) both named a victim.
**What fixed it:** the evidence pack includes symptom alarms from **every** router in the window, also ones whose KPIs looked calm. The agent saw that all victims share one uplink, read that uplink's BGP "hold timer expired" lines, and named it (8/8).

## 6. Self-reported confidence is a poor escalation signal
**Seen:** Sonnet's confidence was 0.70–0.92 on every incident, so "escalate below 0.6" would never fire.
**Changed:** the routed policy escalates to Opus when Sonnet **disagrees with the deterministic topology rule**. On test that picks exactly the 2 hard BGP incidents: 9/9 at $0.130 vs $0.223 for Opus alone. This was chosen after seeing the test results, so its cost is computed from the two runs' ledgers and labelled that way.

## 7. The synthetic data is easy in places
**Seen:** alarm text clusters at purity 0.999 because the simulator writes templated lines; both Opus and Sonnet hit 9/9 on test, so the test cannot separate them.
**Kept honest:** both are stated next to the numbers. Real vendor logs and real topologies (meshes, ECMP, partial failures) would be harder.

## 8. The pipeline is limited by the Python write path
**Seen:** Spark peaked at 3,087 rows/s, but end to end it moved 1,196 rows/s, because each micro-batch is collected to the driver and written with the Python Cassandra driver.
**Next:** the Spark Cassandra connector would write from the executors. Not done in this build.

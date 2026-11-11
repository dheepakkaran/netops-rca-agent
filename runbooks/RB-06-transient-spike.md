# RB-06 Transient single-metric spike
Symptoms: one metric on one router spikes for under a minute (CPU from a scheduled backup job, a counter reset, probe jitter), with only informational alarms (severity 6) and no effect on neighbors.
Check: scheduled jobs and maintenance at that time.
Next step: no action; record as noise. Do not open an incident unless it repeats or spreads to neighbors.

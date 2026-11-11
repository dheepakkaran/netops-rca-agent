"""Replay the simulated telemetry and alarm lines into Kafka topics."""
from __future__ import annotations

import json
import time

from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic


def main() -> None:
    admin = KafkaAdminClient(bootstrap_servers="localhost:9092")
    existing = set(admin.list_topics())
    new = [NewTopic(t, num_partitions=3, replication_factor=1) for t in ("kpi", "alarms") if t not in existing]
    if new:
        admin.create_topics(new)
    p = KafkaProducer(bootstrap_servers="localhost:9092", linger_ms=20, batch_size=256 * 1024,
                      key_serializer=str.encode, value_serializer=lambda v: json.dumps(v).encode())
    t0, n = time.perf_counter(), 0
    with open("data/events.jsonl") as f:
        for line in f:
            e = json.loads(line)
            p.send("kpi", key=e["router"], value=e)       # key = router keeps each router's order per partition
            n += 1
    with open("data/alarms.jsonl") as f:
        for i, line in enumerate(f):
            a = json.loads(line)
            a.pop("fault", None)                           # the label never enters the pipeline
            a["seq"] = i
            p.send("alarms", key=a["router"], value=a)
            n += 1
    p.flush()
    dt = time.perf_counter() - t0
    print(f"produced {n:,} messages in {dt:.1f} s = {n / dt:,.0f} msg/s")


if __name__ == "__main__":
    main()

"""Spark Structured Streaming: Kafka → raw rows + 1-minute features → Cassandra.

Each micro-batch is upserted (idempotent: primary key = router + step/minute),
so a restarted query or a re-run never duplicates rows."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

from .sim import KPIS
from .store import session

KPI_SCHEMA = StructType([StructField("ts", LongType()), StructField("step", IntegerType()), StructField("router", StringType())]
                        + [StructField(k, DoubleType()) for k in KPIS])
ALARM_SCHEMA = StructType([StructField("step", IntegerType()), StructField("router", StringType()),
                           StructField("msg", StringType()), StructField("seq", IntegerType())])


def spark() -> SparkSession:
    root = Path(__file__).resolve().parent.parent
    return (SparkSession.builder.appName("netops-stream").master("local[4]")
            .config("spark.jars.packages", f"org.apache.spark:spark-sql-kafka-0-10_2.13:{__import__('pyspark').__version__}")
            .config("spark.jars.ivy", str(root / ".data/ivy"))
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "1g")
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def main() -> None:
    s = session()
    ins_raw = s.prepare(f"INSERT INTO kpi_raw (router, step, ts, {', '.join(KPIS)}) VALUES (?, ?, ?, {', '.join('?' * len(KPIS))})")
    cols_min = [f"{k}_{a}" for k in KPIS for a in ("mean", "max")]
    ins_min = s.prepare(f"INSERT INTO kpi_minute (router, minute, {', '.join(cols_min)}, n) VALUES (?, ?, {', '.join('?' * len(cols_min))}, ?)")
    ins_alarm = s.prepare("INSERT INTO alarms (router, step, seq, msg) VALUES (?, ?, ?, ?)")
    from cassandra.concurrent import execute_concurrent_with_args

    sp = spark()
    sp.sparkContext.setLogLevel("WARN")
    src = lambda topic: (sp.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9092")
                         .option("subscribe", topic).option("startingOffsets", "earliest").load())
    kpi = src("kpi").select(F.from_json(F.col("value").cast("string"), KPI_SCHEMA).alias("e")).select("e.*")
    alarms = src("alarms").select(F.from_json(F.col("value").cast("string"), ALARM_SCHEMA).alias("a")).select("a.*")
    minute = (kpi.withColumn("minute", (F.col("step") / 6).cast("int"))
              .groupBy("router", "minute")
              .agg(*[f(F.col(k)).alias(f"{k}_{n}") for k in KPIS for f, n in ((F.avg, "mean"), (F.max, "max"))], F.count("*").alias("n")))

    stats = {"raw": 0, "minute": 0, "alarms": 0}

    def write_raw(df, _):
        rows = [(r.router, r.step, r.ts, *[r[k] for k in KPIS]) for r in df.collect()]
        execute_concurrent_with_args(s, ins_raw, rows, concurrency=64)
        stats["raw"] += len(rows)

    def write_min(df, _):
        rows = [(r.router, r.minute, *[r[c] for c in cols_min], r.n) for r in df.collect()]
        execute_concurrent_with_args(s, ins_min, rows, concurrency=64)
        stats["minute"] += len(rows)

    def write_alarms(df, _):
        rows = [(r.router, r.step, r.seq, r.msg) for r in df.collect()]
        execute_concurrent_with_args(s, ins_alarm, rows, concurrency=64)
        stats["alarms"] += len(rows)

    ck = Path(".data/checkpoints")
    t0 = time.perf_counter()
    qs = [kpi.writeStream.foreachBatch(write_raw).option("checkpointLocation", str(ck / "raw")).trigger(availableNow=True).start(),
          minute.writeStream.outputMode("update").foreachBatch(write_min).option("checkpointLocation", str(ck / "minute")).trigger(availableNow=True).start(),
          alarms.writeStream.foreachBatch(write_alarms).option("checkpointLocation", str(ck / "alarms")).trigger(availableNow=True).start()]
    for q in qs:
        q.awaitTermination()
    dt = time.perf_counter() - t0
    progress = [p for q in qs for p in q.recentProgress]
    rps = [p["processedRowsPerSecond"] for p in progress if p.get("processedRowsPerSecond")]
    out = {"seconds": round(dt, 1), "raw_rows": stats["raw"], "minute_rows_upserted": stats["minute"], "alarm_rows": stats["alarms"],
           "end_to_end_rows_per_s": round(stats["raw"] / dt), "spark_peak_processed_rows_per_s": round(max(rps)) if rps else None}
    Path("results").mkdir(exist_ok=True)
    Path("results/pipeline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    sp.stop()


if __name__ == "__main__":
    os.environ.setdefault("PYSPARK_PYTHON", str(Path(__file__).resolve().parent.parent / ".venv/bin/python"))
    main()

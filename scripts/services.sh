#!/usr/bin/env bash
# Start / stop / check Kafka (KRaft, single node) and Cassandra without Docker.
# Java 17 comes from the pixi env; all data stays in .data/ inside the project.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export JAVA_HOME="$ROOT/.pixi/envs/default/lib/jvm" PATH="$ROOT/.pixi/envs/default/lib/jvm/bin:$PATH"
K="$ROOT/.services/kafka_2.13-4.4.0"; C="$ROOT/.services/apache-cassandra-5.0.9"; D="$ROOT/.data"
mkdir -p "$D/kafka" "$D/cassandra" "$D/logs"

setup() {
  sed -i "s#^log.dirs=.*#log.dirs=$D/kafka#" "$K/config/server.properties"
  if [ ! -f "$D/kafka/meta.properties" ]; then
    "$K/bin/kafka-storage.sh" format --standalone -t "$("$K/bin/kafka-storage.sh" random-uuid)" -c "$K/config/server.properties" >/dev/null
  fi
  local y="$C/conf/cassandra.yaml"
  grep -q "^data_file_directories:" "$y" || cat >> "$y" <<YAML
data_file_directories:
  - $D/cassandra/data
commitlog_directory: $D/cassandra/commitlog
saved_caches_directory: $D/cassandra/saved_caches
hints_directory: $D/cassandra/hints
cdc_raw_directory: $D/cassandra/cdc_raw
YAML
}

up() {
  setup
  if ! (exec 3<>/dev/tcp/127.0.0.1/9092) 2>/dev/null; then
    KAFKA_HEAP_OPTS="-Xmx768m -Xms256m" LOG_DIR="$D/logs/kafka" nohup "$K/bin/kafka-server-start.sh" "$K/config/server.properties" > "$D/logs/kafka.out" 2>&1 &
  fi
  if ! (exec 3<>/dev/tcp/127.0.0.1/9042) 2>/dev/null; then
    MAX_HEAP_SIZE=1G HEAP_NEWSIZE=256M CASSANDRA_LOG_DIR="$D/logs/cassandra" nohup "$C/bin/cassandra" -f -p "$D/cassandra.pid" > "$D/logs/cassandra.out" 2>&1 &
  fi
  for i in $(seq 1 90); do
    k=no; c=no
    (exec 3<>/dev/tcp/127.0.0.1/9092) 2>/dev/null && k=up
    (exec 3<>/dev/tcp/127.0.0.1/9042) 2>/dev/null && c=up
    [ "$k$c" = upup ] && { echo "kafka :9092 up · cassandra :9042 up"; return 0; }
    sleep 2
  done
  echo "timeout: kafka=$k cassandra=$c (see .data/logs)"; return 1
}

down() {
  "$K/bin/kafka-server-stop.sh" >/dev/null 2>&1 || true
  [ -f "$D/cassandra.pid" ] && kill "$(cat "$D/cassandra.pid")" 2>/dev/null || true
  echo "stopped"
}

status() {
  for p in 9092 9042; do (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && echo "$p up" || echo "$p down"; done
}

"${1:-status}"

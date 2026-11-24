#!/bin/bash
# Runs ON the EC2 instance (passed as user-data by launch.sh). CPU-only PGW customer-side benchmarks.
# Downloads the code from the run's private S3 bucket, builds it, runs the benchmarks, uploads
# results.tgz to the same bucket, then powers off (the instance is set to terminate on shutdown).
export HOME=/root
BUCKET="__BUCKET__"
REGION="__REGION__"
exec > >(tee -a /var/log/pgw-bench.log | tee /dev/ttyS0) 2>&1
shutdown -h +150 "pgw-bench hard time cap"          # never runs longer than 2.5 h
step() { echo "PGW-STEP $(date -u +%H:%M:%S) $*"; }
OUT=/root/out; mkdir -p $OUT

step "install system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q && apt-get install -yq python3-venv python3-dev build-essential curl unzip
python3 -m venv /root/venv && . /root/venv/bin/activate
pip install -q --upgrade pip awscli

step "fetch code"
cd /root && aws s3 cp --region "$REGION" "s3://$BUCKET/pgw.tgz" pgw.tgz && mkdir -p pgw && tar xzf pgw.tgz -C pgw
cd /root/pgw

step "install python packages"
pip install -q torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q -c constraints.txt -e ".[service,pack,gateway,factory]" \
  || pip install -q -e ".[service,pack,gateway,factory]"

step "build native kernels"
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
. /root/.cargo/env
python rust/tools/build_fast.py && python rust/tools/build_ntt.py && python rust/tools/build_pks.py
(cd rust && cargo build --release -p pgw-pks --bin pks-bench)
export PYTHONPATH=src

step "machine"
{ lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|L3"; grep -o -w -E "avx512ifma|avx512f|avx2" /proc/cpuinfo | sort -u | tr '\n' ' '; echo; } > $OUT/machine.txt
cat $OUT/machine.txt

step "gateway per-token CPU floor (7B / 32B / 72B)"
for p in 7b 32b 72b; do for c in 1 2; do
  timeout 1800 python benchmarks/native_trip_floor_bench.py --preset $p --prime --tokens 20 --cores $c \
    --json $OUT/trip_${p}_c${c}.json > $OUT/trip_${p}_c${c}.log 2>&1 || echo "trip $p c$c failed"
done; done

step "encrypted mask unit costs (pks-v3 kernels)"
B=rust/target/release/pks-bench
for t in 1 8 32; do
  PKS_REPS=11 $B decrypt 4096 4 $t avx512ifma >> $OUT/pks.jsonl 2>&1
  PKS_REPS=11 $B encrypt 5120 4 $t avx512ifma >> $OUT/pks.jsonl 2>&1
done
$B ksk 11 >> $OUT/pks.jsonl 2>&1

step "customer CPU per mask slot, 7B and 32B (plain / mlwe / pks-v3)"
for m in qwen2.5-7b qwen2.5-32b; do
  timeout 2700 python benchmarks/fused2_he_split_bench.py --model $m --slots 256 --rows 256 \
    --json $OUT/maskcost_$m.json > $OUT/maskcost_$m.log 2>&1 || echo "maskcost $m failed"
done

step "summary"
python - <<'PY' > $OUT/summary.txt
import glob, json, os
print(open("/root/out/machine.txt").read().strip())
for f in sorted(glob.glob("/root/out/trip_*.json")):
    d = json.load(open(f))
    print(f"trip {d['preset']:>3} cores={d['cores']} main_ms/token={d['main_core_ms_per_token_full_model']} "
          f"us/trip={d['main_us_per_trip']} helper_ms/token={d['helper_core_ms_per_token']} trips={d['trips_per_token']}")
for line in open("/root/out/pks.jsonl"):
    print("pks", line.strip())
for f in sorted(glob.glob("/root/out/maskcost_*.json")):
    d = json.load(open(f))
    print("maskcost", d["model"], json.dumps(d.get("schemes")), "x_plain", json.dumps(d.get("x_plain")))
for f in sorted(glob.glob("/root/out/*.log")):
    if os.path.getsize(f) and "Traceback" in open(f).read():
        print("ERROR in", os.path.basename(f), open(f).read()[-600:])
PY
cat $OUT/summary.txt

step "upload results"
tar czf /root/results.tgz -C /root out /var/log/pgw-bench.log 2>/dev/null
aws s3 cp --region "$REGION" /root/results.tgz "s3://$BUCKET/results.tgz"
step "done; powering off"
shutdown -h +1 "pgw-bench finished"

#!/bin/bash
# Launch 4 N300 workers in parallel, each on its own board
set -e
cd "$(dirname "$0")"
export TT_METAL_HOME=$PWD
source python_env/bin/activate > /dev/null 2>&1
export TT_SYMBIOTE_RUN_MODE=TRACED
export MESH_DEVICE=N300
export PYTHONUNBUFFERED=1

# PCI bus addresses for the 4 N300 boards (each board = 2 chips with TP=2)
BUSES=("0000:31:00.0" "0000:4b:00.0" "0000:b1:00.0" "0000:ca:00.0")

# Page assignment: 9 pages across 4 workers (page 7 isolated to fresh worker)
PAGES=("1,2,3" "4,5,6" "7" "8,9")

START=$(date +%s.%N)
PIDS=()
for wid in 0 1 2 3; do
    bus=${BUSES[$wid]}
    pages=${PAGES[$wid]}
    echo "Launching worker $wid on bus $bus, pages=$pages"
    (
        export WORKER_ID=$wid
        export WORKER_PAGES=$pages
        export TT_METAL_PCI_BUS_IDS=$bus
        python -u -m pytest models/experimental/tt_symbiote/tests/test_dots_ocr_t1_dp.py -xvs --timeout=0 \
            > /tmp/t1_dp_w$wid.log 2>&1
    ) &
    PIDS+=($!)
    sleep 60  # longer stagger: ensures fabric init completes before next worker triggers it
done

echo "Workers launched, waiting..."

# Graceful interrupt handler — SIGTERM not SIGKILL, allow pipeline.release()
cleanup() {
    echo "Sending SIGTERM to workers for graceful shutdown..."
    for pid in "${PIDS[@]}"; do
        kill -TERM $pid 2>/dev/null
    done
    # Give 30s for graceful exit
    for i in 1 2 3 4 5 6; do
        sleep 5
        ALIVE=0
        for pid in "${PIDS[@]}"; do
            kill -0 $pid 2>/dev/null && ALIVE=$((ALIVE+1))
        done
        [ $ALIVE -eq 0 ] && break
        echo "  $ALIVE workers still releasing... ($((i*5))s)"
    done
    # If still alive after 30s, escalate but warn
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "  WARN: $pid stuck after SIGTERM, SIGKILL (hw may need recover)"
            kill -KILL $pid
        fi
    done
    exit 130
}
trap cleanup INT TERM

for pid in "${PIDS[@]}"; do
    wait $pid
done
END=$(date +%s.%N)

WALL=$(python3 -c "print(f'{$END-$START:.2f}')")
echo "===================="
echo "TOTAL WALL TIME: ${WALL}s"
echo "===================="
for wid in 0 1 2 3; do
    if [ -f /tmp/t1_dp_worker_${wid}.json ]; then
        echo "Worker $wid:"
        python3 -c "import json; d=json.load(open('/tmp/t1_dp_worker_${wid}.json')); print(f'  total={d[\"total_s\"]:.1f}s, pages={[r[\"page\"] for r in d[\"results\"]]}, tokens={[r[\"tokens\"] for r in d[\"results\"]]}')"
    else
        echo "Worker $wid: FAILED, see /tmp/t1_dp_w${wid}.log"
    fi
done

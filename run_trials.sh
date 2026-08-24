#!/bin/bash
# Runs a batch of trials of starpu_halo_ring with a per-run timeout, classifying
# each run as clean / hung (timed out) / crashed (nonzero exit before timeout).
# Usage: run_trials.sh <ranks> <nt> <nsteps> <trials> [timeout_seconds]
set -u
RANKS=$1
NT=$2
NSTEPS=$3
TRIALS=$4
TIMEOUT=${5:-60}

export STARPU_NCPU=2
export STARPU_SILENT=1

clean=0
hung=0
crashed=0
mismatched=0

for i in $(seq 1 "$TRIALS"); do
    out=$(timeout "${TIMEOUT}s" mpirun --allow-run-as-root -n "$RANKS" ./starpu_halo_ring --nt "$NT" --nsteps "$NSTEPS" 2>&1)
    rc=$?
    if [ $rc -eq 124 ]; then
        hung=$((hung + 1))
        echo "trial $i: HUNG"
    elif [ $rc -ne 0 ]; then
        crashed=$((crashed + 1))
        echo "trial $i: CRASHED (rc=$rc)"
        echo "$out" | tail -5
    elif echo "$out" | grep -q "DATA MISMATCH"; then
        mismatched=$((mismatched + 1))
        echo "trial $i: DATA MISMATCH"
    else
        clean=$((clean + 1))
        echo "trial $i: clean"
    fi
done

echo "=== -n $RANKS --nt $NT --nsteps $NSTEPS x$TRIALS (timeout ${TIMEOUT}s) ==="
echo "clean=$clean hung=$hung crashed=$crashed mismatched=$mismatched"

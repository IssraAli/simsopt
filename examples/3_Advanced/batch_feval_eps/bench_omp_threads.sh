#!/bin/bash
# A/B check: does giving each candidate evaluation more threads actually
# speed it up? Runs bench_omp_threads.py once per thread count, each in its
# own fresh process (thread-count env vars are read at library-load time,
# so they can't be swept within a single running process).
#
# On an interactive Slurm allocation (salloc), the shell itself is already
# a job step (Princeton RC wraps salloc as `srun --pty $SHELL`), so it
# carries SLURM_* env vars. A plain `python ...` launched from that shell
# gets caught by Open MPI's PMIx component trying to join that job step and
# fails with "OPAL ERROR: Unreachable ... direct launched using srun". Each
# trial is relaunched through `srun --mpi=pmix -n 1` below to give it a
# clean, correctly-negotiated PMI context instead.
#
# Usage:
#   ./bench_omp_threads.sh                # sweep 1 2 4 8
#   ./bench_omp_threads.sh 1 2 4 8 16      # custom thread counts

set -e
cd "$(dirname "$0")"

THREAD_COUNTS=("$@")
if [ ${#THREAD_COUNTS[@]} -eq 0 ]; then
    THREAD_COUNTS=(1 2 4 8)
fi

# On a cluster (inside a salloc allocation), route each trial through srun
# so it gets its own clean PMI context instead of inheriting the
# allocation shell's job-step env vars. Locally (no srun on the Mac), just
# run python directly.
if command -v srun >/dev/null 2>&1; then
    LAUNCH=(srun --mpi=pmix -n 1)
else
    LAUNCH=()
fi

echo "threads,elapsed_sec"
for n in "${THREAD_COUNTS[@]}"; do
    launch_cmd=("${LAUNCH[@]}")
    if [ ${#launch_cmd[@]} -gt 0 ]; then
        launch_cmd+=(--cpus-per-task="$n")
    fi
    out=$(OMP_NUM_THREADS="$n" MKL_NUM_THREADS="$n" OPENBLAS_NUM_THREADS="$n" NUMEXPR_NUM_THREADS="$n" \
        "${launch_cmd[@]}" python bench_omp_threads.py 2>/tmp/bench_omp_threads_stderr.log) || {
            echo "$n,FAILED (see /tmp/bench_omp_threads_stderr.log)"
            continue
        }
    elapsed=$(echo "$out" | grep -o 'elapsed_sec=[0-9.]*' | cut -d= -f2)
    echo "$n,$elapsed"
done

#!/bin/bash
#SBATCH --job-name=batch_feval_bo
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# One MPI rank evaluates one candidate surface (vmec runs serially per rank,
# see parvmecinfo.txt: "active processors: 1"), so --ntasks == BO batch size
# per iteration. Bump this up/down to change how many candidates are
# evaluated concurrently each round; keep --ntasks-per-node <= 96 on the
# Intel nodes (or <= 128 if you request the AMD nodes).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
# Uncomment and set if you have a specific allocation/partition to target,
# e.g. the PPPL-affiliated partition:
##SBATCH --partition=pppl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=issra@princeton.edu

set -e

# --- environment -------------------------------------------------------
module purge
module load anaconda3/2026.7
module load openmpi/gcc/4.1.8

# Name of the conda env with simsopt + botorch/gpytorch installed, and
# mpi4py built against the openmpi module above (per Princeton RC's
# mpi4py guide: `pip install mpi4py`, not `conda install mpi4py`).
CONDA_ENV=simsopt
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS=1
export OMP_PLACES=threads

# --- run -----------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

srun --mpi=pmix python spline_test_bo.py

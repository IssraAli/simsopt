#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=cpu
#SBATCH --output=%j_output.txt

module load python
# Replace 'simsopt_env' with your actual Conda environment if needed
# conda activate simsopt_env

cd $SCRATCH
srun python NERSC-LHD_opt_w_dipoles_5-31-25.py
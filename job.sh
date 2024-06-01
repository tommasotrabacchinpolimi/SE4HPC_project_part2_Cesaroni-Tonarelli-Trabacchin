#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH --account=tra24_sepolimi
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1

module load profile/advanced
srun singularity exec container.sif /app/build/test_multiplication
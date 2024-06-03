#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH --account=tra24_sepolimi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=g100_usr_prod
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

export SINGULARITYENV_TMPDIR=$TMPDIR
module load profile/base
srun exec --bind $TMPDIR:$TMPDIR SWENG-Ces-Ton-Trab/container.sif bash -c "export OMPI_MCA_tmpdir_base=$TMPDI$




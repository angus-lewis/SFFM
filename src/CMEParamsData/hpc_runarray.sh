#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=3-0:00:00
#SBATCH --mem=8GB
#SBATCH --array=0-24

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=angus.lewis@adelaide.edu.au


module load Julia/1.6.0

echo "job_id $SLURM_ARRAY_TASK_ID"
julia preprocessCMEParams.jl $SLURM_ARRAY_TASK_ID



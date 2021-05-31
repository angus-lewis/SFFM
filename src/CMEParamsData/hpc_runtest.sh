#!/bin/bash
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=0:15:00
#SBATCH --mem=1GB

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=angus.lewis@adelaide.edu.au


module load Julia/1.6.0

julia preprocessCMEParams.jl 



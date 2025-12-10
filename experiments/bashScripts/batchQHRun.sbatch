#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=QHRun
#SBATCH --mail-type=END
#SBATCH --mail-user=meh9673@nyu.edu
#SBATCH --output=QH_%j.out

module purge
cd /scratch/projects/kaptanoglulab/MH
./QHRun${1}.bash

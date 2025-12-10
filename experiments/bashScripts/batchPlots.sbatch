#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=StelPlot
#SBATCH --mail-type=END
#SBATCH --mail-user=meh9673@nyu.edu
#SBATCH --output=plot_%j.out

module purge
cd /scratch/projects/kaptanoglulab/MH
for file in 11_14_Run_QH/Output*; do
       ./plot.bash  ${file}
done

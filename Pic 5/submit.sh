#!/bin/bash
#
#SBATCH --job-name=Pic5_10
#SBATCH --output=res_Pic_5_7-10_3.out
#SBATCH --error=res_Pic_5_7-10_3.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=2-00:00
#SBATCH --mem=100G

srun julia --threads 128 ./run_10_low_memory.jl
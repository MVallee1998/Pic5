#!/bin/bash
#
#SBATCH --job-name=Pic_5
#SBATCH --output=res_Pic_5.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=10

srun julia --threads 256 Pic 5/algo_pic5_link_parallel_all_v_better.jl
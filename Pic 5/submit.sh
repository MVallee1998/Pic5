#!/bin/bash
#
#SBATCH --job-name=Pic_5
#SBATCH --output=res_Pic_5.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --time=20:00
#SBATCH --mem=100G

srun julia --threads=100 ./algo_pic5_link_parallel_all_v_better.jl
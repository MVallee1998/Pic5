#!/bin/bash
#
#SBATCH --job-name=Pic_5_n=10
#SBATCH --output=res_Pic_5_7-10.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00
#SBATCH --mem=100G

srun julia --threads=100 ./algo_pic5_link_parallel_all_v_better.jl
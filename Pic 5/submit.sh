#!/bin/bash
#
#SBATCH --job-name=Pic_5_n=10
#SBATCH --output=res_Pic_5_7-10.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=2-00:00
#SBATCH --mem=100G

srun julia --threads=128 ./algo_pic5_link_parallel_all_v_better.jl
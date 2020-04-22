#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=7-00:00
#SBATCH --mail-user=whogan@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=cnn_and_lstm
#SBATCH --mem=32GB
#SBATCH --output "%x.%j.%N.out"
#SBATCH --get-user-env=L
#SBATCH --gres=gpu:1
#SBATCH -o /projects/ibm_aihl/whogan/acronym_resolution/out/ar_slurm_progress.log
#SBATCH -e /projects/ibm_aihl/whogan/acronym_resolution/out/ar_slurm_progress.err

# Manual env setup
export LD_PRELOAD=/home/whogan/.conda/envs/latest_python/lib/libstdc++.so.6.0.26

source /opt/miniconda3/bin/activate latest_python

# Main file
python /projects/ibm_aihl/whogan/acronym_resolution/gpt2_fine_tune.py

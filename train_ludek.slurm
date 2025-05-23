#!/bin/bash
#SBATCH --job-name=netml
#SBATCH --output=/scratch/izar/$USER/netml/outputs/slurm/train_%j.out
#SBATCH --error=/scratch/izar/$USER/netml/outputs/slurm/train_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account=master

# Ensure output directory exists
mkdir -p /scratch/izar/$USER/netml/outputs/slurm

cd /home/$USER/netml-project
source /home/$USER/venvs/netml/bin/activate

python train.py

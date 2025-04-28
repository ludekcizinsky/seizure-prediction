#!/bin/bash
#SBATCH --job-name=netml
#SBATCH --output=/scratch/izar/${1}/netml/outputs/slurm/train_%j.out
#SBATCH --error=/scratch/izar/${1}/netml/outputs/slurm/train_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --qos=ee-452
#SBATCH --account=ee-452

USERNAME="$1"

cd /home/${USERNAME}/netml-project
source /home/${USERNAME}/venvs/netml/bin/activate

python train.py
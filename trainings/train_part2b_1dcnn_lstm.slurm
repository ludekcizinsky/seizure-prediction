#!/bin/bash
#SBATCH --job-name=part2b_1dcnn_lstm
#SBATCH --output=/scratch/izar/cizinsky/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/netml/outputs/slurm/%x.%j.err
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account=master

USER=cizinsky

# Ensure output directory exists
mkdir -p /scratch/izar/$USER/netml/outputs/slurm

# Activate virtual environment
cd /home/$USER/netml-project
source /home/$USER/venvs/netml/bin/activate

# Part 2b: 1D CNN (1 hour per experiment)
python train.py signal_transform=neural_conv1d_small temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2b, cnn1d_small, lstm]'
python train.py signal_transform=neural_conv1d_medium temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2b, cnn1d_medium, lstm]'
python train.py signal_transform=neural_conv1d_large temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2b, cnn1d_large, lstm]'


#!/bin/bash
#SBATCH --job-name=part2c_window_gcn_lstm
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

# Part 2c: Windowed GCN (1 hour per experiment)
python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=gcn graph_module.num_classes=0 graph_module.conv_layers.conv1.in_channels=100 classifier=lstm data.batch_size=256 'logger.tags=[part2c, cnn1d_small, lstm]'
# python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=gcn graph_module.num_classes=0 graph_module.conv_layers.conv1.in_channels=100 classifier=lstm data.batch_size=256 'logger.tags=[debug]' data.num_workers=10


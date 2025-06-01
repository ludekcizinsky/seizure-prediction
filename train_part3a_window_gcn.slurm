#!/bin/bash
#SBATCH --job-name=part3a_window_gcn_lstm
#SBATCH --output=/scratch/izar/cizinsky/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/netml/outputs/slurm/%x.%j.err
#SBATCH --time=06:00:00
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

# Part 3a: Windowed GCN 
# -- decided to not do these since it takes unreasonably long to run, instead we first need to focus on 2c and getting and seeing how things work with shorter epochs and more frequent validation
# 4h 15 min
# python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm data.batch_size=256 trainer.max_epochs=250 'logger.tags=[part3a, distance, window_gcn, lstm]'
# far too long
# python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_correlation graph_module=window_gcn classifier=lstm data.batch_size=128 trainer.max_epochs=250 'logger.tags=[part3a, correlation, window_gcn, lstm]'
# est. 7 hours
# python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_learnable_adj graph_module=window_gcn classifier=lstm data.batch_size=128 trainer.max_epochs=250 'logger.tags=[part3a, learnable_adj, window_gcn, lstm]'

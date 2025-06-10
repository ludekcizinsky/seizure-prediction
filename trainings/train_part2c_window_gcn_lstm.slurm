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

# Part 2c: Windowed GCN (5 hours per experiment)
# -- Initial try with 250 epochs, validation every 25 epochs
# python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm data.batch_size=256 trainer.max_epochs=250 'logger.tags=[part2c, window_gcn, lstm]'

# -- 50 epochs, validation every 2 epochs, ~90 min
python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm data.batch_size=256 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part2c, window_gcn, lstm, only_50_epochs]'
python train.py optim.lr=1e-4 signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm data.batch_size=256 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part2c, window_gcn, lstm, only_50_epochs, lr]'
python train.py optim.weight_decay=0.1 signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm data.batch_size=256 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part2c, window_gcn, lstm, only_50_epochs, lr, weight_decay]'
python train.py data.use_weighted_sampler=True signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm data.batch_size=256 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part2c, window_gcn, lstm, only_50_epochs, weighted_sampler]'

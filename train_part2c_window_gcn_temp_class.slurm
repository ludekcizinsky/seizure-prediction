#!/bin/bash
#SBATCH --job-name=part2c_window_gcn_temp_class
#SBATCH --output=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --ntasks=1

USER=mlebras

# Ensure output directory exists
mkdir -p /scratch/izar/$USER/netml/outputs/slurm

# Activate virtual environment
cd /home/$USER/nml-project
source /home/$USER/nml-project/venvs/netml/bin/activate

# Part 2c: Windowed GCN (5 hours per experiment)
# -- Initial try with 250 epochs, validation every 25 epochs
python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=conv1d data.batch_size=256 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part2c, window_gcn, conv1d]'
python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=tencoder data.batch_size=256 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part2c, window_gcn, tencoder]'
python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm_att data.batch_size=256 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part2c, window_gcn, lstm_att]'
python train.py signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn graph_module.pool_type=learned classifier=lstm_att data.batch_size=256 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part2c, window_gcn, lstm_att]'

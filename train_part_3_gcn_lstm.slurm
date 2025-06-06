#!/bin/bash
#SBATCH --job-name=cv-gcn-windows-lstm
#SBATCH --output=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.err
#SBATCH --time=8:00:00
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

# gcn ~ 1h per fold
python train.py n_folds=3 fold_id=0 signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm data.batch_size=256 graph_builder.distance_threshold=1.5 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part3, window_gcn, lstm learned_pool, dist_1_5, fold_0]'
python train.py n_folds=3 fold_id=1 signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm data.batch_size=256 graph_builder.distance_threshold=1.5 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part3, window_gcn, lstm learned_pool, dist_1_5, fold_1]'
python train.py n_folds=3 fold_id=2 signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm data.batch_size=256 graph_builder.distance_threshold=1.5 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part3, window_gcn, lstm learned_pool, dist_1_5, fold_2]'

python train.py n_folds=3 fold_id=0 signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm_att data.batch_size=256 graph_builder.distance_threshold=1.5 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part3, window_gcn, lstm_att learned_pool, dist_1_5, fold_0]'
python train.py n_folds=3 fold_id=1 signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm_att data.batch_size=256 graph_builder.distance_threshold=1.5 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part3, window_gcn, lstm_att learned_pool, dist_1_5, fold_1]'
python train.py n_folds=3 fold_id=2 signal_transform=disabled temporal_module=disabled graph_builder=window_distance graph_module=window_gcn classifier=lstm_att data.batch_size=256 graph_builder.distance_threshold=1.5 trainer.max_epochs=50 trainer.check_val_every_n_epoch=2 optim.warmup_epochs=5 'logger.tags=[part3, window_gcn, lstm_att learned_pool, dist_1_5, fold_2]'
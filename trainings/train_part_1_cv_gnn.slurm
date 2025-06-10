#!/bin/bash
#SBATCH --job-name=cv-gnn
#SBATCH --output=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --qos=cs-503
#SBATCH --account=cs-503

USER=mlebras

# Ensure output directory exists
mkdir -p /scratch/izar/$USER/netml/outputs/slurm

# Activate virtual environment
cd /home/$USER/nml-project
source /home/$USER/nml-project/venvs/netml/bin/activate

# gcn ~ 1h per fold
python train.py n_folds=3 fold_id=0 temporal_module=disabled graph_builder=distance graph_module=gcn 'logger.tags=[part1, fft, gcn, baseline, fold_0]'
python train.py n_folds=3 fold_id=1 temporal_module=disabled graph_builder=distance graph_module=gcn 'logger.tags=[part1, fft, gcn, baseline, fold_1]'
python train.py n_folds=3 fold_id=2 temporal_module=disabled graph_builder=distance graph_module=gcn 'logger.tags=[part1, fft, gcn, baseline, fold_2]'

# gat ~ 1h per fold
python train.py n_folds=3 fold_id=0 temporal_module=disabled graph_builder=distance graph_module=gat 'logger.tags=[part1, fft, gat, baseline, fold_0]'
python train.py n_folds=3 fold_id=1 temporal_module=disabled graph_builder=distance graph_module=gat 'logger.tags=[part1, fft, gat, baseline, fold_1]'
python train.py n_folds=3 fold_id=2 temporal_module=disabled graph_builder=distance graph_module=gat 'logger.tags=[part1, fft, gat, baseline, fold_2]'
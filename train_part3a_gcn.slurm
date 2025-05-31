#!/bin/bash
#SBATCH --job-name=part3a_gcn
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

# Part 3a GCN with different graph builders 
# 45 mins
python train.py temporal_module=disabled graph_builder=distance graph_module=part1_best 'logger.tags=[part3a, distance, gcn]'
# 2h 30 mins
python train.py temporal_module=disabled graph_builder=correlation graph_module=part1_best 'logger.tags=[part3a, correlation, gcn]'
# 1 hour
python train.py temporal_module=disabled graph_builder=learnable_adj graph_module=part1_best 'logger.tags=[part3a, learnable_adj, gcn]'

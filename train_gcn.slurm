#!/bin/bash
#SBATCH --job-name=gcn_ablation
#SBATCH --output=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.err
#SBATCH --time=16:00:00
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

# Part 1: GCN ablations (1 epoch = ~3s)
# - baseline
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn 'logger.tags=[part1, fft, gcn, baseline]'

# - lr (default: 5e-4)
python train.py  temporal_module=disabled graph_builder=distance graph_module=gcn optim.lr=1e-4 'logger.tags=[part1, fft, gcn, lr]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn optim.lr=1e-5 'logger.tags=[part1, fft, gcn, lr]'

# - weight decay (default: 0.0)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn optim.weight_decay=0.1 'logger.tags=[part1, fft, gcn, wd]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn optim.weight_decay=0.2 'logger.tags=[part1, fft, gcn, wd]'

# - data normalization (default: False)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn data.normalize=True 'logger.tags=[part1, fft, gcn, normalize]'

# - batch normalisation (default: False)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.use_batchnorm=True 'logger.tags=[part1, fft, gcn, batchnorm]'

# - hidden dim (default: 64)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn_small 'logger.tags=[part1, fft, gcn, hidden_dim]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn_big 'logger.tags=[part1, fft, gcn, hidden_dim]'

# - pooling (default: mean)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.pool_type=max 'logger.tags=[part1, fft, gcn, pooling]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.pool_type=add 'logger.tags=[part1, fft, gcn, pooling]'

# - distance threshold (default: 1)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_builder.distance_threshold=1.5 'logger.tags=[part1, fft, gcn, distance_threshold]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn, distance_threshold]'

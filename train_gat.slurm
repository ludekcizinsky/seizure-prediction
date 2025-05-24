#!/bin/bash
#SBATCH --job-name=gat_ablation
#SBATCH --output=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.err
#SBATCH --time=24:00:00
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

# Part 1: GAT ablations (1 epoch = ~5s)
# - baseline
python train.py temporal_module=disabled graph_builder=distance graph_module=gat 'logger.tags=[part1, fft, gat, baseline]'

# - lr (default: 5e-4)
python train.py  temporal_module=disabled graph_builder=distance graph_module=gat optim.lr=1e-4 'logger.tags=[part1, fft, gat, lr]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gat optim.lr=1e-5 'logger.tags=[part1, fft, gat, lr]'

# - weight decay (default: 0.0)
python train.py temporal_module=disabled graph_builder=distance graph_module=gat optim.weight_decay=0.1 'logger.tags=[part1, fft, gat, wd]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gat optim.weight_decay=0.2 'logger.tags=[part1, fft, gat, wd]'

# - data normalization (default: False)
python train.py temporal_module=disabled graph_builder=distance graph_module=gat data.normalize=True 'logger.tags=[part1, fft, gat, normalize]'

# - batch normalisation (default: False)
python train.py temporal_module=disabled graph_builder=distance graph_module=gat graph_module.use_batchnorm=True 'logger.tags=[part1, fft, gat, batchnorm]'

# - hidden dim (default: 64)
python train.py temporal_module=disabled graph_builder=distance graph_module=gat_small 'logger.tags=[part1, fft, gat, hidden_dim]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gat_big 'logger.tags=[part1, fft, gat, hidden_dim]'

# - num heads (default: 8)
python train.py temporal_module=disabled graph_builder=distance graph_module=gat_few_heads 'logger.tags=[part1, fft, gat, heads]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gat_many_heads 'logger.tags=[part1, fft, gat, heads]'

# - pooling (default: mean)
python train.py temporal_module=disabled graph_builder=distance graph_module=gat graph_module.pool_type=max 'logger.tags=[part1, fft, gat, pooling]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gat graph_module.pool_type=add 'logger.tags=[part1, fft, gat, pooling]'

# - distance threshold (default: 1)
python train.py temporal_module=disabled graph_builder=distance graph_module=gat graph_builder.distance_threshold=1.5 'logger.tags=[part1, fft, gat, distance_threshold]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gat graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gat, distance_threshold]'

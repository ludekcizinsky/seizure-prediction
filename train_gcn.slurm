#!/bin/bash
#SBATCH --job-name=gcn_ablation_0_5_dist
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
# python train.py temporal_module=disabled graph_builder=distance graph_module=gcn 'logger.tags=[part1, fft, gcn, baseline]'

# - lr (default: 5e-4)
python train.py  temporal_module=disabled graph_builder=distance graph_module=gcn optim.lr=1e-4 graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, lr]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn optim.lr=1e-5 graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, lr]'

# - weight decay (default: 0.0)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn optim.weight_decay=0.1 graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, wd]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn optim.weight_decay=0.2 graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, wd]'

# - data normalization (default: False)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn data.normalize=True graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, normalize]'

# - batch normalisation (default: False)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.use_batchnorm=True graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, batchnorm]'

# - hidden dim (default: 64)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn_small graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, hidden_dim]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn_big graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, hidden_dim]'

# - pooling (default: mean)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.pool_type=max graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, pooling]'
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.pool_type=add graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn_0_5_dist, pooling]'

# - distance threshold (default: 1)
# python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_builder.distance_threshold=1.5 'logger.tags=[part1, fft, gcn, distance_threshold]'
# python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_builder.distance_threshold=0.5 'logger.tags=[part1, fft, gcn, distance_threshold]'

# - weighted sampler (default: False)
python train.py temporal_module=disabled graph_builder=distance graph_module=gcn graph_builder.distance_threshold=0.5 data.use_weighted_sampler=True 'logger.tags=[part1, fft, gcn_0_5_dist, weighted_sampler]'
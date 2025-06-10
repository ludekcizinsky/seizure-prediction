#!/bin/bash
#SBATCH --job-name=conv1d_ablation
#SBATCH --output=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/mlebras/netml/outputs/slurm/%x.%j.err
#SBATCH --time=15:00:00
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

# Part 1: conv1d ablations (1 epoch = ~3s)
# - baseline
python train.py temporal_module=conv1d graph_builder=disabled graph_module=disabled 'logger.tags=[part1, fft, conv1d, baseline]'

# - lr (default: 5e-4)
python train.py  temporal_module=conv1d graph_builder=disabled graph_module=disabled optim.lr=1e-4 'logger.tags=[part1, fft, conv1d, lr]'
python train.py temporal_module=conv1d graph_builder=disabled graph_module=disabled optim.lr=1e-5 'logger.tags=[part1, fft, conv1d, lr]'

# - weight decay (default: 0.0)
python train.py temporal_module=conv1d graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part1, fft, conv1d, wd]'
python train.py temporal_module=conv1d graph_builder=disabled graph_module=disabled optim.weight_decay=0.2 'logger.tags=[part1, fft, conv1d, wd]'

# - data normalization (default: False)
python train.py temporal_module=conv1d graph_builder=disabled graph_module=disabled data.normalize=True 'logger.tags=[part1, fft, conv1d, normalize]'

# - hidden dim (default: 64 -> 128 -> 256)
python train.py temporal_module=conv1d_small graph_builder=disabled graph_module=disabled 'logger.tags=[part1, fft, conv1d, hidden_dim]'
python train.py temporal_module=conv1d_big graph_builder=disabled graph_module=disabled 'logger.tags=[part1, fft, conv1d, hidden_dim]'

# - kernels
python train.py temporal_module=conv1d_small_uniform_kernels graph_builder=disabled graph_module=disabled 'logger.tags=[part1, fft, conv1d, kernels]'
python train.py temporal_module=conv1d_big_uniform_kernels graph_builder=disabled graph_module=disabled 'logger.tags=[part1, fft, conv1d, kernels]'
python train.py temporal_module=conv1d_big_kernels graph_builder=disabled graph_module=disabled 'logger.tags=[part1, fft, conv1d, kernels]'

# - weight sampler (default: False)
python train.py temporal_module=conv1d graph_builder=disabled graph_module=disabled data.use_weighted_sampler=True 'logger.tags=[part1, fft, conv1d, weighted_sampler]'

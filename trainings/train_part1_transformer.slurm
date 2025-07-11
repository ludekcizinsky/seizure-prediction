#!/bin/bash
#SBATCH --job-name=lstm_ablation
#SBATCH --output=/scratch/izar/cizinsky/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/netml/outputs/slurm/%x.%j.err
#SBATCH --time=15:00:00
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

# Part 1: Transformer encoder 
# - baseline (total time: 1h30min)
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled 'logger.tags=[part1, fft, tencoder, baseline]'

# - lr (default: 5e-4)
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled optim.lr=1e-4 'logger.tags=[part1, fft, tencoder, lr]'
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled optim.lr=1e-5 'logger.tags=[part1, fft, tencoder, lr]'

# - weight decay (default: 0.0)
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part1, fft, tencoder, wd]'
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled optim.weight_decay=0.2 'logger.tags=[part1, fft, tencoder, wd]'

# - data normalization (default: False)
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled data.normalize=True 'logger.tags=[part1, fft, tencoder, normalize]'

# - hidden dim (default: 64)
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled temporal_module.transformer_dim=32 'logger.tags=[part1, fft, tencoder, hidden_dim]'

# - num heads (default: 4)
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled temporal_module.num_heads=2 'logger.tags=[part1, fft, tencoder, num_heads]'

# - dropout (default: 0.1)
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled temporal_module.dropout=0.2 'logger.tags=[part1, fft, tencoder, dropout]'

# - weight sampler (default: False)
python train.py temporal_module=tencoder graph_builder=disabled graph_module=disabled data.use_weighted_sampler=True 'logger.tags=[part1, fft, tencoder, weighted_sampler]'


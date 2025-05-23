#!/bin/bash
#SBATCH --job-name=lstm_ablation
#SBATCH --output=/scratch/izar/$USER/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/$USER/netml/outputs/slurm/%x.%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account=master

# Ensure output directory exists
mkdir -p /scratch/izar/$USER/netml/outputs/slurm

# Activate virtual environment
cd /home/$USER/netml-project
source /home/$USER/venvs/netml/bin/activate

# Part 1: LSTM ablations (1 epoch = ~2s)
# - baseline
python temporal_module=lstm graph_builder=disabled graph_module=disabled 'logger.tags=[part1, fft, lstm, baseline]'

# - lr (default: 5e-4)
python temporal_module=lstm graph_builder=disabled graph_module=disabled optim.lr=1e-4 'logger.tags=[part1, fft, lstm, lr]'
python temporal_module=lstm graph_builder=disabled graph_module=disabled optim.lr=1e-5 'logger.tags=[part1, fft, lstm, lr]'

# - weight decay (default: 0.0)
python temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part1, fft, lstm, wd]'
python temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.2 'logger.tags=[part1, fft, lstm, wd]'

# - data normalization (default: False)
python temporal_module=lstm graph_builder=disabled graph_module=disabled data.normalize=True 'logger.tags=[part1, fft, lstm, normalize]'

# - hidden dim (default: 64)
python temporal_module=lstm graph_builder=disabled graph_module=disabled temporal_module.hidden_dim=32 'logger.tags=[part1, fft, lstm, hidden_dim]'
python temporal_module=lstm graph_builder=disabled graph_module=disabled temporal_module.hidden_dim=128 'logger.tags=[part1, fft, lstm, hidden_dim]'

# - dropout (default: 0.3)
python temporal_module=lstm graph_builder=disabled graph_module=disabled temporal_module.dropout=0.1 'logger.tags=[part1, fft, lstm, dropout]'
python temporal_module=lstm graph_builder=disabled graph_module=disabled temporal_module.dropout=0.2 'logger.tags=[part1, fft, lstm, dropout]'

# - bi-directional (default: False)
python temporal_module=lstm graph_builder=disabled graph_module=disabled temporal_module.bidirectional=True 'logger.tags=[part1, fft, lstm, bidirectional]'
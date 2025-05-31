#!/bin/bash
#SBATCH --job-name=lstm_signal_transform
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

# Part 2: LSTM ablations (1 epoch = ~3s)
# - decimate
python train.py signal_transform=decimate signal_transform.q=1 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, decimate, lstm]'
python train.py signal_transform=decimate signal_transform.q=2 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, decimate, lstm]'
python train.py signal_transform=decimate signal_transform.q=5 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, decimate, lstm]'
python train.py signal_transform=decimate signal_transform.q=10 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, decimate, lstm]'
python train.py signal_transform=decimate signal_transform.q=15 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, decimate, lstm]'
python train.py signal_transform=decimate signal_transform.q=30 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, decimate, lstm]'

# - window downsample
python train.py signal_transform=window_downsample signal_transform.window=2 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, window_downsample, lstm]'
python train.py signal_transform=window_downsample signal_transform.window=5 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, window_downsample, lstm]'
python train.py signal_transform=window_downsample signal_transform.window=10 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, window_downsample, lstm]'
python train.py signal_transform=window_downsample signal_transform.window=12 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, window_downsample, lstm]'
python train.py signal_transform=window_downsample signal_transform.window=25 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, window_downsample, lstm]'
python train.py signal_transform=window_downsample signal_transform.window=50 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, window_downsample, lstm]'

# - dct dowsample
python train.py signal_transform=dct_downsample signal_transform.K=1500 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, dct_downsample, lstm]'
python train.py signal_transform=dct_downsample signal_transform.K=600 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, dct_downsample, lstm]'
python train.py signal_transform=dct_downsample signal_transform.K=300 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, dct_downsample, lstm]'
python train.py signal_transform=dct_downsample signal_transform.K=100 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, dct_downsample, lstm]'
python train.py signal_transform=dct_downsample signal_transform.K=50 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, dct_downsample, lstm]'

# - wavelet
python train.py signal_transform=wavelet signal_transform.level=1 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, wavelet, lstm]'
python train.py signal_transform=wavelet signal_transform.level=2 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, wavelet, lstm]'
python train.py signal_transform=wavelet signal_transform.level=3 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, wavelet, lstm]'
python train.py signal_transform=wavelet signal_transform.level=4 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, wavelet, lstm]'
python train.py signal_transform=wavelet signal_transform.level=5 temporal_module=lstm graph_builder=disabled graph_module=disabled optim.weight_decay=0.1 'logger.tags=[part2, wavelet, lstm]'
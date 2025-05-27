#!/bin/bash
#SBATCH --job-name=gcn_signal_transform
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

# Part 2: GCN ablations (1 epoch = ~4s)
# - decimate
python train.py signal_transform=decimate signal_transform.q=1 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=3000 graph_builder.distance_threshold=0.5 'logger.tags=[part2, decimate, gcn]'
python train.py signal_transform=decimate signal_transform.q=2 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=1500 graph_builder.distance_threshold=0.5 'logger.tags=[part2, decimate, gcn]'
python train.py signal_transform=decimate signal_transform.q=5 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=600 graph_builder.distance_threshold=0.5 'logger.tags=[part2, decimate, gcn]'
python train.py signal_transform=decimate signal_transform.q=10 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=300 graph_builder.distance_threshold=0.5 'logger.tags=[part2, decimate, gcn]'
python train.py signal_transform=decimate signal_transform.q=15 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=200 graph_builder.distance_threshold=0.5 'logger.tags=[part2, decimate, gcn]'
python train.py signal_transform=decimate signal_transform.q=30 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=100 graph_builder.distance_threshold=0.5 'logger.tags=[part2, decimate, gcn]'

# - window downsample
python train.py signal_transform=window_downsample signal_transform.window=2 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=1500 graph_builder.distance_threshold=0.5 'logger.tags=[part2, window_downsample, gcn]'
python train.py signal_transform=window_downsample signal_transform.window=5 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=600 graph_builder.distance_threshold=0.5 'logger.tags=[part2, window_downsample, gcn]'
python train.py signal_transform=window_downsample signal_transform.window=10 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=300 graph_builder.distance_threshold=0.5 'logger.tags=[part2, window_downsample, gcn]'
python train.py signal_transform=window_downsample signal_transform.window=12 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=250 graph_builder.distance_threshold=0.5 'logger.tags=[part2, window_downsample, gcn]'
python train.py signal_transform=window_downsample signal_transform.window=25 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=120 graph_builder.distance_threshold=0.5 'logger.tags=[part2, window_downsample, gcn]'
python train.py signal_transform=window_downsample signal_transform.window=50 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=60 graph_builder.distance_threshold=0.5 'logger.tags=[part2, window_downsample, gcn]'

# - dct dowsample
python train.py signal_transform=dct_downsample signal_transform.K=1500 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=1500 graph_builder.distance_threshold=0.5 'logger.tags=[part2, dct_downsample, gcn]'
python train.py signal_transform=dct_downsample signal_transform.K=600 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=600 graph_builder.distance_threshold=0.5 'logger.tags=[part2, dct_downsample, gcn]'
python train.py signal_transform=dct_downsample signal_transform.K=300 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=300 graph_builder.distance_threshold=0.5 'logger.tags=[part2, dct_downsample, gcn]'
python train.py signal_transform=dct_downsample signal_transform.K=100 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=100 graph_builder.distance_threshold=0.5 'logger.tags=[part2, dct_downsample, gcn]'
python train.py signal_transform=dct_downsample signal_transform.K=50 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=50 graph_builder.distance_threshold=0.5 'logger.tags=[part2, dct_downsample, gcn]'

# - wavelet
python train.py signal_transform=wavelet signal_transform.level=1 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=1503 graph_builder.distance_threshold=0.5 'logger.tags=[part2, wavelet, gcn]'
python train.py signal_transform=wavelet signal_transform.level=2 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=755 graph_builder.distance_threshold=0.5 'logger.tags=[part2, wavelet, gcn]'
python train.py signal_transform=wavelet signal_transform.level=3 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=381 graph_builder.distance_threshold=0.5 'logger.tags=[part2, wavelet, gcn]'
python train.py signal_transform=wavelet signal_transform.level=4 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=194 graph_builder.distance_threshold=0.5 'logger.tags=[part2, wavelet, gcn]'
python train.py signal_transform=wavelet signal_transform.level=5 temporal_module=disabled graph_builder=distance graph_module=gcn graph_module.conv_layers.conv1.out_channels=128 graph_module.conv_layers.conv2.in_channels=128 graph_module.conv_layers.conv1.in_channels=100 graph_builder.distance_threshold=0.5 'logger.tags=[part2, wavelet, gcn]'
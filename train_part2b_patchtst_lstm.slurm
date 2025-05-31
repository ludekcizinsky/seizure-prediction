#!/bin/bash
#SBATCH --job-name=part2b_patchtst
#SBATCH --output=/scratch/izar/cizinsky/netml/outputs/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/netml/outputs/slurm/%x.%j.err
#SBATCH --time=10:00:00
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

# Part 2b: PatchTST (max 5 hours per experiment)
# - baseline (non-overlapping patches, length/stride=10 -> 19 x 300 out)
python train.py signal_transform=neural_patch_tst signal_transform.patch_length=10 signal_transform.patch_stride=10 temporal_module=lstm graph_builder=disabled graph_module=disabled data.batch_size=85 trainer.max_epochs=250 optim.warmup_epochs=10 optim.weight_decay=0.1 'logger.tags=[part2b, patch_tst, non_over_lap]'
# - overlapping patches, length=15 stride = 10, 5 windows overlap -> 19 x 299 out
python train.py signal_transform=neural_patch_tst signal_transform.patch_length=15 signal_transform.patch_stride=10 temporal_module=lstm graph_builder=disabled graph_module=disabled data.batch_size=85 trainer.max_epochs=250 optim.warmup_epochs=10 optim.weight_decay=0.1 'logger.tags=[part2b, patch_tst, over_lap]'


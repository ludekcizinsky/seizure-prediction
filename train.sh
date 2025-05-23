# this is an example script on how one cun launch multiple experiments via single script
# note that it is only suing 1000 samples and 100 epochs for debugging purposes
# Graph only
# - GCN
python train.py data.subset=1000 trainer.max_epochs=100 'logger.tags=[part1, gcn]'
# - GAT
python train.py data.subset=1000 trainer.max_epochs=100 graph_module=gat 'logger.tags=[part1, gat]'

# Temporal only
# - lstm (baseline from TAs)
python train.py data.subset=1000 trainer.max_epochs=100 temporal_module=lstm graph_builder=disabled graph_module=disabled 'logger.tags=[part1, lstm]'
# - conv1d
python train.py data.subset=1000 trainer.max_epochs=100 temporal_module=conv1d graph_builder=disabled graph_module=disabled 'logger.tags=[part1, conv1d]'
# - tencoder
python train.py data.subset=1000 trainer.max_epochs=100 temporal_module=tencoder graph_builder=disabled graph_module=disabled 'logger.tags=[part1, tencoder]'

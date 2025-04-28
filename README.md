### ⚙️ Environment Setup

First, move data to your scratch directory (make sure to replace username with your actual `izar` username):

```bash
mkdir -p /scratch/izar/username/netml/
rsync -ah --info=progress2 /home/ogut/data/ /scratch/izar/username/netml/
```

Second, create virtual environment and install dependencies:

```bash
module load gcc python
virtualenv --system-site-packages venvs/netml
source venvs/netml/bin/activate
pip install -r requirements.txt
```

Third, authenticate yourself with W&B (if not done already), follow this [quick start quide](https://docs.wandb.ai/quickstart/).

### 🏋🏻‍♀️ Training
To train the model, first check [train config](configs/train.yaml) and adjust the parameters as needed. Then, run the training script:

```bash
python train.py 
```

When you want to overwrite some default config setting, you can do so as follows:

```bash
python train.py debug=True data.subset=500
```

If you are using logging to W&B, you can view the training progress in the main metrics [overview page](https://wandb.ai/ludekcizinsky/seizure-prediction/workspace?nw=whk83ic2jml).

You can also submit a job to the cluster using the provided Slurm script:

```bash
sbatch train.slurm
```

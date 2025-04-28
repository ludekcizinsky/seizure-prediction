

### Environment Setup

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
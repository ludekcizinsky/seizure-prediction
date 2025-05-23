from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
import wandb
from helpers.dataset import get_datasets
from helpers.pl_module import SeizurePredictor

def get_the_best_checkpoint(run_id, ckpt_dir):

    artifact_path = f"ludekcizinsky/seizure-prediction/model-{run_id}:best"
    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")
    artifact.download(ckpt_dir)
    print(f"Downloaded the best checkpoint to {ckpt_dir}.")

def run_eval_and_save_submission(trainer, cfg):

    print("FYI: Running evaluation and saving submission")
    test_dataset = get_datasets(cfg, split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    ckpt_dir = os.path.join(cfg.output_dir, "tmp")
    wandb_logger = trainer.logger
    run_id = wandb_logger.experiment.id
    get_the_best_checkpoint(run_id, ckpt_dir)
    pl_module = SeizurePredictor.load_from_checkpoint(f"{ckpt_dir}/model.ckpt")

    outputs = trainer.predict(pl_module, test_dataloader)
    preds = torch.cat([output["preds_batch"] for output in outputs]).cpu().numpy()
    sample_ids = []
    for output in outputs:
        sample_ids.extend(output["y_batch"])

    submission_df = pd.DataFrame({"id": sample_ids, "label": preds})
    print(f"FYI: Submission dataframe (head): {submission_df.head()}")


    subm_path = os.path.join("submissions", f"run_{run_id}.csv")
    submission_df.to_csv(subm_path, index=False)
    print(f"Kaggle submission file generated: {subm_path}")
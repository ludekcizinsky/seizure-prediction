import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os
import random

def display_metrics(num_epochs: int, metrics: dict):
    epochs = np.arange(1,num_epochs+1,1)
    fig, axs = plt.subplots(1,3, figsize=(15,10))

    sns.lineplot(x=epochs,y=metrics["train"]["loss"], ax=axs[0], label="Train", color="blue")
    sns.lineplot(x=epochs,y=metrics["eval"]["loss"], ax=axs[0], label="Val", color="orange")
    axs[0].set_xlabel("Epochs")
    axs[0].set_xlabel("Loss")

    sns.lineplot(x=epochs,y=metrics["train"]["acc"], ax=axs[1], label="Train", color="blue")
    sns.lineplot(x=epochs,y=metrics["eval"]["acc"], ax=axs[1], label="Val", color="orange")
    axs[1].set_xlabel("Epochs")
    axs[1].set_xlabel("Accuracy")

    sns.lineplot(x=epochs,y=metrics["train"]["f1"], ax=axs[2], label="Train", color="blue")
    sns.lineplot(x=epochs,y=metrics["eval"]["f1"], ax=axs[2], label="Val", color="orange")
    axs[2].set_xlabel("Epochs")
    axs[2].set_xlabel("F1-Score")

    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_everything(seed: int):
    # Python random module
    random.seed(seed)
    # Numpy random module
    np.random.seed(seed)
    # Torch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
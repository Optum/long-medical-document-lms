#!/usr/bin/env python
# coding: utf-8

"""
Given the path to a trainer state from Hugging Face transformers, plot the learning curves and learning rate.
Plots are saved to the checkpoint directory as well as the working directory for easy access.
This script assumes the number of logging steps, save steps, and eval steps is equal.
It also assumes that validation loss is used as the monitoring metric for checkpointing.
"""

# Open imports
import os
import json
import yaml
import matplotlib.pyplot as plt

# Set checkpoint path to evaluate
CHECKPOINT_PATH = (
    "/mnt/azureblobshare/models/bigbird_for_mlm_mimic50/run_0/checkpoint-2000/"
)


def main():

    # Get working dir
    working_dir = os.getcwd()

    # Load trainer state
    trainer_state_file = os.path.join(CHECKPOINT_PATH, "trainer_state.json")
    with open(trainer_state_file) as f:
        states = json.load(f)

    # Get global step and best val loss
    global_step = states["global_step"]
    best_val_loss = states["best_metric"]

    # Get log history
    steps = []
    train_loss = []
    val_loss = []
    lrs = []
    for i, state in enumerate(states["log_history"]):

        # Every other entry in the log history
        # Contains the training info
        if i % 2 == 0:
            steps.append(state["step"])
            train_loss.append(state["loss"])
            lrs.append(state["learning_rate"])
        else:
            val_loss.append(state["eval_loss"])

    # Plot train and eval loss
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(steps, train_loss, label="Train Loss")
    ax.plot(steps, val_loss, label="Val Loss")
    ax.legend(loc="best", prop={"size": 10})
    plt.title(f"Learning Curves at Checkpoint {global_step}", fontsize=10)
    plt.xlabel("Step", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_PATH, "learning_curves.png"))
    plt.savefig(os.path.join(working_dir, "learning_curves.png"))
    print(f"Best validation loss = {best_val_loss}.")

    # Plot learning rate
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(steps, lrs, label="Learning Rate")
    ax.legend(loc="best", prop={"size": 10})
    plt.title(f"Learning Rates at Checkpoint {global_step}", fontsize=10)
    plt.xlabel("Step", fontsize=10)
    plt.ylabel("Learning Rate", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_PATH, "learning_rates.png"))
    plt.savefig(os.path.join(working_dir, "learning_rates.png"))
    print(f"Current learning rate = {lrs[-1]}.")


if __name__ == "__main__":

    main()

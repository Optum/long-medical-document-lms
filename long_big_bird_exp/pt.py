#!/usr/bin/env python
# coding: utf-8

"""
Continue pretraining an LM for masked language modeling
"""

# Open imports
import os
import glob
import time
import yaml
import torch
import random
import transformers
import logging
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    BigBirdTokenizerFast,
    BigBirdForMaskedLM,
    BigBirdConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)

# Project imports
from utils import check_empty_count_gpus, create_current_run, create_log_dir, get_bigbird_config

# Load run parameters
with open("params.yml", "r") as stream:
    PARAMS = yaml.safe_load(stream)

def get_model_and_tokenizer():
    """
    Define function to get model used in "Extend and Explain."
    """

    # Load model and tokenizer
    tokenizer = BigBirdTokenizerFast(
        tokenizer_file=PARAMS["lm_path"] + "bpe_tokenizer.json"
    )
    model = BigBirdForMaskedLM(
        config=BigBirdConfig(**get_bigbird_config(tokenizer, clf=False)),
    )

    # Set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load model checkpoint
    checkpoint = torch.load(PARAMS["lm_path"] + "bigbird_mlm.ckpt", map_location=device)

    # Update layer names for non-DDP setting
    # Layers are stored as 'model.layer_name' so we remove 'model.'
    new_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        name = k[6:]
        new_state_dict[name] = v

    # Apply model weights from checkpoint to initialized model
    model.load_state_dict(new_state_dict, strict=True)

    return model, tokenizer

def main():

    # Define logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Log CUDNN, PyTorch, and Transformers versions
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")

    # Check, empty, and count GPUs
    check_empty_count_gpus(logger=logger)

    # Create run directory
    current_run_dir = create_current_run(
        save_path=PARAMS["output_path"], params=PARAMS, logger=logger
    )
    logger.info(f"Created run directory: {current_run_dir}.")

    # Create logging dir
    logging_dir = create_log_dir(current_run_dir)

    # Set run name
    run_name = current_run_dir.split("/")[-1]
    logger.info(f"Starting run {run_name}...")

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # Load Data
    X_train = np.load(PARAMS["base_path"] + "X_train.npy", allow_pickle=True)
    X_val = np.load(PARAMS["base_path"] + "X_val.npy", allow_pickle=True)

    # Build Hugging Face Dataset
    dd = DatasetDict(
        {
            "train": Dataset.from_dict({"text": X_train.tolist()}),
            "val": Dataset.from_dict({"text": X_val.tolist()}),
        }
    )

    # Define Function to Tokenize Text
    def tokenize_function(batch):

        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=PARAMS["max_seq_len"],
        )

    # Tokenize Text
    dd["train"] = (
        dd["train"]
        .map(
            tokenize_function,
            batched=True,
            batch_size=PARAMS["per_device_train_batch_size"],
            remove_columns=["text"],
        )
        .with_format("torch")
    )
    dd["val"] = (
        dd["val"]
        .map(
            tokenize_function,
            batched=True,
            batch_size=PARAMS["per_device_eval_batch_size"],
            remove_columns=["text"],
        )
        .with_format("torch")
    )

    # Log data shapes
    logger.info(f"Train samples: {len(dd['train'])}.")
    logger.info(f"Val samples: {len(dd['val'])}.")

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=PARAMS["mlm_prob"],
        return_tensors="pt",
    )

    # Define early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=PARAMS["early_stopping_patience"]
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=current_run_dir,
        max_steps=PARAMS["max_steps"],
        per_device_train_batch_size=PARAMS["per_device_train_batch_size"],
        gradient_accumulation_steps=PARAMS["accumulation_steps"],
        per_device_eval_batch_size=PARAMS["per_device_eval_batch_size"],
        metric_for_best_model=PARAMS["metric_for_best_model"],
        eval_steps=PARAMS["eval_steps"],
        save_steps=PARAMS["save_steps"],
        logging_steps=PARAMS["logging_steps"],
        evaluation_strategy=PARAMS["evaluation_strategy"],
        save_strategy=PARAMS["save_strategy"],
        lr_scheduler_type=PARAMS["lr_scheduler_type"],
        warmup_steps=PARAMS["warmup_steps"],
        weight_decay=PARAMS["weight_decay"],
        learning_rate=PARAMS["learning_rate"],
        fp16=PARAMS["fp16"],
        fp16_full_eval=PARAMS["fp16_eval"],
        seed=PARAMS["seed"],
        adam_beta1=PARAMS["adam_beta1"],
        adam_beta2=PARAMS["adam_beta2"],
        adam_epsilon=PARAMS["adam_epsilon"],
        sharded_ddp=PARAMS["sharded_ddp"],
        optim="adamw_torch",
        disable_tqdm=PARAMS["disable_tqdm"],
        load_best_model_at_end=True,
        logging_dir=logging_dir,
        dataloader_num_workers=PARAMS["dataloader_num_workers"],
        run_name=run_name,
        prediction_loss_only=True,
        eval_accumulation_steps=PARAMS["eval_accumulation_steps"],
        gradient_checkpointing=PARAMS["gradient_checkpointing"],
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dd["train"],
        eval_dataset=dd["val"],
        data_collator=data_collator,
        callbacks=[early_stopping],
    )

    # Start training timer
    start_time = time.time()

    # Start from model provided above and new training parameters defined above
    if not PARAMS["resume_training"]:

        # Train model
        trainer.train()

    # Resume using model and training parameters defined in checkpoint
    else:

        # Train model
        trainer.train(PARAMS["checkpoint"])

    # Log training time
    end_time = time.time()
    execution_time_hours = round((end_time - start_time) / 3600.0, 2)
    logger.info(f"Training took {execution_time_hours} hours.")

    # Save best model
    trainer.model.save_pretrained(
        os.path.join(current_run_dir, f'{PARAMS["model_name"]}_model')
    )
    tokenizer.save_pretrained(
        os.path.join(current_run_dir, f'{PARAMS["model_name"]}_tokenizer')
    )

    # Evaluate model
    trainer.evaluate()


if __name__ == "__main__":

    main()

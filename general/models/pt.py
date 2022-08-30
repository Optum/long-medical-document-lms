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
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)

# Project imports
from utils import check_empty_count_gpus, create_current_run, create_log_dir


def main():

    # Load run parameters
    with open("pt_params.yml", "r") as stream:
        PARAMS = yaml.safe_load(stream)

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

    # Load LM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PARAMS["lm_path"])

    # Load data and optionally set offline mode
    if PARAMS["offline"]:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        dataset = load_from_disk(PARAMS["data_path"])
    else:
        dataset = load_dataset(PARAMS["data_path"])

    # Optionally rename columns
    if PARAMS["rename_text_col"]:
        dataset = dataset.rename_column(PARAMS["text_col_name"], "text")

    # Add validation split if non-existent
    if PARAMS["create_val_split"]:
        dataset["train"] = dataset["train"].shuffle()
        split_dataset = dataset["train"].train_test_split(
            test_size=PARAMS["val_frac_to_create"]
        )
        dataset["train"], dataset["val"] = split_dataset["train"], split_dataset["test"]

    # Log data shapes
    logger.info(f"Train samples: {len(dataset['train'])}.")
    logger.info(f"Val samples: {len(dataset['val'])}.")
    logger.info(f"Test samples: {len(dataset['test'])}.")

    # Define Function to Tokenize Text
    def tokenize_function(batch):

        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=PARAMS["max_seq_len"],
        )

    # Define transformation to tokenize data in batches
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "label"],  # remove labels used for fine-tuning
        batch_size=PARAMS["per_device_train_batch_size"],
    ).with_format("torch")

    # Define data collator which will create MLM labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=PARAMS["mlm_prob"],
        return_tensors="pt",
    )

    # Load MLM from checkpoint
    model = AutoModelForMaskedLM.from_pretrained(
        PARAMS["lm_path"],
        return_dict=True,
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
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
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

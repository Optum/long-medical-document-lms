#!/usr/bin/env python
# coding: utf-8

"""
Explain LM Predictions with SOC
Explain important multi-token text blocks from text classifier using Sampling and Occlusion (SOC) from [Jin et al.](https://arxiv.org/pdf/1911.06194.pdf)
"""

# Open imports
import os
import yaml
import time
import yaml
import torch
import shutil
import pickle
import logging
import transformers
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
)

# Project imports
from utils import check_average_precision
from soc import predict_with_soc_algo, post_process_and_save_soc_results


def main():

    # Load Run Parameters
    with open("params.yml", "r") as stream:
        PARAMS = yaml.safe_load(stream)

    # Define Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Log CUDNN, PyTorch, and Transformers versions
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")

    # Output path
    output_path = (
        f"./soc_results_{PARAMS['data']}/"  # will be deleted if it already exists
    )

    # Create Directory to Save Results
    # This script is for demo purposes and **will delete** the `output_path` directory if it exists on each new run.
    # Save important results elsewhere.
    if os.path.exists(output_path) and os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Configure Device and Empty GPU Cache
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load Data, Tokenizer, and Model
    if PARAMS["offline"]:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        dataset = load_from_disk(PARAMS["data"])
    else:
        dataset = load_dataset(PARAMS["data"])
    tokenizer = AutoTokenizer.from_pretrained(PARAMS["tokenizer"])
    model = AutoModelForSequenceClassification.from_pretrained(PARAMS["model"])
    lm = AutoModelForMaskedLM.from_pretrained(PARAMS["lm"])

    # Tokenize Test Data
    def tokenize_function(batch):
        """Tokenize batch by padding to max length"""

        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=PARAMS["max_seq_len"],
        )

    # Tokenize Text
    # Code runs on the test data split by default
    dataset["test"] = dataset["test"].map(
        tokenize_function, batched=True, batch_size=PARAMS["batch_size"]
    )

    # Take a Random Sample of the Test Data
    sample_data = dataset["test"].shuffle()[0 : PARAMS["num_sample"]]

    # Check Average Precision of Classifier
    # To Do: Add CIs to prediction
    check_average_precision(
        model=model,
        data=sample_data,
        device=device,
        class_strategy=PARAMS["class_strategy"],
        average="macro",
    )

    # Start timer
    start_time = time.time()

    # Run and Clock SOC
    times = []
    all_results = []
    all_num_inferences = []
    for s, (doc_input_ids, doc_y) in enumerate(
        zip(sample_data["input_ids"], sample_data["label"])
    ):

        # Indicate sample number
        logger.info(f"Running SOC for sample {s} of {PARAMS['num_sample']}...")

        # Generate predictions with masked texts
        results, num_inferences = predict_with_soc_algo(
            model=model,
            lm=lm,
            input_ids=doc_input_ids,
            doc_y=doc_y,
            samples=PARAMS["N"],
            n_gram_range=PARAMS["K"],
            radius=PARAMS["K"],
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
            idx2label=PARAMS["idx2label"],
            vocab_size=len(tokenizer),
            print_every=PARAMS["print_every"],
            debug=PARAMS["debug"],
            device=device,
            max_seq_len=PARAMS["max_seq_len"],
            class_strategy=PARAMS["class_strategy"],
        )
        all_results.append(results)
        all_num_inferences.append(num_inferences)

        # Compute time to run SOC on one doc
        doc_time = time.time()
        times.append(doc_time)

    # End timer
    end_time = time.time()

    # Compute results
    time_hours = (end_time - start_time) / 3600.0
    total_num_inferences = sum(all_num_inferences)
    time_per_doc = time_hours / PARAMS["num_sample"]

    # View Runtime Results for SOC
    logger.info(
        f"Simulation took {time_hours} hours for {PARAMS['num_sample']} samples with {PARAMS['N']} iterations."
    )
    logger.info(f"Ran a total of {total_num_inferences} model inferences.")
    logger.info(
        f"Time per doc (hours) averaged across {PARAMS['num_sample']} docs: {time_per_doc}."
    )

    # Start timer
    start_time = time.time()

    # Post-Process and Save Results
    post_process_and_save_soc_results(
        model=model,
        all_results=all_results,
        all_input_ids=sample_data["input_ids"],
        all_labels=sample_data["label"],
        times=times,
        device=device,
        tokenizer=tokenizer,
        num_sample=PARAMS["num_sample"],
        max_seq_len=PARAMS["max_seq_len"],
        class_strategy=PARAMS["class_strategy"],
        idx2label=PARAMS["idx2label"],
        output_path=output_path,
        n=PARAMS["N"],
        k=PARAMS["K"],
        m=PARAMS["M"],
    )

    # End timer
    end_time = time.time()

    # Compute results
    time_hours = (end_time - start_time) / 3600.0

    # View Runtime Results for Post-Processing
    logger.info(
        f"Post-processing took {time_hours} hours for {PARAMS['num_sample']} samples."
    )


if __name__ == "__main__":

    main()

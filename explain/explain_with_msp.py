#!/usr/bin/env python
# coding: utf-8

"""
Explain LM Predictions with Masked Sampling Procedure
Explain important multi-token text blocks from text classifier using Masked Sampling Procedure (MSP).
"""

# Open imports
import os
import yaml
import time
import yaml
import json
import torch
import shutil
import pickle
import logging
import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Project imports
from utils import check_average_precision
from msp import predict_with_masked_texts, post_process_and_save_msp_results


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

    # Define output path according to offline or online data
    if PARAMS["offline"]:
        output_path = f"./msp_results_{PARAMS['data'].rstrip('/').split('/')[-1]}/"
    else:
        output_path = f"./msp_results_{PARAMS['data']}/"

    # Create output directory
    # This script is for demo purposes and **will delete** the `output_path` directory if it exists on each new run.
    # Save important results elsewhere.
    if os.path.exists(output_path) and os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Load Data
    if PARAMS["offline"]:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        dataset = load_from_disk(PARAMS["data"])[PARAMS["split_name"]]
    elif PARAMS["is_csv"]:
        dataset = load_dataset('csv', data_files={"test" : PARAMS["data"]}, delimiter=",")[PARAMS["split_name"]]
    else:
        dataset = load_dataset(PARAMS["data"])[PARAMS["split_name"]]

    # Optionally one hot encode labels if not already
    if PARAMS["one_hot"] and PARAMS["class_strategy"] == "multi_class":
        label_enum = {k:j for j, k in enumerate(set(dataset['label']))}
        dataset = dataset.remove_columns("label").add_column(
            "label", [[1.0 if label_enum[row_label]==i else 0.0 for i in range(len(label_enum))] for row_label in dataset['label']]
        )

    # Log dataset print summary
    logger.info(f"Dataset: {dataset}.")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(PARAMS["tokenizer"])
    model = AutoModelForSequenceClassification.from_pretrained(PARAMS["model"])

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
    dataset = dataset.map(
        tokenize_function, batched=True, batch_size=PARAMS["batch_size"]
    )

    # Optionally sample the data
    if PARAMS["num_sample"] > 0 and not PARAMS["fixed_sample"]:
        sample_data = dataset.shuffle()[0 : PARAMS["num_sample"]]
    elif PARAMS["num_sample"] == -1:
        sample_data = dataset[0 : len(dataset)]
    elif PARAMS["num_sample"] > 0 and PARAMS["fixed_sample"]:
        sample_data = dataset[0 : PARAMS["num_sample"]]
        idx2id = {i: uid for i, uid in enumerate(sample_data['id'])}
        logger.info(f"Indices to IDs: {idx2id}.")
        with open("./sample_idx2id.json", 'w') as f:
            json.dump(idx2id, f)
    else:
        raise ValueError(f"Unexpected params {PARAMS['num_sample']}, {PARAMS['fixed_sample']} provided as num_sample and fixed_sample.")

    # Check Average Precision of Classifier
    logger.info("Computing average precision on labels...")
    ap_out_str = check_average_precision(
        model=model,
        data=sample_data,
        class_strategy=PARAMS["class_strategy"],
        average="micro",
        batch_size=PARAMS["batch_size"]
    )
    logger.info(ap_out_str)

    # Start timer
    start_time = time.time()

    # Run MSP
    times = []
    all_results = []
    for s, (doc_input_ids, doc_text) in enumerate(
        zip(sample_data["input_ids"], sample_data["text"])
    ):

        # Indicate sample number
        logger.info(f"Running MSP for sample {s} of {PARAMS['num_sample']}...")

        # Generate predictions with masked texts
        results = predict_with_masked_texts(
            model=model,
            input_ids=doc_input_ids,
            text=doc_text,
            n=PARAMS["N"],
            k=PARAMS["K"],
            p=PARAMS["P"],
            idx2label=PARAMS["idx2label"],
            print_every=PARAMS["print_every"],
            debug=PARAMS["debug"],
            max_seq_len=PARAMS["max_seq_len"],
            class_strategy=PARAMS["class_strategy"],
            tokenizer=tokenizer,
            by_sent_segments=PARAMS["by_sent_segments"],
            batch_size=PARAMS["batch_size"],
        )
        all_results.append(results)

        results["indices_len"] = results["masked_text_indices"].apply(lambda x: len(x))
        results["tokens_len"] = results["masked_text_tokens"].apply(lambda x: len(x))

        # Compute time to run MSP on one doc
        doc_time = time.time()
        times.append(doc_time)

    # End timer
    end_time = time.time()

    # Compute results
    time_hours = (end_time - start_time) / 3600.0
    time_per_doc = time_hours / PARAMS["num_sample"]

    # Print results
    logger.info(
        f"Simulation took {time_hours} hours for {PARAMS['num_sample']} samples with {PARAMS['N']} iterations."
    )
    logger.info(
        f"Ran a total of {PARAMS['num_sample'] * PARAMS['N']} model inferences."
    )
    logger.info(
        f"Time per doc (hours) averaged across {PARAMS['num_sample']} docs: {time_per_doc}."
    )

    # Start timer
    start_time = time.time()

    # Post-Process and Save Results
    post_process_and_save_msp_results(
        model=model,
        all_results=all_results,
        all_input_ids=sample_data["input_ids"],
        all_labels=sample_data["label"],
        times=times,
        tokenizer=tokenizer,
        num_sample=PARAMS["num_sample"],
        max_seq_len=PARAMS["max_seq_len"],
        class_strategy=PARAMS["class_strategy"],
        idx2label=PARAMS["idx2label"],
        num_bootstrap=PARAMS["num_bootstrap"],
        output_path=output_path,
        n=PARAMS["N"],
        k=PARAMS["K"],
        p=PARAMS["P"],
        m=PARAMS["M"],
        by_sent_segments=PARAMS["by_sent_segments"],
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

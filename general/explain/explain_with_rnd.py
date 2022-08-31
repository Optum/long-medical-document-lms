#!/usr/bin/env python
# coding: utf-8

"""
Explain LM Predictions with RND
Explain important multi-token text blocks from text classifier using a random algorithm.
"""

# Open imports
import os
import yaml
import time
import torch
import shutil
import pickle
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Project imports
from utils import check_average_precision
from rnd import predict_with_rnd_algo, post_process_and_save_rnd_results

# Model parameters
# Provide local paths to a classifier checkpoint after fine-tuning
# Or point to a model from the HuggingFace Model Hub
MODEL = "/mnt/azureblobshare/models/bigbird_imdb/run_9/checkpoint-1000/"
TOKENIZER = "/mnt/azureblobshare/models/bigbird-roberta-large/"
MAX_SEQ_LEN = 4096
CLASS_STRATEGY = "multi_class"  # one of ['binary', 'multi_label', 'multi_class']

# Data parameters
DATA = "imdb"  # one of ['sst2', 'yelp_polarity', 'mimic50', 'imdb']
BATCH_SIZE = 32  # for tokenization only - To Do: collect sequences in batches when running inference
NUM_SAMPLE = 500  # number of documents for which to generate explainations
IDX2LABEL = {
    0: "Negative Sentiment",
    1: "Positive Sentiment",
}  # for binary, the 0 index should be the negative label

# Define label mapping
if DATA == "mimic50":
    with open(
        "/mnt/azureblobshare/nlp-modernisation/database/BYOL-mimic50_exp9/model_artifacts/goat_label2idx.pkl",
        "rb",
    ) as f:
        IDX2LABEL = {v: k for k, v in pickle.load(f).items()}
else:
    IDX2LABEL = {
        0: "Negative Sentiment",
        1: "Positive Sentiment",
    }  # provide dictionary of class indices and label names

# Set run parameters for experiment
K = 5  # subwords in a masked block of text (size of phrase and also radius)
N = 100  # number of samples to take
M = 5  # show the M most important blocks which led to the greatest difference in predicted probability of the given label

# Output path
OUTPUT_PATH = f"./rnd_results_{DATA}/"  # will be deleted if it already exists


# Create Directory to Save Results
# This script is for demo purposes and **will delete** the `OUTPUT_PATH` directory if it exists on each new run.
# Save important results elsewhere.
if os.path.exists(OUTPUT_PATH) and os.path.isdir(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH)

# Configure Device and Empty GPU Cache
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Data, Tokenizer, and Model
dataset = load_from_disk(
    f"/mnt/azureblobshare/hf_datasets/{DATA}.hf"
)  # use load_dataset(DATA) to pull from HuggingFace Datasets
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Tokenize Test Data
def tokenize_function(batch):
    """Tokenize batch by padding to max length"""

    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LEN
    )


# Tokenize Text
dataset["test"] = dataset["test"].map(
    tokenize_function, batched=True, batch_size=BATCH_SIZE
)

# Take a Random Sample of the Test Data
sample_data = dataset["test"].shuffle()[0:NUM_SAMPLE]

# Check Average Precision of Classifier
# To Do: Add CIs to prediction
check_average_precision(
    model=model,
    data=sample_data,
    device=device,
    class_strategy=CLASS_STRATEGY,
    average="macro",
)

# Start timer
start_time = time.time()

# Run and Clock RND
times = []
all_results = []
for s, (doc_input_ids, doc_y) in enumerate(
    zip(sample_data["input_ids"], sample_data["label"])
):

    # Indicate sample number
    print(f"Running RND for sample {s} of {NUM_SAMPLE}...")

    # Generate predictions with masked texts
    results = predict_with_rnd_algo(
        input_ids=doc_input_ids,
        doc_y=doc_y,
        n=N,
        k=K,
        m=M,
        idx2label=IDX2LABEL,
        max_seq_len=MAX_SEQ_LEN,
        class_strategy=CLASS_STRATEGY,
    )

    all_results.append(results)

    # Compute time to run RND on one doc
    doc_time = time.time()
    times.append(doc_time)

# End timer
end_time = time.time()

# Compute results
time_hours = (end_time - start_time) / 3600.0
time_per_doc = time_hours / NUM_SAMPLE

# View Runtime Results for RND
print(
    f"Simulation took {time_hours} hours for {NUM_SAMPLE} samples with {N} iterations."
)
print(f"Time per doc (hours) averaged across {NUM_SAMPLE} docs: {time_per_doc}.")

# Start timer
start_time = time.time()

# Post-Process and Save Results
post_process_and_save_rnd_results(
    model=model,
    all_results=all_results,
    all_input_ids=sample_data["input_ids"],
    all_labels=sample_data["label"],
    times=times,
    tokenizer=tokenizer,
    device=device,
    num_sample=NUM_SAMPLE,
    max_seq_len=MAX_SEQ_LEN,
    class_strategy=CLASS_STRATEGY,
    idx2label=IDX2LABEL,
    output_path=OUTPUT_PATH,
    n=N,
    k=K,
    m=M,
)

# End timer
end_time = time.time()

# Compute results
time_hours = (end_time - start_time) / 3600.0

# View Runtime Results for Post-Processing
print(f"Post-processing took {time_hours} hours for {NUM_SAMPLE} samples.")

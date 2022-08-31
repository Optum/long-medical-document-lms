"""
Script: Identify Text Blocks used by Big Bird Language Model to Classify Diagnoses from MIMIC III Discharge Summaries with Sampling and Occlusion Algorithm

- 05-26-22
- Joel Stremmel

About:

This script examines the difference in probabilities of each positive label for a random sample of the test set of the MIMIC 50 data between
an unmasked sample (full discharge summary) and the averaged probabilities of randomly masked samples.
To compute the probabilities of each label for masked samples, we use the SOC algorithm from Jin et al. (https://arxiv.org/pdf/1911.06194.pdf)
based on: https://github.com/INK-USC/hierarchical-explanation-neural-sequence-models.

Output:

The script saves text blocks generated via the SOC algorithm with statistics from the experiment and a row ID.
It saves just the text blocks and diagnoses with a row ID, so that a physician can review the text blocks and
determine which blocks are informative for predicting the given diagnosis.
The results of this clinical review can be joined via row ID to the saved statistics.
"""

# Imports
import os
import yaml
import time
import torch
import random
import itertools
import warnings
import numpy as np
import pandas as pd
import pickle5 as pickle
from transformers import (
    BigBirdConfig,
    BigBirdForMaskedLM,
    BigBirdForSequenceClassification,
    BigBirdTokenizerFast,
)
from utils import predict_on_sample_with_clf_model, check_empty_count_gpus
from soc import predict_with_soc_algo

# Ignore warnings to make the output easy to read
warnings.filterwarnings("ignore")

# Toggle debug mode
DEBUG = False
if DEBUG:
    print("-------------Running in DEBUG mode-------------")
else:
    print("---------- 🔥🔥🔥 Doing it live 🔥🔥🔥 ----------")

# Set run parameters for experiment
K = 10  # subwords in a masked block of text (size of phrase and also radius)
N = 100  # number of samples to take
M = 5  # show the M most important blocks which led to the greatest difference in predicted probability of the given label
R = 3  # ensure sample has at least R positive labels before running the experiment

# Set some quality of life parameters
PRINT_EVERY = 10  # always print progress after this many iterations
LABEL_THRESHOLD = 3  # only run iterations if there are at least this many positive labels for the sample

# Load the Big Bird Sequence Classifier trained with these parameters and use the Mimic 50 data at the base path
PARAMS_PATH = "params.yml"
BASE_PATH = (
    "/mnt/azureblobshare/nlp-modernisation/database/BYOL-mimic50_exp9/model_artifacts/"
)
ICD9_DESC = "/mnt/azureblobshare/D_ICD_DIAGNOSES.csv"

# Save sample outputs
OUTPUT_PATH = f"/mnt/azureblobshare/soc_pt_outputs/"

# Empty Cache and check number of GPUs
check_empty_count_gpus()

# Load pipeline parameters
with open(PARAMS_PATH, "r") as stream:
    PARAMS = yaml.safe_load(stream)

# Get label2idx mapping and reverse mapping
with open(BASE_PATH + "goat_label2idx.pkl", "rb") as f:
    label2idx = pickle.load(f)
idx2label = {v: k for k, v in label2idx.items()}

# Load text data
X_test_path = BASE_PATH + "X_test.npy"
y_test_path = BASE_PATH + "y_test.npy"
X_test = np.load(X_test_path, allow_pickle=True)
y_test = np.load(y_test_path, allow_pickle=True)

# Load tokenizer and check mask token
tokenizer = BigBirdTokenizerFast(
    tokenizer_file=PARAMS["tokenizer_path"] + "bpe_tokenizer.json"
)
print(f"Mask token: {tokenizer.mask_token}.")
print(f"Mask token ID: {tokenizer.mask_token_id}.")

# Tokenize the text in the test set
batch_encoding_plus = tokenizer.batch_encode_plus(
    X_test.tolist(),
    padding="max_length",
    truncation=True,
    max_length=PARAMS["max_seq_len"],
    return_attention_mask=True,
)
tokenized_texts = batch_encoding_plus["input_ids"]

# Take a random sample from the tokenized test set
sample = random.randint(0, len(tokenized_texts))
print(f"Sampled record {sample} from test set.")

# Get sample text and labels
sample_text = tokenized_texts[sample]
sample_y = y_test[sample]

# Only proceed if the sample is positive for at least a few labels
assert sum(sample_y) >= LABEL_THRESHOLD, "Selected a sample without many labels."

# View the discharge summary text of the sample
tokenizer.decode(sample_text)

# View the ICD 9 labels for the sample
descs = pd.read_csv(ICD9_DESC)[["icd9_code", "long_title"]]
sample_icds = pd.DataFrame(
    {"icd9_code": [x.replace(".", "") for x in label2idx.keys()], "label": sample_y}
)
sample_icds_descs = pd.merge(sample_icds, descs, on="icd9_code", how="left")

# Create and ICD to description mapping
icd2desc = dict(
    list(
        zip(
            sample_icds_descs["icd9_code"].tolist(),
            sample_icds_descs["long_title"].tolist(),
        )
    )
)

# Run and clock SOC iterations
# Start timer
start_time = time.time()

# Generate predictions with masked texts
results, num_inferences = predict_with_soc_algo(
    sample_text=sample_text,
    sample_y=sample_y,
    samples=N,
    n_gram_range=K,
    radius=K,
    mask_token_id=tokenizer.mask_token_id,
    pad_token_id=tokenizer.pad_token_id,
    idx2label=idx2label,
    vocab_size=len(tokenizer),
    print_every=PRINT_EVERY,
    debug=DEBUG,
    params=PARAMS,
)

# End timer
end_time = time.time()

# Compute results
time_hours = round((end_time - start_time) / 3600.0, 2)
preds_per_hour = round(num_inferences / time_hours, 2)

# Print results
print(f"SOC algo took {time_hours} hours.")
print(f"Ran a total of {num_inferences} model inferences.")
print(f"Inferences per hour: {preds_per_hour}.")

# Predict on the phrase text with LM with CLF head
prob = predict_on_sample_with_clf_model(sample_text, params=PARAMS)

# Build dataframe of results consisting of the average scores for each label from each round of sampling for a given phrase
base_results = pd.DataFrame(
    [prob], columns=[idx2label[idx] for idx in range(len(sample_y))]
)

# Select columns for which true label is present
base_results = base_results.iloc[:, np.where(sample_y == 1)[0]]

# Compute difference in base score for each phrase
for col in base_results.columns:
    results["diff_" + col] = base_results[col].values[0] - results[col]

# Iterate through results to compute and save top M blocks
dfs = []
for i, col in enumerate(results.columns):

    # Only process code columns for which we computed a diff
    if "diff_" not in col:
        continue

    # Get text block, index, and score diff for code column
    final_df = results[["text_tokens", "text_indices"]]
    final_df["avg_score_diff"] = results[col]
    final_df["text_block"] = results["text_tokens"].apply(lambda x: tokenizer.decode(x))
    final_df = final_df.drop(["text_tokens", "text_indices"], 1)

    # Get code name from column name
    code = col.replace("diff_", "")

    # Get code description and add code and description to dataframe
    desc = icd2desc[code.replace(".", "")]
    final_df["icd_code"] = code
    final_df["icd_description"] = desc

    # Only process code columns for which we have descriptions
    # If the description is not equal to itself, the description is a NaN value
    if desc != desc:
        print(f"Code: {code} doesn't have a valid description.")
        continue
    else:
        print(f"Processing code {code} with description {desc}.")

    # Sort dataframe and add info about SOC algo run
    top_m_df = final_df.sort_values(by="avg_score_diff", ascending=False).head(M)
    top_m_df["sample"] = sample
    top_m_df["draws"] = N
    top_m_df["K"] = K
    top_m_df["P"] = "SOC"
    top_m_df["runtime_hours"] = time_hours

    # Add to all blocks dataframe
    dfs.append(top_m_df)

# Combine output dataframes
all_blocks_df = pd.concat(dfs)

# Add random row IDs
row_ids = np.arange(len(all_blocks_df))
np.random.shuffle(row_ids)
all_blocks_df["row_id"] = row_ids

# View output dataframe
print("All info dataframe - top 100 records: ")
print(all_blocks_df.head(100))

# Save output dataframe with all experiment information
all_blocks_df.to_csv(
    OUTPUT_PATH + f"sample_{sample}_all_info_{time_hours}_hours.csv", index=False
)

# View output dataframe with hidden information
blind_all_blocks_df = all_blocks_df[
    ["row_id", "icd_code", "icd_description", "text_block"]
]
blind_all_blocks_df["informative_0_or_1"] = "-"
print("Blinded dataframe - top 100 records: ")
print(blind_all_blocks_df.head(100))

# Save output dataframe with hidden information
blind_all_blocks_df.to_csv(
    OUTPUT_PATH + f"sample_{sample}_exp_data_{time_hours}_hours.csv", index=False
)

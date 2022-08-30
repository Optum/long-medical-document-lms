"""
Script: Identify Text Blocks used by Big Bird Language Model to Classify Diagnoses from MIMIC III Discharge Summaries with Masked Sampling Procedure

- 05-26-22
- Joel Stremmel

About:

This script examines the difference in probabilities of each positive label for a random sample of the test set of the MIMIC 50 data between
an unmasked sample (full discharge summary) and the averaged probabilities of randomly masked samples.
To compute the probabilities of each label for masked samples, we run N iterations
where each block of K subwords in the sample is masked with probability P.
The difference between the probabilities of each label for unmasked and masked samples represents
the importance of the block of K subwords in classifying the discharge summary as having the given label,
which, for this dataset, can be any of 50 ICD 9 codes.
The same sample can have many ICD 9 labels, and we compute the M most import blocks of size K
for all true labels for the sampled discharge summary.


Output:

The script saves text blocks generated via the masking procedure with statistics from the experiment and a row ID.
It also saves randomly sampled text blocks.  The script saves just the text blocks and diagnoses with a row ID,
so that a physician can review the text blocks and determine which blocks are informative for predicting the given diagnosis.
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
from msp import predict_with_masked_texts, compute_p_val_from_bootstrap

# Ignore warnings to make the output easy to read
warnings.filterwarnings("ignore")

# Toggle debug mode
DEBUG = False
if DEBUG:
    print("-------------Running in DEBUG mode-------------")
else:
    print("---------- 🔥🔥🔥 Doing it live 🔥🔥🔥 ----------")

# Add random text blocks for blind clinical review of block importances
ADD_RANDOM = True

# Set run parameters for experiment
K = 10  # subwords in a masked block of text
P = 0.1  # probability that a block of size K is masked - set such that expected draws for a single block is 100
N = 1000  # number of iterations to run - set such that expected draws for a single block is 100 by computing 100 / P
M = 5  # show the M most important blocks which led to the greatest difference in predicted probability of the given label
R = 3  # ensure sample has at least R positive labels before running the experiment
NUM_BOOTSTRAP = 10000  # Set run parameters to compute p values

# Set some quality of life parameters
PRINT_EVERY = 100  # always print progress after this many iterations
LABEL_THRESHOLD = 3  # only run iterations if there are at least this many positive labels for the sample

# Load the Big Bird Sequence Classifier trained with these parameters and use the Mimic 50 data at the base path
PARAMS_PATH = "/mnt/azureblobshare/models/F/epoch_02599_mighty_island_xhygwy0j/icy_camel_5kjjknp2/params.yml"
BASE_PATH = (
    "/mnt/azureblobshare/nlp-modernisation/database/BYOL-mimic50_exp9/model_artifacts/"
)
ICD9_DESC = "/mnt/azureblobshare/D_ICD_DIAGNOSES.csv"

# Save sample outputs
OUTPUT_PATH = "/mnt/azureblobshare/msp_and_rand_outputs0.1/"

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

# Run and clock masking iterations
# Start timer
start_time = time.time()

# Generate predictions with masked texts
results = predict_with_masked_texts(
    sample_text=sample_text,
    n=N,
    k=K,
    p=P,
    mask_token_id=tokenizer.mask_token_id,
    idx2label=idx2label,
    print_every=PRINT_EVERY,
    debug=DEBUG,
    params=PARAMS,
)

# End timer
end_time = time.time()

# Compute results
time_hours = round((end_time - start_time) / 3600.0, 2)
preds_per_hour = round(N / time_hours, 2)

# Print results
print(f"Simulation took {time_hours} hours with {N} iterations.")
print(f"Ran a total of {N} model inferences.")
print(f"Inferences per hour: {preds_per_hour}.")

# Explode results to create a single block per dataframe row with some blocks represented many times according to the number of times the block was selected
exploded_results = results.explode(["masked_text_indices", "masked_text_tokens"])

# Get predictions on sample with no masking
full_yhat = predict_on_sample_with_clf_model(sample_text, params=PARAMS)

# Combine predictions, labels, descriptions and view positive labels
labels, ys, yhats, descs = [], [], [], []
for i in range(len(sample_y)):
    labels.append(idx2label[i])
    ys.append(sample_y[i])
    yhats.append(full_yhat[i])
    descs.append(icd2desc[idx2label[i].replace(".", "")])
full_preds = pd.DataFrame(
    {"y": ys, "yhat": yhats, "label": labels, "description": descs}
)
pos_full_preds = full_preds[full_preds["y"] == 1].sort_values(
    by="yhat", ascending=False
)

# Drop labels with missing descriptions
pos_full_preds = pos_full_preds[pos_full_preds["description"].notnull()]

# Save all the scores from the masking procedure for all the positive codes
codes = pos_full_preds["label"].tolist()
all_scores = {code: results[code].tolist() for code in codes}

# For the positive labels, we compute which blocks of text account for the greatest difference in the predicted probability of each label when randomly masked
dfs = []
for i, code in enumerate(codes):

    # Get code description
    desc = icd2desc[code.replace(".", "")]

    # Compute average differences from unmasked prediction for masked blocks for code
    print(
        f"Computing average differences between probability of '{code}: {desc}' for unmasked and masked text..."
    )

    # Compute the difference in probabilities between the unmasked prediction and the masked predictions
    code_yhat = pos_full_preds[pos_full_preds["label"] == code]["yhat"].item()
    code_df = exploded_results[[code, "masked_text_indices", "masked_text_tokens"]]
    code_df["score_diff"] = code_yhat - code_df[code]

    # Find the token blocks with the highest average score difference for the selected code
    sorted_df = (
        code_df.groupby(["masked_text_indices"])
        .agg(
            draws=("masked_text_indices", "size"), avg_score_diff=("score_diff", "mean")
        )
        .reset_index()
    )

    # Get the English text string of each token associated with the blocks in the sorted dataframe
    final_df = pd.merge(
        sorted_df,
        exploded_results[["masked_text_indices", "masked_text_tokens"]],
        on="masked_text_indices",
        how="inner",
    )

    # Decode the tokens to display the text
    final_df["text_block"] = final_df["masked_text_tokens"].apply(
        lambda x: tokenizer.decode(x)
    )

    # Drop duplicates
    final_df = final_df[["draws", "avg_score_diff", "text_block"]].drop_duplicates()

    # Add code and description
    final_df["icd_code"] = code
    final_df["icd_description"] = desc

    # Remove padding
    all_pad_block = (tokenizer.pad_token + " ") * (K - 1) + tokenizer.pad_token
    final_df = final_df[final_df["text_block"] != all_pad_block]

    # Build the top M dataframe based on the masking procedure and add p values
    top_m_df = final_df.sort_values(by="avg_score_diff", ascending=False).head(M)
    top_m_df["p_val"] = top_m_df.apply(
        lambda record: compute_p_val_from_bootstrap(
            draws=record.draws,
            avg_score_diff=record.avg_score_diff,
            code=code,
            code_yhat=code_yhat,
            num_bootstrap=NUM_BOOTSTRAP,
            all_scores=all_scores,
        ),
        axis=1,
    )

    # Add parameters used to generate results
    top_m_df["sample"] = sample
    top_m_df["P"] = P
    top_m_df["K"] = K
    top_m_df["N"] = N
    top_m_df["runtime_hours"] = time_hours

    # Add to all blocks dataframe
    dfs.append(top_m_df)

    # Add random results to all blocks dataframe
    if ADD_RANDOM:

        # Build the top M dataframe using random blocks
        top_m_random_df = final_df.sample(n=M)
        top_m_random_df["p_val"] = top_m_random_df.apply(
            lambda record: compute_p_val_from_bootstrap(
                draws=record.draws,
                avg_score_diff=record.avg_score_diff,
                code=code,
                code_yhat=code_yhat,
                num_bootstrap=NUM_BOOTSTRAP,
                all_scores=all_scores,
            ),
            axis=1,
        )

        # Add parameters used to generate results
        top_m_random_df["sample"] = sample
        top_m_random_df["P"] = "RANDOM"
        top_m_random_df["K"] = K
        top_m_random_df["N"] = N
        top_m_random_df["runtime_hours"] = time_hours

        # Add to all blocks dataframe
        dfs.append(top_m_random_df)

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

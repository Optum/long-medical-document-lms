"""
Module: Masked Sampling Procedure

"""
import os
import torch
import numpy as np
import pandas as pd
from utils import (
    predict_with_clf_model,
    configure_model_for_inference,
    convert_binary_to_multi_label,
)


def predict_with_masked_texts(
    model,
    input_ids,
    n,
    k,
    p,
    mask_token_id,
    idx2label,
    print_every,
    debug,
    device,
    max_seq_len,
    class_strategy,
):
    """
    Returns the probabilities for each label for each iteration with the masked strings and labels.
    Used to determine the importance of text blocks in a document by randomly masking blocks.
    """

    # Start
    print(
        f"Running {n} iterations to estimate label probabilities with masked blocks of text."
    )
    print(f"Each block of {k} subword tokens is masked with probability {p}.")

    # Configure model for inference
    model = configure_model_for_inference(model, device)

    # Track the text strings masked in each trial and their indices
    all_masked_text_tokens = []
    all_masked_text_indices = []

    # Track the probabilities for each label from each round of masking for each trail
    all_probs = []

    # Run trials
    for i in range(n):

        # Only run a few iterations if in debug mode
        if debug:
            if i == 20:
                break

        # Notify trial
        if i % print_every == 0:
            print(f"   On iteration {i} of {n}...")

        # At each trial, save the strings of masked text and the start index of each string
        masked_text_tokens = []
        masked_text_indices = []

        # For each sample, create a new sample consisting of masked and unmasked blocks
        new_sample = []
        for j in range(0, len(input_ids), k):
            block = input_ids[j : j + k]

            # Mask a block with probability P and add the block to the new sample
            if np.random.random() < p:
                mask_block = [mask_token_id] * k
                new_sample.extend(mask_block)
                masked_text_indices.append(j)
                masked_text_tokens.append(block)
            else:
                new_sample.extend(block)

        # Compute probabilities of each label on the new sample
        prob = predict_with_clf_model(
            model,
            sample_input_ids=[new_sample[0:max_seq_len]],
            device=device,
            class_strategy=class_strategy,
        )[0]

        # Save the probabilities, text strings, and indices from this trial
        all_probs.append(prob)
        all_masked_text_tokens.append(masked_text_tokens)
        all_masked_text_indices.append(masked_text_indices)

    # Build dataframe of results
    results = pd.DataFrame(
        all_probs, columns=[idx2label[idx] for idx in range(len(all_probs[0]))]
    )

    results["masked_text_tokens"] = all_masked_text_tokens
    results["masked_text_indices"] = all_masked_text_indices

    return results


def post_process_and_save_msp_results(
    model,
    all_results,
    all_input_ids,
    all_labels,
    times,
    device,
    tokenizer,
    num_sample,
    max_seq_len,
    class_strategy,
    idx2label,
    num_bootstrap,
    output_path,
    n,
    k,
    p,
    m,
):
    """
    This step iterates through the results of running MSP for each document and:
    1. Generates baseline inferences on the original text sequences to compare against the masked sequences
    2. Computes the differences in predictions for positive labels between unmasked and masked sequences as the importance score for each text block
    3. Sorts text blocks by importance score
    4. Decodes the tokens in each important text block into text strings
    5. Computes p-values to verify the importance score rankings
    6. Adds other metadata for each block including a random ID and the document number from the loop (could modify to use a unique ID if the dataset has one)
    7. Saved blinded and unblinded versions of the important blocks and metadata for a document
    """

    # Configure model for inference
    model = configure_model_for_inference(model, device)

    # Iterate through results on all documents to post-process and save explanations
    for s, (results, doc_input_ids, doc_y, doc_time) in enumerate(
        zip(all_results, all_input_ids, all_labels, times)
    ):

        # Indicate sample number
        print(f"Post-processing explanations for sample {s} of {num_sample}...")

        # Explode results to create a single block per dataframe row
        # with some blocks represented many times according to the number of times the block was selected
        exploded_results = results.explode(
            ["masked_text_indices", "masked_text_tokens"]
        )

        # Get predictions on sample with no masking
        full_yhat = predict_with_clf_model(
            model,
            sample_input_ids=[doc_input_ids[0:max_seq_len]],
            device=device,
            class_strategy=class_strategy,
        )[0]

        # Convert a single binary label to two labels or leave alone if already multi-target array
        doc_y = convert_binary_to_multi_label(doc_y)

        # Combine predictions, labels, descriptions and view positive labels
        labels, ys, yhats = [], [], []
        for idx in range(len(doc_y)):
            labels.append(idx2label[idx])
            ys.append(doc_y[idx])
            yhats.append(full_yhat[idx])

        # Create dataframe of true labels and predicted labels
        # Sort to focus on positive labels
        full_preds = pd.DataFrame({"y": ys, "yhat": yhats, "label": labels})
        pos_full_preds = full_preds[full_preds["y"] == 1].sort_values(
            by="yhat", ascending=False
        )

        # Save all the scores from the masking procedure for all the positive labels
        all_pos_labels = pos_full_preds["label"].tolist()
        all_pos_label_scores = {
            label: results[label].tolist() for label in all_pos_labels
        }

        # For the positive labels, compute which blocks of text account
        # for the greatest difference in the predicted probability of each label when randomly masked
        dfs = []
        for i, label in enumerate(all_pos_labels):

            # Compute average differences from unmasked prediction for masked blocks for code
            print(
                f"Computing average differences between probability of {label} for unmasked and masked text..."
            )

            # Compute the difference in probabilities between the unmasked prediction and the masked predictions
            label_yhat = pos_full_preds[pos_full_preds["label"] == label]["yhat"].item()
            label_df = exploded_results[
                [label, "masked_text_indices", "masked_text_tokens"]
            ]
            label_df["score_diff"] = label_yhat - label_df[label]

            # Find the token blocks with the highest average score difference for the selected label
            sorted_df = (
                label_df.groupby(["masked_text_indices"])
                .agg(
                    draws=("masked_text_indices", "size"),
                    avg_score_diff=("score_diff", "mean"),
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
            final_df = final_df[
                ["draws", "avg_score_diff", "text_block"]
            ].drop_duplicates()

            # Add label and description
            final_df["label"] = label

            # Remove padding
            all_pad_block = tokenizer.pad_token * k
            final_df = final_df[final_df["text_block"] != all_pad_block]

            # Build the top M dataframe based on the masking procedure and add p values
            top_m_df = final_df.sort_values(by="avg_score_diff", ascending=False).head(
                m
            )
            top_m_df["p_val"] = top_m_df.apply(
                lambda record: compute_p_val_from_bootstrap(
                    draws=record.draws,
                    avg_score_diff=record.avg_score_diff,
                    label=label,
                    label_yhat=label_yhat,
                    num_bootstrap=num_bootstrap,
                    all_scores=all_pos_label_scores,
                ),
                axis=1,
            )

            # Add parameters used to generate results
            top_m_df["sample"] = s
            top_m_df["P"] = p
            top_m_df["K"] = k
            top_m_df["N"] = n
            top_m_df["runtime_secs"] = doc_time

            # Add to all blocks dataframe
            dfs.append(top_m_df)

        # Combine output dataframes
        all_blocks_df = pd.concat(dfs)

        # Add random row IDs
        row_ids = np.arange(len(all_blocks_df))
        np.random.shuffle(row_ids)
        all_blocks_df["row_id"] = row_ids

        # Save output dataframe with all experiment information
        all_blocks_df.to_csv(
            os.path.join(output_path, f"doc_{s}_all_info.csv"), index=False
        )

        # Select blind blocks columns and add column for review
        blind_all_blocks_df = all_blocks_df[["row_id", "label", "text_block"]]
        blind_all_blocks_df["informative_0_or_1"] = "-"

        # Save output dataframe with hidden information
        blind_all_blocks_df.to_csv(
            os.path.join(output_path, f"doc_{s}_exp_data.csv"), index=False
        )


def compute_p_val_from_bootstrap(
    draws, avg_score_diff, label, label_yhat, num_bootstrap, all_scores
):
    """
    Compute p values for text blocks using all scores recorded for each code via bootstrap sampling
    """

    results = []
    score_diffs = label_yhat - np.array(all_scores[label])
    for i in range(num_bootstrap):
        random_sample = np.random.choice(score_diffs, size=draws)
        bootstrap_score = np.mean(random_sample)
        results.append(bootstrap_score)

    p_val = np.count_nonzero(np.array(results) > avg_score_diff) / num_bootstrap

    return p_val

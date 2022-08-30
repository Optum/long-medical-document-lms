"""
Module: Generate random blocks
"""

import os
import numpy as np
import pandas as pd
from utils import (
    convert_binary_to_multi_label,
    predict_with_clf_model,
    configure_model_for_inference,
)

# Add code to generate random blocks and save results in similar format to MSP and SOC


def predict_with_rnd_algo(
    input_ids,
    doc_y,
    n,
    k,
    m,
    idx2label,
    max_seq_len,
    class_strategy,
):
    """
    Returns the probabilities for each label for each iteration with the masked strings and labels.
    Used to determine the importance of text blocks in a document by randomly masking blocks.
    """

    # Start
    print(f"Generating random blocks...")

    # Convert a single binary label to two labels or leave alone if already multi-target array
    doc_y = convert_binary_to_multi_label(doc_y)

    # Iterate through labels and generate random blocks for each
    results = None
    for i, _ in enumerate(doc_y):

        # Compute all block indices and select m random index pairs
        block_indices = [(j, j + k) for j in range(0, len(input_ids), k)]
        selected_indices = [select_random_index_pairs(block_indices) for _ in range(m)]

        # Slice to retrieve the token block associated with the selected index pairs
        all_tokens = [input_ids[idx[0] : idx[1]] for idx in selected_indices]

        # Save the start index for each selected token block
        all_indices = [idx[0] for idx in selected_indices]

        label = idx2label[i]

        if i == 0:
            results = pd.DataFrame(select_m_random_neg_to_pos_one(m), columns=[label])
        else:
            results[label] = select_m_random_neg_to_pos_one(m)
            results["masked_text_tokens"] = all_tokens
            results["masked_text_indices"] = all_indices

    return results


def select_random_index_pairs(pair_list):
    """
    Return a randomly selected pair from a list of pairs of block indices.
    """

    pair_arr = np.array(pair_list)

    return pair_arr[np.random.randint(pair_arr.shape[0], size=1), :][0]


def select_m_random_neg_to_pos_one(m):
    """
    Return a 1D array of m elements from a uniform random between -1 and 1.
    """

    return np.random.rand(
        m,
    ) * np.random.choice([-1, 1], size=(m,))


def post_process_and_save_rnd_results(
    model,
    all_results,
    all_input_ids,
    all_labels,
    times,
    tokenizer,
    device,
    num_sample,
    max_seq_len,
    class_strategy,
    idx2label,
    output_path,
    n,
    k,
    m,
):
    """ """

    # Iterate through results on all documents to post-process and save explanations
    for s, (results, doc_input_ids, doc_y, doc_time) in enumerate(
        zip(all_results, all_input_ids, all_labels, times)
    ):

        # Indicate sample number
        print(f"Post-processing explanations for sample {s} of {num_sample}...")

        # Predict on the phrase text with LM with CLF head
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
        # Filter to focus on positive labels and sort by scores
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
            label_df = results[[label, "masked_text_indices", "masked_text_tokens"]]
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
                results[["masked_text_indices", "masked_text_tokens"]],
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

            # Add parameters used to generate results
            top_m_df["sample"] = s
            top_m_df["P"] = "RND"
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

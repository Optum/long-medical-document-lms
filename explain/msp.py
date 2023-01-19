"""
Module: Masked Sampling Procedure

"""
import os
import pysbd
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from utils import (
    configure_model_for_inference,
    convert_binary_to_multi_label,
    torch_model_predict,
    torch_model_predict_indiv,
)


def run_trial_on_fixed_blocks(tokenizer, input_ids, p, k):

    # At each trial, save the masked tokens and the start index of each block
    masked_text_tokens = []
    masked_text_indices = []

    # For each sample, create a new sample consisting of masked and unmasked blocks
    new_sample = []

    # Iterate through fixed length blocks
    for j in range(0, len(input_ids), k):
        block = input_ids[j : j + k]

        # Mask a block with probability P and add the block to the new sample
        if random.uniform(0, 1) < p:

            # Create a masked block of appropriate length
            # Save the index of the block start
            # Add the block to the new sample
            # Save the block
            mask_block = [tokenizer.mask_token_id] * k
            new_sample.extend(mask_block)
            masked_text_indices.append(j)
            masked_text_tokens.append(block)

        else:
            new_sample.extend(block)

    return new_sample, masked_text_tokens, masked_text_indices


def run_trial_on_sentences(segmenter, tokenizer, text, p):

    # Check that tokenizer always gives us a CLS and SEP token at the start and end
    test_sent = "This is a test sentence."
    e_test_sent = tokenizer.encode(test_sent)
    assert_msg = "Tokenizer should always add CLS and SEP tokens to the start and end of each input sequence respectively."
    assert (
        e_test_sent[0] == tokenizer.cls_token_id
        and e_test_sent[-1] == tokenizer.sep_token_id
    ), assert_msg

    # At each trial, save the masked tokens and the start index of each block
    masked_text_tokens = []
    masked_text_indices = []

    # For each sample, create a new sample consisting of masked and unmasked blocks
    # We're removing CLS tokens, so make sure the new sample starts with one
    new_sample = [tokenizer.cls_token_id]

    # Iterate through logical text segments
    for segment in segmenter.segment(text):

        # Encode a block and remove the CLS and SEP tokens
        # Slicing should be faster than removing [tokenizer.cls_token_id, tokenizer.sep_token_id] explicitly
        # But it assumes these are always present when encoding
        block = tokenizer.encode(segment)
        cleaned_block = block[1:-1]

        # Mask a block with probability P and add the block to the new sample
        if random.uniform(0, 1) < p:

            # Create a masked block of appropriate length
            # Save the index of the block start
            # Add the block to the new sample
            # Save the block
            mask_block = [tokenizer.mask_token_id] * len(cleaned_block)
            masked_text_indices.append(len(new_sample))
            new_sample.extend(mask_block)
            masked_text_tokens.append(cleaned_block)

        else:
            new_sample.extend(cleaned_block)

    return new_sample, masked_text_tokens, masked_text_indices


def predict_with_masked_texts(
    model,
    input_ids,
    text,
    n,
    k,
    p,
    idx2label,
    print_every,
    debug,
    max_seq_len,
    class_strategy,
    tokenizer,
    by_sent_segments,
    batch_size,
):
    """
    Returns the probabilities for each label for each iteration with the masked strings and labels.
    Used to determine the importance of text blocks in a document by randomly masking blocks.
    """

    # Start
    print(
        f"Running {n} iterations to estimate label probabilities with masked blocks of text."
    )

    # Track the text strings masked in each trial and their indices
    all_masked_text_tokens = []
    all_masked_text_indices = []

    # Collect the new samples created from each round of masking for each trial
    collected_new_samples = []

    # Initialize segmenter if generating per sentence explanations
    if by_sent_segments:

        # Initialize segmenter
        segmenter = pysbd.Segmenter(language="en", clean=False)

    # Run trials
    for i in range(n):

        # Only run a few iterations if in debug mode
        if debug:
            if i == 20:
                break

        # Notify trial
        if i % print_every == 0:
            print(f"   On iteration {i} of {n}...")

        # Generate sentence-level or fixed block-level explanations
        if by_sent_segments:
            (
                new_sample,
                masked_text_tokens,
                masked_text_indices,
            ) = run_trial_on_sentences(
                segmenter=segmenter, tokenizer=tokenizer, text=text, p=p
            )
        else:
            (
                new_sample,
                masked_text_tokens,
                masked_text_indices,
            ) = run_trial_on_fixed_blocks(
                tokenizer=tokenizer, input_ids=input_ids, p=p, k=k
            )

        # Save the masked blocks and start indices from this trial
        all_masked_text_tokens.append(masked_text_tokens)
        all_masked_text_indices.append(masked_text_indices)

        # Collect the new sample from this trial
        collected_new_samples.append(new_sample[0:max_seq_len])

    # Pad new sequences
    # Generate pointers to check that predictions are returned in the same order
    padded_sequences = pad_sequence(
        torch.tensor(collected_new_samples, dtype=torch.int64),
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    pointers_orig = torch.tensor([[i] for i in range(len(padded_sequences))])

    # Build dataset of new sequences to use to generate label probabilities
    # Include the pointers we created
    ds = TensorDataset(padded_sequences, pointers_orig)
    dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
    )

    # Iterate through data loader to make predictions and return the pointers
    all_probs, pointers_returned = torch_model_predict(
        model=model,
        test_loader=dataloader,
        class_strategy=class_strategy,
        return_data_loader_targets=True,
    )

    # Check that predictions came back in the same order
    # We want all_probs, all_masked_text_tokens, and all_masked_text_indices in corresponding order
    assert np.array_equal(
        pointers_orig.numpy(), pointers_returned
    ), "Order of records was shuffled during inference!"

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
    by_sent_segments,
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
    model = configure_model_for_inference(model)

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
        full_yhat = torch_model_predict_indiv(
            model,
            sample_input_ids=[doc_input_ids[0:max_seq_len]],
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
            top_m_df["K"] = k if not by_sent_segments else "By Sentence Segments"
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

"""
Module: Sampling and Occlusion Algorithm

"""
import os
import copy
import torch
import numpy as np
import pandas as pd
from utils import (
    predict_with_clf_model,
    configure_model_for_inference,
    convert_binary_to_multi_label,
)


def predict_with_soc_algo(
    model,
    lm,
    input_ids,
    doc_y,
    samples,
    n_gram_range,
    radius,
    mask_token_id,
    pad_token_id,
    idx2label,
    vocab_size,
    print_every,
    debug,
    device,
    max_seq_len,
    class_strategy,
):
    """
    Returns the SOC algo score for each phrase in the sample text according to the number of samples to take, n_gram_range, and radius.
    See https://arxiv.org/pdf/1911.06194.pdf for details.
    """

    # Start
    print(
        f"Running {samples} iterations for each phrase of {n_gram_range} blocks to estimate probabilities with masked blocks of text."
    )
    print(
        f"Each block of {n_gram_range} subword tokens is padded while the {n_gram_range} blocks to the left and the right are sampled from the MLM in each of the {samples} iterations.  The score of each padded block is taken as the average probability of each label over all iterations."
    )

    # Configure model for inference
    lm = configure_model_for_inference(lm, device)
    model = configure_model_for_inference(model, device)

    # Count inferences
    num_inferences = 0

    # Define tokens in vocab to use later
    vocab = np.array(range(vocab_size))

    # We will compute scores for every phrase of size n_gram_range
    # We also save the start index of the phrase along with the associated tokens
    all_scores = []
    text_tokens = []
    text_indices = []

    # First we iterate over the input text and pick out each target block as our phrase
    for i in range(0, len(input_ids), n_gram_range):

        # Only run a few iterations if in debug mode
        if debug:
            if i == 20:
                break

        # If all we have left of the text is padding, we're done with this sample
        if input_ids[i : i + n_gram_range] == [pad_token_id] * n_gram_range:
            break

        # Save the start index and tokens
        text_indices.append(i)
        text_tokens.append(input_ids[i : i + n_gram_range])

        # Notify progress on this sample
        if i % print_every == 0:
            print(f"   On phrase {i}...")

        # Copy the input text as new_text which we will modify with masking and padding
        lm_text = copy.deepcopy(input_ids)

        # Replace the target phrase with padding
        lm_text[i : i + n_gram_range] = [pad_token_id] * n_gram_range

        # Get the size of the left context (usually of size radius)
        if i - radius < 0:
            left_idx = 0
        else:
            left_idx = i - radius
        left_context_len = len(lm_text[left_idx:i])

        # Replace the left context with the correct number of mask tokens
        if left_context_len > 0:
            lm_text[left_idx:i] = [mask_token_id] * left_context_len

        # Get the size of the right context (usually of size radius)
        if i + radius > max_seq_len:
            right_idx = max_seq_len
        else:
            right_idx = i + radius
        right_context_len = len(lm_text[i:right_idx])

        # Replace the right context with the correct number of mask tokens
        if right_context_len > 0:
            lm_text[i:right_idx] = [mask_token_id] * right_context_len

        # Predict with LM to get probabilities of masked tokens
        with torch.no_grad():
            lm_input_ids = torch.tensor([lm_text[0:max_seq_len]]).to(device)
            lm_output = lm(lm_input_ids).logits
            lm_prob = (
                torch.nn.functional.softmax(lm_output, dim=2).data.cpu().numpy()[0]
            )

        # Add to the total number of inferences made
        num_inferences += 1

        # For the target phrase, initialize a new array of scores and predict labels
        # for the sequence while the target phrase is padded.  Sample values for the masked
        # tokens "samples" times and predict on the sequence with these replacements
        # such that we get as many scores as there are sampling rounds.
        scores = []
        for j in range(samples):

            # Copy the input text as clf_text to predict on to generate scores after filling masks with the LM
            clf_text = copy.deepcopy(input_ids)

            # Extract predictions of left context masked tokens and replace the masked tokens with the predicted tokens
            if left_context_len > 0:
                left_context_probs = lm_prob[left_idx:i]
                left_context_tokens = [
                    np.random.choice(vocab, size=1, p=p)[0] for p in left_context_probs
                ]
                clf_text[left_idx:i] = left_context_tokens

            # Extract predictions of right context masked tokens and replace the masked tokens with the predicted tokens
            if right_context_len > 0:
                right_context_probs = lm_prob[i:right_idx]
                right_context_tokens = [
                    np.random.choice(vocab, size=1, p=p)[0] for p in right_context_probs
                ]
                clf_text[i:right_idx] = right_context_tokens

            # Predict on the phrase text with LM with CLF head
            clf_prob = predict_with_clf_model(
                model,
                sample_input_ids=[clf_text[0:max_seq_len]],
                device=device,
                class_strategy=class_strategy,
            )[0]

            # Add the predicted probabilities of each label from this round of sampling
            scores.append(clf_prob)

            # Add to the total number of inferences made
            num_inferences += 1

        # Add the scores generated for the target phrase to the array of total scores
        all_scores.append(scores)

    # Convert a single binary label to two labels or leave alone if already multi-target array
    doc_y = convert_binary_to_multi_label(doc_y)

    # Build dataframe of results consisting of the average scores for each label from each round of sampling for a given phrase
    results = pd.DataFrame(
        [np.mean(np.vstack(scores), axis=0) for scores in all_scores],
        columns=[idx2label[idx] for idx in range(len(doc_y))],
    )

    # Select columns for which true label is present
    results = results.iloc[:, np.where(doc_y == 1)[0]]

    # Add tokens and text indices
    results["text_tokens"] = text_tokens
    results["text_indices"] = text_indices

    return results, num_inferences


def post_process_and_save_soc_results(
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
    output_path,
    n,
    k,
    m,
):

    # Configure model for inference
    model = configure_model_for_inference(model, device)

    # Iterate through results on all documents to post-process and save explanations
    for s, (results, doc_input_ids, doc_y, doc_time) in enumerate(
        zip(all_results, all_input_ids, all_labels, times)
    ):

        # Indicate sample number
        print(f"Post-processing explanations for sample {s} of {num_sample}...")

        # Predict on the phrase text with LM with CLF head
        # Get predictions on sample with no masking
        prob = predict_with_clf_model(
            model,
            sample_input_ids=[doc_input_ids[0:max_seq_len]],
            device=device,
            class_strategy=class_strategy,
        )[0]

        # Convert a single binary label to two labels or leave alone if already multi-target array
        doc_y = convert_binary_to_multi_label(doc_y)

        # Build dataframe of results consisting of the average scores for each label from each round of sampling for a given phrase
        base_results = pd.DataFrame(
            [prob], columns=[idx2label[idx] for idx in range(len(doc_y))]
        )

        # Select columns for which true label is present
        base_results = base_results.iloc[:, np.where(doc_y == 1)[0]]

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
            final_df["text_block"] = results["text_tokens"].apply(
                lambda x: tokenizer.decode(x)
            )
            final_df = final_df.drop(["text_tokens", "text_indices"], 1)

            # Add label and description
            final_df["label"] = col.replace("diff_", "")

            # Sort dataframe and add info about SOC algo run
            top_m_df = final_df.sort_values(by="avg_score_diff", ascending=False).head(
                m
            )
            top_m_df["sample"] = s
            top_m_df["draws"] = n
            top_m_df["K"] = k
            top_m_df["P"] = "SOC"
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

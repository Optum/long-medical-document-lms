"""
Module: Masked Sampling Procedure

"""
import torch
import random
import numpy as np
import pandas as pd
from utils import get_model_for_eval, configure_device


def predict_with_masked_texts(
    sample_text, n, k, p, mask_token_id, idx2label, print_every, debug, params
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

    # Get the model to use for inference
    model = get_model_for_eval(model_type="clf", params=params)

    # Get device
    device = configure_device()

    # Put model on device
    model.to(device)

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
        for j in range(0, len(sample_text), k):
            block = sample_text[j : j + k]

            # Mask a block with probability P and add the block to the new sample
            if random.random() < p:
                mask_block = [mask_token_id] * k
                new_sample.extend(mask_block)
                masked_text_indices.append(j)
                masked_text_tokens.append(block)
            else:
                new_sample.extend(block)

        # Compute probabilities of each label on the new sample
        with torch.no_grad():
            input_ids = torch.tensor([new_sample[0 : params["max_seq_len"]]]).to(device)
            output = model(input_ids).logits
            prob = torch.sigmoid(output).data.cpu().numpy()[0]

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


def compute_p_val_from_bootstrap(
    draws, avg_score_diff, code, code_yhat, num_bootstrap, all_scores
):
    """
    Compute p values for text blocks using all scores recorded for each code via bootstrap sampling
    """

    results = []
    score_diffs = code_yhat - np.array(all_scores[code])
    for i in range(num_bootstrap):
        random_sample = np.random.choice(score_diffs, size=draws)
        bootstrap_score = np.mean(random_sample)
        results.append(bootstrap_score)

    p_val = np.count_nonzero(np.array(results) > avg_score_diff) / num_bootstrap

    return p_val

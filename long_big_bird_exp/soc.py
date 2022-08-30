"""
Module: Sampling and Occlusion Algorithm

"""
import copy
import torch
import random
import numpy as np
import pandas as pd
from utils import get_model_for_eval, configure_device


def predict_with_soc_algo(
    sample_text,
    sample_y,
    samples,
    n_gram_range,
    radius,
    mask_token_id,
    pad_token_id,
    idx2label,
    vocab_size,
    print_every,
    debug,
    params,
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

    # Count inferences
    num_inferences = 0

    # Define tokens in vocab to use later
    vocab = np.array(range(vocab_size))

    # Get device
    device = configure_device()

    # Get the LM to fill masks
    lm_model = get_model_for_eval(model_type="mlm", params=params)

    # Put LM on device
    lm_model.to(device)

    # Get the LM with CLF head to predict labels
    model = get_model_for_eval(model_type="clf", params=params)

    # Put LM with CLF head on device
    model.to(device)

    # We will compute scores for every phrase of size n_gram_range
    # We also save the start index of the phrase along with the associated tokens
    all_scores = []
    text_tokens = []
    text_indices = []

    # First we iterate over the input text and pick out each target block as our phrase
    for i in range(0, len(sample_text), n_gram_range):

        # Only run a few iterations if in debug mode
        if debug:
            if i == 20:
                break

        # If all we have left of the text is padding, we're done with this sample
        if sample_text[i : i + n_gram_range] == [pad_token_id] * n_gram_range:
            break

        # Save the start index and tokens
        text_indices.append(i)
        text_tokens.append(sample_text[i : i + n_gram_range])

        # Notify progress on this sample
        if i % print_every == 0:
            print(f"   On phrase {i}...")

        # Copy the input text as new_text which we will modify with masking and padding
        lm_text = copy.deepcopy(sample_text)

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
        if i + radius > params["max_seq_len"]:
            right_idx = params["max_seq_len"]
        else:
            right_idx = i + radius
        right_context_len = len(lm_text[i:right_idx])

        # Replace the right context with the correct number of mask tokens
        if right_context_len > 0:
            lm_text[i:right_idx] = [mask_token_id] * right_context_len

        # Predict with LM to get probabilities of masked tokens
        with torch.no_grad():
            lm_input_ids = torch.tensor([lm_text[0 : params["max_seq_len"]]]).to(device)
            lm_output = lm_model(lm_input_ids).logits
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
            clf_text = copy.deepcopy(sample_text)

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
            with torch.no_grad():
                clf_input_ids = torch.tensor([clf_text[0 : params["max_seq_len"]]]).to(
                    device
                )
                clf_output = model(clf_input_ids).logits
                clf_prob = torch.sigmoid(clf_output).data.cpu().numpy()[0]

            # Add the predicted probabilities of each label from this round of sampling
            scores.append(clf_prob)

            # Add to the total number of inferences made
            num_inferences += 1

        # Add the scores generated for the target phrase to the array of total scores
        all_scores.append(scores)

    # Build dataframe of results consisting of the average scores for each label from each round of sampling for a given phrase
    results = pd.DataFrame(
        [np.mean(np.vstack(scores), axis=0) for scores in all_scores],
        columns=[idx2label[idx] for idx in range(len(sample_y))],
    )

    # Select columns for which true label is present
    results = results.iloc[:, np.where(sample_y == 1)[0]]

    # Add tokens and text indices
    results["text_tokens"] = text_tokens
    results["text_indices"] = text_indices

    return results, num_inferences

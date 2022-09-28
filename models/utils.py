#!/usr/bin/env python
# coding: utf-8

"""
Utility functions for deep learning experiments
"""

import os
import yaml
import torch
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


def create_current_run(save_path, params, logger=None):
    """
    Create a directory for the current run, save the
    current pipeline parameters, and return the
    path to the current run directory.
    """

    # Create current run dir
    src_dirs = os.listdir(save_path)
    max_run = (
        max([int(dir.split("_")[1]) for dir in src_dirs]) if len(src_dirs) > 0 else -1
    )
    current_run_dir = os.path.join(save_path, "run_" + str(max_run + 1) + "/")
    os.makedirs(current_run_dir)

    if logger:
        logger.info(f"Created current run dir: {current_run_dir}.")

    # Save run params in current run dir for reference
    with open(os.path.join(current_run_dir, "params.yml"), "w") as stream:
        yaml.dump(params, stream, default_flow_style=False)

    if logger:
        logger.info(f"Saved run parameter to current run dir.")

    return current_run_dir


def create_log_dir(current_run_dir, logger=None):

    logging_dir = os.path.join(current_run_dir, "logs/")

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

        if logger:
            logger.info(f"Created logging directory: {logging_dir}.")


def check_empty_count_gpus(logger=None):
    """
    Check that GPU is available, empty the cache,
    and count the number of available devices.
    """

    # Check that a GPU is available:
    assert torch.cuda.is_available(), "No GPU found.  Please run on a GPU."

    # Empty GPU cache
    torch.cuda.empty_cache()

    # Count available devices
    device_count = torch.cuda.device_count()

    if logger:
        logger.info(f"Found {device_count} GPU(s)!")


def convert_1d_binary_labels_to_2d(labels):
    """
    Convert 1D binary labels to a 2D representation.
    """

    # Convert a 1D, binary label array to 2D
    if isinstance(labels[0], np.integer) or isinstance(labels[0], int):

        # Check that we have a 1D array of 1s and 0s
        assert len(np.array(labels).shape), "Expected labels to be 1D."
        assert all(
            x == 0 or x == 1 for x in labels
        ), "Expected only 1s and 0s in labels."

        # Convert to 2D representation
        new_labels = np.zeros(shape=(len(labels), 2))
        for i, target in enumerate(labels):
            if target == 0:
                new_labels[i] = [1, 0]
            elif target == 1:
                new_labels[i] = [0, 1]
            else:
                raise ValueError(f"Unexpected target: {target}.")

        return new_labels

    # Return 2D array
    else:

        if isinstance(labels, (np.ndarray, np.generic)):
            return labels
        else:
            return np.array(labels)


def make_lr_model_and_target_multi_class(model, y, class_strategy, n_jobs=-1):
    """
    Given an sklearn LogisticRegression model and
    a parameter indicating the multi-class training strategy
    convert the model to a OneVsRestClassifier or
    multinomial regression and return it with the
    n_jobs parameter set to parallelize training.
    Also returns the target array such that the final
    return type is a tuple of (model, y) and y is
    modified to use multi_class indices if
    class_strategy='multi_class'.
    """

    if class_strategy == "multi_label":

        # Wrap model in OVR classifier
        model = OneVsRestClassifier(model, n_jobs=n_jobs)

    elif class_strategy == "multi_class":

        # Set model attributes
        model.multi_class = "multinomial"
        model.n_jobs = n_jobs

        # Transform target array
        y = transform_target_to_multi_class_indices(y)

    else:

        # Raise exception
        raise ValueError(
            f"Expected class_strategy to be one of ['multi_label', 'multi_class'] but got {class_strategy}."
        )

    return model, y


def transform_target_to_multi_class_indices(y):
    """
    Given a 2d numpy array of one hot encoded
    targets, return an array of the indices
    representing the encoded label for each sample
    as is required for sklearn multi-class classification.
    """

    return np.argmax(y, axis=1)

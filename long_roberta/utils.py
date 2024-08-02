#!/usr/bin/env python
# coding: utf-8

"""
Utility functions for deep learning experiments
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def create_current_run(save_path, params, logger=None):
    """
    Create a directory for the current run, save the 
    current pipeline parameters, and return the 
    path to the current run directory.
    """
    
    # Create current run dir
    src_dirs = os.listdir(save_path)
    max_run = max([int(dir.split('_')[1]) for dir in src_dirs]) if len(src_dirs) > 0 else -1
    current_run_dir = os.path.join(save_path, 'run_' + str(max_run + 1) + '/')
    os.makedirs(current_run_dir)
    
    if logger:
        logger.info(f'Created current run dir: {current_run_dir}.')

    # Save run params in current run dir for reference
    with open(os.path.join(current_run_dir, 'params.yml'), 'w') as stream:
        yaml.dump(params, stream, default_flow_style=False)
    
    if logger:
        logger.info(f'Saved run parameter to current run dir.')
        
    return current_run_dir

def check_empty_count_gpus(logger=None):
    """
    Check that GPU is available, empty the cache,
    and count the number of available devices.
    """
    
    # Check that a GPU is available:
    assert torch.cuda.is_available(), 'No GPU found.  Please run on a GPU.'

    # Empty GPU cache
    torch.cuda.empty_cache()

    # Count available devices
    device_count = torch.cuda.device_count()

    if logger:
        logger.info(f'Found {device_count} GPU(s)!')

def np_sigmoid(z):
    """
    Convert logits to probabilities:
    https://en.wikipedia.org/wiki/Sigmoid_function.
    """
    
    return 1 / (1 + np.exp(-z))

def to_binary_one_hot(y):
    """
    Convery 0 and 1 labels to 
    [1, 0] and [0, 1] for generality
    """
    
    yn = np.zeros((len(y), 2), dtype=int)
    for i, val in enumerate(y):
        yn[i, 0] = 1 - val # 0 -> 1 & 1 -> 0
        yn[i, 1] = val # 0 -> 0 & 1 -> 1 
    
    return yn

def load_and_split_imdb_data(path, seed=42, num_labels=1):
    """
    Load and split imdb data for testing code.
    """
    
    # Read data and create features and labels
    df = pd.read_csv(path)
    texts = df['sentence'].tolist()
    labels = df['target'].tolist()
    
    # Create train, val, test splits
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=.15, random_state=seed, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15, random_state=seed, shuffle=False)
    
    # Convert binary labels to one hot labels for generality or use a single binary label wrapped in an extra dim
    if num_labels == 1:
        y_train = np.array([[x] for x in y_train])
        y_val = np.array([[x] for x in y_val])
        y_test = np.array([[x] for x in y_test])
    elif num_labels == 2:
        y_train = to_binary_one_hot(y_train)
        y_val = to_binary_one_hot(y_val)
        y_test = to_binary_one_hot(y_test)
    else:
        raise ValueError("For this dataset, the labels should be encoded using either 1 or 2 columns.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_learning_curve(result, current_run_dir, prefix):
    
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize = (10,10))

    for i, (key, value) in enumerate(result.items()):
        ax.plot(value, '-',label=key,color=cmap(i))
        ax.legend()
    
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Learning Curves')
    plt.tight_layout()
    plt.savefig(current_run_dir + f'{prefix}_learning_curves.png', transparent=False)

#!/usr/bin/env python
# coding: utf-8

# Open imports
import os
import json
import yaml
import logging
import pickle
import numpy as np
import pandas as pd
from datasets import load_from_disk

# Project imports
from main import BERTClassificationModelWithPooling, BERTClassificationModel
from utils import check_empty_count_gpus, create_current_run, np_sigmoid, load_and_split_imdb_data, plot_learning_curve
from metrics import BootstrapMultiLabelMetrics

# Load run parameters
with open("params.yml", "r") as stream:
    PARAMS = yaml.safe_load(stream)

# Define logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log parameters
logger.info(PARAMS)

# Check, empty, and count GPUs
check_empty_count_gpus(logger=logger)

# Set gpus
os.environ["CUDA_VISIBLE_DEVICES"]= PARAMS['visible_gpus']

# Create run directory
current_run_dir = create_current_run(save_path=PARAMS['output_path'], params=PARAMS, logger=logger)
logger.info(f"Created run directory: {current_run_dir}.")

# Set run name
run_name = current_run_dir.split('/')[-1]
logger.info(f"Starting run {run_name}...")

# Load data
if PARAMS['test_with_imdb_data']:
    
    # Use IMDB data to test model
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_imdb_data(PARAMS['imdb_data'], num_labels=PARAMS['num_labels'])

else:

    # Load real data
    d = load_from_disk(PARAMS['dataset_path'])
    X_train = np.array(d['train']['text'])
    y_train = np.array(d['train']['label']).reshape(-1, 1)  
    X_val = np.array(d['val']['text'])
    y_val = np.array(d['val']['label']).reshape(-1, 1)  
    X_test = np.array(d['test']['text'])
    y_test = np.array(d['test']['label']).reshape(-1, 1)  

# Print data shapes
logger.info(f'Train shapes: {len(X_train), y_train.shape}')
logger.info(f'Val shapes: {len(X_val), y_val.shape}')
logger.info(f'Test shapes: {len(X_test), y_test.shape}')

# Load model
if PARAMS['use_pooled_bert']:
    model = BERTClassificationModelWithPooling()
else:
    model = BERTClassificationModel()

# Train and evaluate
result = model.train_and_evaluate(
    X_train,
    X_val,
    X_test,
    y_train, 
    y_val,
    y_test,
    epochs=PARAMS['epochs'],
    early_stopping_epochs=PARAMS['early_stopping_epochs'],
    logger=logger
)

# Find best epoch
best_epoch = np.argmin(result['val_loss'])
logger.info(f'Val losses: {result["val_loss"]}.')
logger.info(f'Best val loss: {np.min(result["val_loss"])}.')
logger.info(best_epoch)

# Get test preds
test_preds = np.array(result['test_preds'][best_epoch])
test_labels = np.array(result['test_labels'][best_epoch])

# Save final preds and labels
with open(f"./{PARAMS['model_name']}_scores.pkl", "wb") as f:
    pickle.dump(test_preds, f)
with open(f"./{PARAMS['model_name']}_labels.pkl", "wb") as f:
    pickle.dump(test_labels, f)

# Compute final performance
evaluator = BootstrapMultiLabelMetrics(labels=test_labels, preds=test_preds)
metrics_dict = evaluator.get_all_bootstrapped_metrics_as_dict(n_bootstrap=1000)
logger.info(metrics_dict)

# Save metrics
with open(current_run_dir + 'metrics.json', "w") as f:
    json.dump(metrics_dict, f)
with open(f'./{PARAMS["model_name"]}_metrics.json', "w") as f:
    json.dump(metrics_dict, f)

# Plot learning curves from training
nresult = {k:v for k, v in result.items() if 'test' not in k}
plot_learning_curve(nresult, current_run_dir, prefix=PARAMS['model_name'])
plot_learning_curve(nresult, './', prefix=PARAMS['model_name'])

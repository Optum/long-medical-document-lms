# -*- coding: utf-8 -*-

import os
import sys
import yaml
import time
import torch
import pickle
import logging
import transformers
from datasets import DatasetDict, Dataset
from typing import List, Any, Dict
from datasets import load_dataset, load_from_disk
from transformers.data import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np
import pandas as pd
from scipy.special import expit  
from sklearn.metrics import average_precision_score

from utils import check_empty_count_gpus, create_current_run, create_log_dir
from modeling_llama_local import LlamaForSequenceClassification 
from modeling_unllama import UnmaskingLlamaForSequenceClassification

os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Load Run Parameters
with open("params.yml", "r") as stream:
    PARAMS = yaml.safe_load(stream)
    
batch_size = PARAMS["batch_size"]
gradient_accumulation_steps = PARAMS["gradient_accumulation_steps"]
learning_rate = PARAMS["learning_rate"]
lora_r = PARAMS["lora_r"]
lora_a = PARAMS["lora_a"]
max_length = PARAMS["max_length"]
warmup_steps = PARAMS["warmup_steps"]
eval_steps = PARAMS["eval_steps"]
save_steps = PARAMS["save_steps"]
logging_steps = PARAMS["logging_steps"]
early_stopping_patience = PARAMS["early_stopping_patience"]
pooling_strategy = PARAMS["pooling_strategy"]
dataset_path = PARAMS["dataset_path"]
train = PARAMS["train"]
resume_training = PARAMS["resume_training"]
resume_checkpoint = PARAMS["resume_checkpoint"]
test_checkpoint = PARAMS["test_checkpoint"]
model_id = PARAMS["model_id"]
output_path = PARAMS["output_path"]
model_name = PARAMS["model_name"]
unllama = PARAMS["unllama"]
id2label = PARAMS["id2label"]

label2id = {v: k for k, v in id2label.items()}
ds = load_from_disk(dataset_path)

# # This is to avoid using a map function which seems to be unreliable when used
# # in combination with the preprocess_function.  I should understand this better,
# # but I'm just using Pandas for now to ensure we properly transform the label column.
# def wrap_label_column(dataset):  
#     df = dataset.to_pandas()  
#     df['label'] = df['label'].apply(lambda x: [int(x)])  
#     return Dataset.from_pandas(df)  
  
# ds = DatasetDict({  
#     'train': wrap_label_column(ds['train']),  
#     'val': wrap_label_column(ds['val']),  
#     'test': wrap_label_column(ds['test']),  
# })  
    
# Define Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log CUDNN, PyTorch, and Transformers versions
logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
logger.info(f"Torch version: {torch.__version__}")
logger.info(f"Transformers version: {transformers.__version__}")

# Check, Empty, and Count GPUs
check_empty_count_gpus(logger=logger)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PARAMS["tokenizer_id"]) # hot fix

# llama doesn't have a pad token so we add one as the eos token
tokenizer.pad_token = tokenizer.eos_token

# Only create a run directory if training a new model
if train:

    # Create Run Directory
    current_run_dir = create_current_run(
        save_path=output_path, params=PARAMS, logger=logger
    )
    logger.info(f"Created run directory: {current_run_dir}.")

    # Create logging dir
    logging_dir = create_log_dir(current_run_dir)

    # Set Run Name
    run_name = current_run_dir.split("/")[-1]
    logger.info(f"Starting run {run_name}...") 

def compute_metrics(eval_pred):  
    predictions, labels = eval_pred
    sigmoid_predictions = expit(predictions) 
    micro_avg_pr_auc = average_precision_score(labels, sigmoid_predictions, average='micro')  
    return {"micro_avg_pr_auc": micro_avg_pr_auc}  

def preprocess_function(examples):
    
    return tokenizer(examples["text"], padding='longest', max_length=max_length, truncation=True)

tokenized_ds = ds.map(preprocess_function, batched=True)

# this is messing with things: https://huggingface.co/docs/transformers/en/main_classes/data_collator
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

df = tokenized_ds['train'].to_pandas()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

# Train
if train:
    
    if unllama:
        model = UnmaskingLlamaForSequenceClassification.from_pretrained(model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id).bfloat16()
        model.set_pooling(pooling_strategy)
    else:
        model = LlamaForSequenceClassification.from_pretrained(model_id, num_labels=len(label2id)).bfloat16()
        
    # set the pad token of the model's configuration
    # https://stackoverflow.com/questions/68084302/assertionerror-cannot-handle-batch-sizes-1-if-no-padding-token-is-defined
    model.config.pad_token_id = model.config.eos_token_id
    
    # # Set problem type explicitly 
    # model.config.problem_type = "binary_classification"
        
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_r, lora_alpha=lora_a, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=current_run_dir,
        max_steps=1000000000,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        metric_for_best_model="eval_loss",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        logging_dir=logging_dir,
        lr_scheduler_type='linear',
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=0.00000001,
        optim="adamw_torch",
        load_best_model_at_end=True,
        push_to_hub=False,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        label_names='label'
    )
    
    # Define early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["val"],
        callbacks=[early_stopping],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # data_collator=data_collator,
    
    # Start training timer
    start_time = time.time()

    # Start from model provided above and new training parameters defined above
    if not resume_training:
        trainer.train()

    # Resume using model and training parameters defined in checkpoint
    else:
        trainer.train(resume_checkpoint)
        
    # Log training time
    end_time = time.time()
    execution_time_hours = round((end_time - start_time) / 3600.0, 2)
    logger.info(f"Training took {execution_time_hours} hours.")

    # Save best model
    trainer.model.save_pretrained(
        os.path.join(current_run_dir, f'{model_name}_model')
    )
    tokenizer.save_pretrained(
        os.path.join(current_run_dir, f'{model_name}_tokenizer')
    )

# Test
else:
    
    if unllama:
        model = UnmaskingLlamaForSequenceClassification.from_pretrained(test_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id).bfloat16()
        model.set_pooling(pooling_strategy)  
    else:
        model = LlamaForSequenceClassification.from_pretrained(test_checkpoint, num_labels=len(label2id)).bfloat16()
        
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=True, r=lora_r, lora_alpha=lora_a, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = Trainer(model=model)

# Predict on test data
# Apply softmax or sigmoid to these outputs!
output = trainer.predict(tokenized_ds["test"])
labels = output.label_ids
probs = torch.tensor(output.predictions)

with open(f"./{model_name}_raw_scores.pkl", "wb") as f:
    pickle.dump(probs.cpu().detach().numpy(), f)
with open(f"./{model_name}_raw_labels.pkl", "wb") as f:
    pickle.dump(labels, f)
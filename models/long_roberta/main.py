#!/usr/bin/env python
# coding: utf-8

"""
Main BERT classes and functions
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast, AutoModel, AdamW
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from transformers.optimization import get_linear_schedule_with_warmup
from architecture import BERTSequenceClassificationArch
from base_model import Model
from custom_datasets import TokenizedDataset, collate_fn_pooled_tokens
from text_preprocessors import BERTTokenizer, BERTTokenizerPooled

class BERTClassificationModel(Model):
    def __init__(self):
        super().__init__()

        with open("params.yml", "r") as stream:
            params = yaml.safe_load(stream)

        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizer(tokenizer)
        self.dataset_class = TokenizedDataset
        self.nn = initialize_model(bert, self.params['device'])
        self.optimizer = AdamW(
            self.nn.parameters(),
            lr=self.params['learning_rate'],
            betas=(self.params['adam_beta1'], self.params['adam_beta2']),
            weight_decay=self.params['weight_decay'],
            eps=self.params['adam_epsilon']
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.params['warmup_steps'],
            num_training_steps=1000000000000
        )

    def evaluate_single_batch(self, batch, model, device):

        # Push the batch to gpu
        batch = [t.to(device) for t in batch]

        # Predict
        model_input = batch[:-1]
        labels = batch[-1]
        preds = model(*model_input).cpu()
        labels = labels.float().cpu()

        return preds, labels


class BERTClassificationModelWithPooling(Model):
    def __init__(self):
        super().__init__()

        with open("params.yml", "r") as stream:
            params = yaml.safe_load(stream)

        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooled(
            tokenizer, params['size'], params['step'], params['minimal_length'], params['max_num_segments']
        )
        self.dataset_class = TokenizedDataset
        self.collate_fn = collate_fn_pooled_tokens
        self.nn = initialize_model(bert, self.params['device'])
        self.optimizer = AdamW(
            self.nn.parameters(),
            lr=self.params['learning_rate'],
            betas=(self.params['adam_beta1'], self.params['adam_beta2']),
            weight_decay=self.params['weight_decay'],
            eps=self.params['adam_epsilon']
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.params['warmup_steps'],
            num_training_steps=1000000000000
        )

    def evaluate_single_batch(self, batch, model, device):

        # Extract elements from batch
        input_ids = batch[0]
        attention_mask = batch[1]
        number_of_chunks = [len(x) for x in input_ids]
        labels = batch[2]

        # Concatenate all input_ids into one batch
        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined])

        # Concatenate all attention maska into one batch
        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())
        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined])

        # Get model predictions for the combined batch
        preds = model(
            input_ids_combined_tensors,
            attention_mask_combined_tensors
        )

        # Move predictions to CPU
        preds = preds.cpu()

        if self.params['num_labels'] > 1:

            # Split result preds into chunks
            preds_split = torch.split(preds, number_of_chunks)

            # Pooling - torch.max return tuples where the first element is the aggregate value
            if self.params['pooling_strategy'] == 'mean':
                pooled_preds = torch.stack([torch.mean(x, dim=0) for x in preds_split])
            elif self.params['pooling_strategy'] == 'max':
                pooled_preds = torch.stack([torch.max(x, dim=0)[0] for x in preds_split])
            elif self.params['pooling_strategy'] == 'custom_agg':
                c = self.params['custom_agg_c']
                pooled_preds = torch.stack([
                    (torch.max(x, dim=0)[0] + torch.mean(x, dim=0) * number_of_chunks[i]/c) / (1 + number_of_chunks[i]/c) for i, x in enumerate(preds_split)
                ])
            else:
                raise ValueError(f"Expected pooling strategy to be one of ['mean', 'max', 'custom_agg'] but got {self.params['pooling_strategy']}.")

        else:

            # Flatten preds
            preds = preds.flatten()

            # Split result preds into chunks
            preds_split = torch.split(preds, number_of_chunks)

            # Pooling - torch.max return tuples where the first element is the aggregate value
            if self.params['pooling_strategy'] == 'mean':
                pooled_preds = torch.stack([torch.mean(x).reshape(1) for x in preds_split])
            elif self.params['pooling_strategy'] == 'max':
                pooled_preds = torch.stack([torch.max(x).reshape(1) for x in preds_split])
            else:
                raise ValueError(f"Expected pooling strategy to be one of ['mean', 'max'] but got {self.params['pooling_strategy']}.")

        # Move labels to CPU
        labels_detached = torch.tensor(labels).float()

        return pooled_preds, labels_detached

def load_pretrained_model():

    tokenizer = load_tokenizer()
    model = load_bert()

    return tokenizer, model

def load_tokenizer():

    with open("params.yml", "r") as stream:
        params = yaml.safe_load(stream)

    tokenizer = AutoTokenizer.from_pretrained(params['tokenizer_path'])

    return tokenizer

def load_bert():

    with open("params.yml", "r") as stream:
        params = yaml.safe_load(stream)
    
    model = AutoModel.from_pretrained(
        params['bert_path'],
        num_labels=params['num_labels'],
        return_dict=True
    )

    return model

def initialize_model(bert, device):

    model = BERTSequenceClassificationArch(bert)
    model = model.to(device)
    model = nn.DataParallel(model)

    return model

#!/usr/bin/env python
# coding: utf-8

"""
Base PyTorch model code for training and evaluation
"""

import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class Model():
    """
    Abstract class for models
    """

    def __init__(self):
        
        with open("params.yml", "r") as stream:
            params = yaml.safe_load(stream)
        
        self.params = params
        self.preprocessor = None
        self.dataset_class = None
        self.collate_fn = None

    def evaluate_single_batch(self, batch, model, device):
        
        raise NotImplementedError("This is implemented for subclasses only")

    def create_dataset(self, X_preprocessed, y):
        
        dataset = self.dataset_class(X_preprocessed, y)
        
        return dataset

    def train_and_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test, epochs, early_stopping_epochs, logger):
        
        # Compute number of samples
        number_of_train_samples = len(X_train)
        number_of_val_samples = len(X_val)
        number_of_test_samples = len(X_test)
        
        # Text preprocessing
        X_train_preprocessed = self.preprocessor.preprocess(X_train)
        X_val_preprocessed = self.preprocessor.preprocess(X_val)
        X_test_preprocessed = self.preprocessor.preprocess(X_test)
        
        # Creating datasets
        train_dataset = self.create_dataset(X_train_preprocessed, y_train)
        val_dataset = self.create_dataset(X_val_preprocessed, y_val)
        test_dataset = self.create_dataset(X_test_preprocessed, y_test)
        
        # Creating dataloaders
        train_dataloader = create_train_dataloader(
            train_dataset, self.params['batch_size'], self.collate_fn)
        val_dataloader = create_train_dataloader(
            val_dataset, self.params['batch_size'], self.collate_fn)
        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)
        
        # Training and evaluating
        result = self.train_and_evaluate_preprocessed(
            number_of_train_samples,
            train_dataloader,
            number_of_val_samples,
            val_dataloader,
            number_of_test_samples,
            test_dataloader,
            epochs,
            early_stopping_epochs,
            logger
        )
        
        return result

    def train_and_evaluate_preprocessed(
            self,
            number_of_train_samples,
            train_dataloader,
            number_of_val_samples,
            val_dataloader,
            number_of_test_samples,
            test_dataloader,
            epochs,
            early_stopping_epochs,
            logger
    ):
        
        result = {
            'train_loss': [],
            'val_loss': [],
            'test_preds': [],
            'test_labels': []
        }
        
        for epoch in range(epochs):
            
            # Run train epoch
            avg_loss, avg_lr = self.train_single_epoch(number_of_train_samples, train_dataloader)
            logger.info(f'Epoch: {epoch}, Train Loss: {avg_loss:.10f}, Avg LR: {avg_lr:.10f}')
            result['train_loss'].append(avg_loss)
            
            # Evaluate
            avg_loss, _, _ = self.evaluate_single_epoch(number_of_val_samples, val_dataloader)
            logger.info(f'Epoch: {epoch}, Val Loss: {avg_loss:.10f}')
            result['val_loss'].append(avg_loss)
            
            # Predict on test set and save (we should really only do this at the end but need to save the model somehow first)
            preds, labels = self.predict(number_of_test_samples, test_dataloader, with_labels=True)
            result['test_preds'].append(preds)
            result['test_labels'].append(labels)
            
            # Compute best epoch
            best_epoch = np.argmin(result['val_loss'])
            best_val_loss = np.min(result["val_loss"])
            epochs_since_best = result['val_loss'][best_epoch:]
            
            # Early stop if too many epochs have passed since the best epoch (we should also checkpoint the model here)
            if len(epochs_since_best) > early_stopping_epochs:
                logger.info(f"Stopping at epoch {epoch}.  Best val loss of {best_val_loss:.10f} occurred at epoch {best_epoch}.")
                return result
            
        return result

    def predict(self, number_of_test_samples, test_dataloader, with_labels=False):
        
        # Predict on test data loader
        _, preds, labels = self.evaluate_single_epoch(number_of_test_samples, test_dataloader)
        
        # Return labels if specificed
        if with_labels:
            return preds, labels
        else:
            return preds

    def train_single_epoch(self, number_of_train_samples, train_dataloader):
        
        model = self.nn
        model.train()

        total_loss = 0
        # total_micro_f1 = 0
        # total_macro_f1 = 0
        total_lr = 0

        # Iterate over batches
        for step, batch in enumerate(train_dataloader):

            preds, labels = self.evaluate_single_batch(batch, model, self.params['device'])

            # Compute the loss between actual and predicted values
            loss = compute_loss(preds, labels)

            # Backward pass to calculate the gradients
            loss.backward()
            
            # Add to total loss
            total_loss += loss.detach().cpu().numpy()
            
            # # Accumulate gradients
            # step_plus_one = step + 1
            if (step + 1) % self.params['accumulation_steps'] == 0:

                # Update parameters
                self.optimizer.step()
                self.scheduler.step()

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Add LR at step
                total_lr += self.optimizer.param_groups[0]['lr']

        # Compute the train loss of the epoch
        avg_loss = total_loss / number_of_train_samples
        avg_lr = total_lr / number_of_train_samples
        
        return avg_loss, avg_lr

    def evaluate_single_epoch(self, val_samples, val_dataloader):
        
        model = self.nn
        model.eval()
        
        total_loss = 0
        preds_total = []
        labels_total = []

        # Iterate over batches
        for step, batch in enumerate(val_dataloader):

            # Deactivate autograd
            with torch.no_grad():
                
                # Generate predictions
                preds, labels = self.evaluate_single_batch(batch, model, self.params['device'])
                preds_total.extend(preds)
                labels_total.extend(labels)

                # Compute the validation loss between actual and predicted values
                loss = compute_loss(preds, labels)
                total_loss += loss.detach().cpu().numpy()

        # Compute the evaluation loss of the epoch
        preds_total = [x.tolist() for x in preds_total]
        labels_total = [x.tolist() for x in labels_total]
        avg_loss = total_loss / val_samples
        
        return avg_loss, preds_total, labels_total


def create_dataloader(data, sampler_class, batch_size, collate_fn=None):
    
    sampler = sampler_class(data)
    dataloader = DataLoader(
        data,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn)
    
    return dataloader

def create_train_dataloader(train_data, batch_size, collate_fn=None):
    
    train_dataloader = create_dataloader(
        train_data, RandomSampler, batch_size, collate_fn)
    
    return train_dataloader

def create_val_dataloader(val_data, batch_size, collate_fn=None):
    
    val_dataloader = create_dataloader(
        val_data, SequentialSampler, batch_size, collate_fn)
    
    return val_dataloader

def create_test_dataloader(test_data, batch_size, collate_fn=None):
    
    test_dataloader = create_dataloader(
        test_data, SequentialSampler, batch_size, collate_fn)
    
    return test_dataloader


def create_dataloaders(train_data, val_data, batch_size, collate_fn=None):

    train_dataloader = create_train_dataloader(
        train_data, batch_size, collate_fn)
    val_dataloader = create_val_dataloader(val_data, batch_size, collate_fn)

    return train_dataloader, val_dataloader


def compute_loss(preds, labels):
    
    loss = F.binary_cross_entropy(preds, labels.type_as(preds), reduction='sum')
    
    return loss


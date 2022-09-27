#!/usr/bin/env python
# coding: utf-8

"""
Bootstrapped multi-label metrics functions for test set evaluation and
function to compute metrics during model training.
"""

import torch
import numpy as np
import torch.nn.functional as F
from scipy import interp
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    auc,
    roc_curve,
    f1_score,
)
from utils import convert_1d_binary_labels_to_2d


class BootstrapMultiLabelMetrics(object):
    """
    Class containing methods for evaluating performance
    of multi-label classifiers by bootstrapping the test set.

    :param labels: 2d numpy array of true labels
    :type labels: :class:`numpy.ndarray`
    :param preds: 2d numpy array of predicted probabilities for each label
    :type preds: :class:`numpy.ndarray`
    """

    def __init__(self, labels, preds):

        self.labels = labels
        self.preds = preds

    def assert_2d_array(self):
        """
        Check that labels and preds are 2d numpy arrays.
        """

        assert_msg = "Make sure labels and preds are 2d numpy arrays.  Use np.stack(array) if passing an array of arrays."
        assert len(self.labels.shape) == len(self.preds.shape) == 2, assert_msg

    def get_bootstrapped_average_precision(self, n_bootstrap=1000):
        """
        Bootstrap sample the predictions and labels to
        compute micro and macro average precisions across all
        labels with the average and standard deviation of
        these values across all boostrap iterations.

        :return: micro_average_precision_mean_stdv, macro_average_precision_mean_stdv
        :rtype: (dict, dict)
        """

        # Ensure labels and preds are 2d arrays
        self.assert_2d_array()

        # Run bootstrap iterations
        micro_average_precision_mean_stdv, macro_average_precision_mean_stdv = {}, {}
        micro_average_precisions, macro_average_precisions = [], []
        for i in range(n_bootstrap):

            # Sample N records with replacement where N is the total number of records
            sample_indices = np.random.choice(len(self.labels), len(self.labels))
            sample_labels = self.labels[sample_indices]
            sample_preds = self.preds[sample_indices]

            micro_average_precision = average_precision_score(
                sample_labels, sample_preds, average="micro"
            )
            micro_average_precisions.append(micro_average_precision)

            macro_average_precision = average_precision_score(
                sample_labels, sample_preds, average="macro"
            )
            macro_average_precisions.append(macro_average_precision)

        # Compute means and stdvs
        micro_average_precision_mean_stdv["mean"] = np.mean(micro_average_precisions)
        micro_average_precision_mean_stdv["stdv"] = np.std(micro_average_precisions)
        macro_average_precision_mean_stdv["mean"] = np.mean(macro_average_precisions)
        macro_average_precision_mean_stdv["stdv"] = np.std(macro_average_precisions)

        return micro_average_precision_mean_stdv, macro_average_precision_mean_stdv

    def get_bootstrapped_roc_auc(self, n_bootstrap=1000):
        """
        Bootstrap sample the predictions and labels to
        compute micro and macro ROC AUC across all
        labels with the average and standard deviation of
        these values across all boostrap iterations.

        :return: micro_roc_auc_mean_stdv, macro_roc_auc_mean_stdv
        :rtype: (dict, dict)
        """

        # Ensure labels and preds are 2d arrays
        self.assert_2d_array()

        # Get number of classes
        n_classes = self.labels.shape[1]

        # Run bootstrap iterations
        micro_roc_auc_mean_stdv, macro_roc_auc_mean_stdv = {}, {}
        micro_roc_aucs, macro_roc_aucs = [], []
        for i in range(n_bootstrap):

            # Sample N records with replacement where N is the total number of records
            sample_indices = np.random.choice(len(self.labels), len(self.labels))
            sample_labels = self.labels[sample_indices]
            sample_preds = self.preds[sample_indices]

            # Compute micro average ROC AUC
            fpr_micro, tpr_micro, _ = roc_curve(
                sample_labels.ravel(), sample_preds.ravel()
            )
            micro_roc_auc = auc(fpr_micro, tpr_micro)
            micro_roc_aucs.append(micro_roc_auc)

            # Compute fpr, tpr for each class
            fpr, tpr = {}, {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(sample_labels[:, i], sample_preds[:, i])

            # Compute macro-average ROC AUC using fprs and tprs
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            macro_roc_auc = auc(all_fpr, mean_tpr)
            macro_roc_aucs.append(macro_roc_auc)

        # Compute means and stdvs
        micro_roc_auc_mean_stdv["mean"] = np.mean(micro_roc_aucs)
        micro_roc_auc_mean_stdv["stdv"] = np.std(micro_roc_aucs)
        macro_roc_auc_mean_stdv["mean"] = np.mean(macro_roc_aucs)
        macro_roc_auc_mean_stdv["stdv"] = np.std(macro_roc_aucs)

        return micro_roc_auc_mean_stdv, macro_roc_auc_mean_stdv

    def get_bootstrapped_f1(self, n_bootstrap=1000):
        """
        Bootstrap sample the predictions and labels to
        compute micro and macro F1 across all
        labels with the average and standard deviation of
        these values across all boostrap iterations.

        :return: micro_f1_mean_stdv, macro_f1_mean_stdv
        :rtype: (dict, dict)
        """

        # Ensure labels and preds are 2d arrays
        self.assert_2d_array()

        # Get number of classes
        n_classes = self.labels.shape[1]

        # Run bootstrap iterations
        threshold = 0.5
        micro_f1_mean_stdv, macro_f1_mean_stdv = {}, {}
        micro_f1s, macro_f1s = [], []
        for i in range(n_bootstrap):

            # Sample N records with replacement where N is the total number of records
            sample_indices = np.random.choice(len(self.labels), len(self.labels))
            sample_labels = self.labels[sample_indices]
            sample_preds = self.preds[sample_indices]

            # Compute f1s
            preds_at_threshold = np.array((sample_preds >= threshold), dtype=int)
            micro_f1 = f1_score(sample_labels, preds_at_threshold, average="micro")
            micro_f1s.append(micro_f1)
            macro_f1 = f1_score(sample_labels, preds_at_threshold, average="macro")
            macro_f1s.append(macro_f1)

        # Compute means and stdvs
        micro_f1_mean_stdv["mean"] = np.mean(micro_f1s)
        micro_f1_mean_stdv["stdv"] = np.std(micro_f1s)
        macro_f1_mean_stdv["mean"] = np.mean(macro_f1s)
        macro_f1_mean_stdv["stdv"] = np.std(macro_f1s)

        return micro_f1_mean_stdv, macro_f1_mean_stdv

    def get_all_bootstrapped_metrics_as_dict(self, n_bootstrap=1000):
        """
        Returns all bootstrapped metrics in a nice dictionary.
        :return: metrics_dict
        :rtype: dict
        """

        (
            micro_average_precision_mean_stdv,
            macro_average_precision_mean_stdv,
        ) = self.get_bootstrapped_average_precision(n_bootstrap=n_bootstrap)
        (
            micro_roc_auc_mean_stdv,
            macro_roc_auc_mean_stdv,
        ) = self.get_bootstrapped_roc_auc(n_bootstrap=n_bootstrap)
        micro_f1_mean_stdv, macro_f1_mean_stdv = self.get_bootstrapped_f1(
            n_bootstrap=n_bootstrap
        )

        metrics_dict = {}
        metrics_dict["micro_ap_mean"] = micro_average_precision_mean_stdv["mean"]
        metrics_dict["micro_ap_stdv"] = micro_average_precision_mean_stdv["stdv"]
        metrics_dict["macro_ap_mean"] = macro_average_precision_mean_stdv["mean"]
        metrics_dict["macro_ap_stdv"] = macro_average_precision_mean_stdv["stdv"]
        metrics_dict["micro_roc_auc_mean"] = micro_roc_auc_mean_stdv["mean"]
        metrics_dict["micro_roc_auc_stdv"] = micro_roc_auc_mean_stdv["stdv"]
        metrics_dict["macro_roc_auc_mean"] = macro_roc_auc_mean_stdv["mean"]
        metrics_dict["macro_roc_auc_stdv"] = macro_roc_auc_mean_stdv["stdv"]
        metrics_dict["micro_f1_mean"] = micro_f1_mean_stdv["mean"]
        metrics_dict["micro_f1_stdv"] = micro_f1_mean_stdv["stdv"]
        metrics_dict["macro_f1_mean"] = macro_f1_mean_stdv["mean"]
        metrics_dict["macro_f1_stdv"] = macro_f1_mean_stdv["stdv"]

        return metrics_dict


def compute_training_metrics(pred, class_strategy, threshold=0.5, task="ft"):
    """
    Returns dictionary of metrics computed during training
    :return: training_metrics
    :rtype: dict
    """

    #
    if task == "pt":

        # Compute loss
        y_prob = torch.tensor(pred.predictions)
        y_true = torch.tensor(pred.label_ids).type_as(y_prob)
        loss = F.binary_cross_entropy_with_logits(y_prob, y_true).numpy().item()

        # Build metrics dict
        training_metrics = {"loss": loss}

        return training_metrics

    elif task == "ft":

        # Compute f1s
        labels = pred.label_ids
        preds_at_threshold = np.array((pred.predictions >= threshold), dtype=int)

        # Convert a 1D, binary label array to 2D if it's not already
        if class_strategy == "multi_label" or class_strategy == "multi_class":
            labels = convert_1d_binary_labels_to_2d(labels)

        micro_f1 = f1_score(labels, preds_at_threshold, average="micro")
        macro_f1 = f1_score(labels, preds_at_threshold, average="macro")

        # Compute loss
        y_prob = torch.tensor(pred.predictions, dtype=torch.float32)
        y_true = torch.tensor(labels).type_as(y_prob)

        if class_strategy == "multi_label":
            loss = F.binary_cross_entropy_with_logits(y_prob, y_true).numpy().item()
        elif class_strategy == "multi_class":
            loss = F.cross_entropy(y_prob, y_true).numpy().item()
        else:
            raise ValueError(
                f"Expected class_strategy to be one of ['multi_label', 'multi_class'] but got {class_strategy}."
            )

        # Build metrics dict
        training_metrics = {"micro_f1": micro_f1, "macro_f1": macro_f1, "loss": loss}

        return training_metrics

    else:

        raise ValueError(f"Expected task to be one of ['pt', 'ft'] but got {task}.")

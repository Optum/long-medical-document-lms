#!/usr/bin/env python
# coding: utf-8

"""
Automatically Evaluate Explainability Algorithms

Auto evalaution is based on [Murdoch et al.](https://arxiv.org/pdf/1801.05453.pdf) and [Jin et al.](https://arxiv.org/pdf/1911.06194.pdf).
The idea is to compute the correlation between importance scores from each explainability algorithm and coefficients from Logistic Regression.
Here we use a Logisitic Regression model trained on multi-token blocks (n-grams) of text from each document in the dataset.
A better evaluation requires human-annotators with domain expertise but can be conducted by providing a blinded dataset of
the top K most important text blocks from each algorithm for each label, document pair in a representative subset of the test data.
Note that a major limitation of evaluation with Logisitc Regression (among others) is that multicollinearity will distort feature weights.
"""

# Open imports
import os
import glob
import yaml
import shutil
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def main():

    # Load Run Parameters
    with open("ae_params.yml", "r") as stream:
        PARAMS = yaml.safe_load(stream)

    # Define Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load N-Gram Feature Weights from Logisitic Regression
    with open(PARAMS["lr_sorted_pairs"], "rb") as f:
        sorted_pairs = pickle.load(f)

    # Load Results from Explainability Experiments
    all_info_dfs = []
    exp_data_dfs = []
    for i, results_dir in enumerate(
        [PARAMS["rnd_results"], PARAMS["soc_results"], PARAMS["msp_results"]]
    ):
        for j, file in enumerate(glob.glob(os.path.abspath(results_dir + "*"))):
            if "all_info" in file:

                # Read all info file
                logger.info(f"Processing file: {file}...")
                d_all_info_df = pd.read_csv(file)
                d_all_info_df["row_id"] = d_all_info_df["row_id"].apply(
                    lambda x: f"{i}_{j}_" + str(x)
                )
                all_info_dfs.append(d_all_info_df)

                # Read exp data file
                companion_file = file.replace("all_info", "exp_data")
                logger.info(f"Processing companion file: {companion_file}...")
                d_exp_data_df = pd.read_csv(companion_file)
                d_exp_data_df["row_id"] = d_exp_data_df["row_id"].apply(
                    lambda x: f"{i}_{j}_" + str(x)
                )
                exp_data_dfs.append(d_exp_data_df)

    # At this point, the experiment dataframe can be used to run a blind experiment with human annotators to determine the informativeness of text blocks for each label.
    #  Before providing the experiment data:
    # - Sample the input data such that each algorithm is represented the same number of times.
    # - Ensure a random global ID exists that can map samples in the `all_info` dataframe to the experiment dataframe.
    # - Check power to detect differences in the number of informative samples from different algorithms.
    # - Hide any columns or IDs (potentially the original `row_id`) that could tip off human reviewers.
    # - Shuffle rows in the experiment dataframe.

    # Build Combined Dataframe
    all_info_df = pd.concat(all_info_dfs)
    exp_data_df = pd.concat(exp_data_dfs)
    combined_df = pd.merge(
        all_info_df,
        exp_data_df.drop(["label", "text_block"], axis=1),
        on="row_id",
        how="inner",
    )
    logger.info(all_info_df.shape)
    logger.info(exp_data_df.shape)
    logger.info(combined_df.shape)

    # In the absense of human annotators for this demo notebook, we use the procedure from [Murdoch et al.](https://arxiv.org/pdf/1801.05453.pdf)
    # and [Jin et al.](https://arxiv.org/pdf/1911.06194.pdf) to compute the correlation between importance scores from each explainability algorithm
    # and coefficients from Logistic Regression.

    # Subset to algorithms to evaluate
    rnd_df = combined_df[combined_df["P"] == "RND"][["text_block", "avg_score_diff"]]
    soc_df = combined_df[combined_df["P"] == "SOC"][["text_block", "avg_score_diff"]]
    msp_df = combined_df[combined_df["P"] == 0.1][["text_block", "avg_score_diff"]]

    # Create lr coefficient dataframe
    lr_coef_df = pd.DataFrame(sorted_pairs, columns=["text_block", "coefficient"])

    # Run Automated Evaluation
    for algo, df in zip(["RND", "SOC", "MSP"], [rnd_df, soc_df, msp_df]):
        joined_df = pd.merge(lr_coef_df, df, how="inner", on="text_block")
        corr, p = pearsonr(
            joined_df["coefficient"].tolist(), joined_df["avg_score_diff"].tolist()
        )
        logger.info(
            f"Algo: {algo} correlation with logistic regression coefficients: {round(corr, 5)} (p={round(p, 5)})."
        )


if __name__ == "__main__":

    main()

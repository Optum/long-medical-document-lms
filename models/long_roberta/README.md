# torch_long_bert

Code to fine-tune and evaluate long versions of BERT and BERT-like LMs from Hugging Face Transformers

### Contents

- [About](#about)
- [Environment](#environment)
- [Data Prep](#data-prep)
- [Train and Evaluate](#train-and-evaluate)

### About

This repository contains code to fine-tune and evaluate long versions of BERT and BERT-like LMs from Hugging Face Transformers using base PyTorch.  The code in this directory has been modified from [this repository](https://github.com/mim-solutions/roberta_for_longer_texts)
and was originally written by [MichalBrzozowski91](https://github.com/MichalBrzozowski91) to implement [this suggestion](https://github.com/google-research/bert/issues/27#issuecomment-435265194) from [jacobdevlin-google](https://github.com/jacobdevlin-google).  The core idea is to fine-tune a base BERT model by getting the representations from multiple concatenated windows of text with some overlap and applying sigmoid over each window to generate predictions.  The final predictions are then taken as either the average or max value of the sigmoid output of all windows in a sample.

### Environment

To build the Python 3.10 environment required to run this code, create a Python 3.10 virtual environment with [Anaconda](https://www.anaconda.com/products/individual) and install the dependencies in `../requirements.txt`. 

### Data Prep

This code takes as input a HuggingFace dataset with text and label columns.

### Training

Training and evaluation are combined into one script.  After modifying `params.yml`, run `python train_and_evaluate.py` to fine-tune a long version of a BASE BERT model specified in `params.yml`.  Make sure the BERT model you wish to fine-tune exists on the file system from which you run `train_and_evaluate.py`.  Predictions on the test set are generated after every epoch but only used for the best model checkpoint to compute test set performance.  This behavior could be adjusted to improve training efficiency, but because checkpoints are not actually saved, it would be necessary to implement checkpoint saving and loading in the code first.

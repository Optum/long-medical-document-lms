# General Implementation of MSP to Compute Text Block Importance for Fine-Tuned LMs

[![python38](https://img.shields.io/badge/python-3.8-orange.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Scripts to compute the importance of individual text blocks in the classification of long documents using fine-tuned LMs.  

### About

This codebase contains scripts that compute the importance of text blocks to LM predictions for a set of documents using a variety of methods.
We propose a novel method called Masked Sampling Procedure (MSP) and compare it to 1) random sampling and 2) the [Sampling and Occlusion (SOC) algorithm
from Jin et al.](https://arxiv.org/pdf/1911.06194.pdf)  See our paper, [*Extend and Explain: Interpreting Very Long Language Models*](https://arxiv.org/abs/2209.01174),
for more details.

The code currently supports [HuggingFace LMs](https://huggingface.co/models) and [Datasets](https://huggingface.co/datasets) and would require slight modifications to use other types of models and input data.  If you need to fine-tune or continue pretraining an existing LM, check out `models/README.md`.  To create a Hugging Face Dataset, check out the documentation [here](https://huggingface.co/docs/datasets/index).

### Environment

All scripts are intended to be run in a Python 3.8 conda environment.  To create such an environment run `conda create --name text-blocks python=3.8` then `source activate text-blocks` to activate the environment. Dependencies can be installed from `requirements.txt` by running `pip install -r requirements.txt` from the base directory of this repository.  

The explainability experiment scripts are equipped to run LM inference on a single GPU VM and have been tested on Azure `Standard_NC6s_v3` machines which have 16 GB GPU RAM, while the training scripts in `models/` (except for `lr.py`) use [PyTorch Data Parallelism](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) to distribute training over a single node with multiple GPUs. LM training scripts were tested on Azure `Standard_ND40rs_v2` machines which have 32 GB GPU RAM per GPU and 8 GPUs in total.  

### Experiment Parameters

Input parameters such as data and models as well as MSP and SOC parameters described in our paper and [Jin et al.](https://arxiv.org/pdf/1911.06194.pdf) can be found in `explain/params.yml`.  Adjust the input data path, trained classifier path, and LM path (in the case of SOC), as well as the parameters for the explainability algorithms before running each script.

### Running Experiments

Explainability scripts can be run from the `explain` directory as follows:

- To compute text block importance with MSP, adjust the MSP parameters in `params.yml`, then run `python explain_with_msp.py`.  
- To compute text block importance with SOC, adjust the SOC parameters in `params.yml`, then run `python explain_with_soc.py`.

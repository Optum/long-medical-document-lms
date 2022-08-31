# Text Block Importance for Big Bird Long LM

[![python38](https://img.shields.io/badge/python-3.8-orange.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Scripts to compute the importance of individual text blocks according to a fine-tuned [Big Bird LM](https://huggingface.co/google/bigbird-roberta-base).

### About

The code in this subdirectory was used in XXX to generate explanations used the Masked Sampling Procedure (MSP) and compare it to 1) random sampling and 2) the [Sampling and Occlusion (SOC) algorithm from Jin et al.](https://arxiv.org/pdf/1911.06194.pdf)

### Environment

These scripts were run in a Python 3.8 conda environment.  To create such an environment run `conda create --name text-blocks python=3.8` then `source activate text-blocks` to activate the environment. Dependencies can be installed from `requirements.txt` by running `pip install -r requirements.txt` from the base directory of this repository.  The experiment scripts are equipped to run LM inference on a single GPU VM and have been tested on Azure `Standard_NC6s_v3` machines.

### Running Experiments

Scripts can be run one at a time:

- To compute text block importance with MSP, run `python gen_blocks_msp.py`.  Arguments are in capital letters at the top of the script following imports.  This script can optionally output randomly sampled text blocks for comparison.
- To compute text block importance with SOC, run `python gen_blocks_soc.py`.  Arguments are in capital letters at the top of the script following imports.

To quickly run experiments on multiple machines:

1. Launch each machine.
2. Run `export BASE_PATH=<path-to-project-directory-on-machine>`.
3. Run `bash $BASE_PATH/launch_msp_soc.sh`.

Step #3 will remove any old conda environments called `text-blocks`, build a new environment called `text-blocks`, install the Python library requirements, and kick off the experiment scripts.

### Runtimes

We found the time it takes to compute the importance score of text blocks of size K=10 subword tokens where for the SOC algorithm we sample 100 contexts per text block from a 10-block radius and for the MSP algorithm the expected number of times a given text block is masked is 100.  Runtimes were averaged over 20 randomly sampled documents of median length 1,429.5 tokens (IQR [1,029-1,929]).  For the MSP algorithm, we observed that masking probability P=0.1 yields slightly better performance than P=0.5 but is ~5x slower.  The SOC algorithm is ~20x slower than MSP with P=0.1 and ~100x slower than MSP with P=0.5.

| Algorithm | Mean Observed Runtime | Stdv. Observed Runtime |
|-----------|-----------------------|------------------------|
| SOC | 17.81 hours | 6.05 hours |
| MSP (P=0.1) | 0.89 hours | 0.05 hours |
| MSP (P=0.5) | **0.18 hours** | **0.01 hours** |

Below are mean runtimes over 20 experiments for each algorithm on documents of various fixed lengths.  Standard deviation is reported in parentheses.  Note the rapid increase in SOC runtimes, even at these modest document lengths, making SOC intractable for very long documents.

| Algorithm | 50 Token Document Runtime | 100 Token Document Runtime | 500 Token Document Runtime | 1,000 Token Document Runtime |
|-----------|---------------------------|----------------------------|----------------------------|------------------------------|
| SOC | 0.31 (0.05) mins | 0.47 (0.04) mins | 2.17 (0.03) mins | 65.49 (0.59) mins |
| MSP (P=0.1) | 0.27 (0.03) mins | 0.26 (0.03) mins | 0.33 (0.02) mins | 6.31 (0.08) mins |
| MSP (P=0.5) | **0.11 (0.02) mins** | **0.12 (0.02) mins** | **0.12 (0.02) mins** | **1.41 (0.04) mins** |

# Long Medical Document LMs

[![python38](https://img.shields.io/badge/python-3.8-orange.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Train and explain language models that extract information from long medical documents with the Masked Sampling Procedure (MSP)

### Contents

- [About](#about)
- [Environment](#environment)
- [LM Training](#lm-training)
- [Generating Explanations](#generating-explanations)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Maintainers](#maintainers)
- [Citation](#citation)

### About

This codebase contains scripts to:

1. Pretrain language models (LMs) in a self-supervised fashion using masked language modeling (MLM) and fine-tune these LMs for document classification.
2. Compute the importance of multi-token text blocks to fine-tuned LM predictions for a given document or set of documents using a variety of methods.

In our paper, [*Extend and Explain: Interpreting Very Long Language Models*](https://arxiv.org/abs/2209.01174), we propose a novel method called the Masked Sampling Procedure (MSP) and compare it to 1) random sampling and 2) the [Sampling and Occlusion (SOC) algorithm
from Jin et al.](https://arxiv.org/pdf/1911.06194.pdf).  MSP is well-suited to very long, sparse-attention LMs, and has been validated for medical documents using two physician annotators.  

The code to run MSP currently supports [HuggingFace LMs](https://huggingface.co/models) and [Datasets](https://huggingface.co/datasets) and would require slight modifications to use other types of models and input data.  If you need to fine-tune or continue pretraining an existing LM, check out `models/README.md`.  To create a Hugging Face Dataset, check out the documentation [here](https://huggingface.co/docs/datasets/index).

### Environment

All scripts are intended to be run in a Python 3.8 [Anaconda](https://www.anaconda.com/products/individual) environment.  To create such an environment run `conda create --name text-blocks python=3.8` then `source activate text-blocks` to activate the environment. Dependencies can be installed from `requirements.txt` by running `pip install -r requirements.txt` from the base directory of this repository.  

The explainability experiment scripts are equipped to run LM inference on a single GPU VM and have been tested on Azure `Standard_NC6s_v3` machines which have 16 GB GPU RAM, while the training scripts in `models/` (except for `lr.py`) use [PyTorch Data Parallelism](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) to distribute training over a single node with multiple GPUs. LM training scripts were tested on Azure `Standard_ND40rs_v2` machines which have 32 GB GPU RAM per GPU and 8 GPUs in total.  

### LM Training

All of the explainability algorithms work by interpreting a trained text classifier.  Additionally, SOC, the baseline to which we compare in our paper from [Jin et al.](https://arxiv.org/pdf/1911.06194.pdf), uses an LM that has undergone further pretraining on the dataset used for sequence classification.  To fine-tune or continue pretraining an LM, check out the code in `general/models` and the associated `README.md` file for assistance with training.

### Generating Explanations

Explainability scripts can be run from the `explain` directory as follows:

- To compute text block importance with MSP, adjust the MSP parameters in `explain/params.yml`, then run `python explain_with_msp.py`.  
- To compute text block importance with SOC, adjust the SOC parameters in `explain/params.yml`, then run `python explain_with_soc.py`.

Be sure to check out and adjust the input data path, trained classifier path, and LM path (in the case of SOC), as well as the parameters for the explainability algorithms before running each script.

### Evaluation

We used expert human annotaters to validate MSP, and you can take a look at the notebooks we used to generate blind experiment data for these annotators and analyze the results at `blind_experiment`.  For different datasets, you might wish to generate your own annotations using a similar approach.  Alternatively, [Murdoch et al.](https://arxiv.org/abs/1801.05453) and [Jin et al.](https://arxiv.org/pdf/1911.06194.pdf) used a method to automatically assess explanations by comparing importance scores of words or phrases from an explainability algorithm with Logistic Regression.  This approach is limited in that multicollinearity and other factors can impact coefficient estimates, and the method also assumes a linear relationship between bag-of-words or bag-of-phrase representations of text within a document and the document labels.  Still, this method provides a rough estimate of explanation fidelity. Code for this automated evaluation procedure can be found in `evaluate` with parameters in `evaluate/ae_params.yml`.

### Contributing

If you have a suggestion to improve this repository, please read `CONTRIBUTING.md`.  PRs are greatly appreciated!  After following instructions in `CONTRIBUTING.md`, add a feature by:

1. Forking the project
2. Creating your feature branch (`git checkout -b feature/AmazingFeature`)
3. Committing your changes (`git commit -m 'Add some AmazingFeature'`)
4. Pushing to the branch (`git push origin feature/AmazingFeature`)
5. Opening a pull request with a description of your changes

### License

Distributed under the Apache 2.0 license. See `LICENSE.txt` for more information.

### Maintainers

- Joel Stremmel
  - GitHub Username: [jstremme](https://github.com/jstremme)
  - Email: joel_stremmel@optum.com
- Brian Hill
  - GitHub Username: [brianhill11](https://github.com/brianhill11)
  - Email: brian.l.hill@optum.com

### Citation

If you use this code in your research, please cite our paper: [*Extend and Explain: Interpreting Very Long Language Models*](https://arxiv.org/abs/2209.01174).

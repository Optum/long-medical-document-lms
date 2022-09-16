# Long Medical Document LMs

[![python38](https://img.shields.io/badge/python-3.8-orange.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Explain and train language models that extract information from long medical documents with the Masked Sampling Procedure (MSP)

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

1. Pretrain language models (LMs) in a self-supervised fashion using masked language modeling (MLM) and fine-tune these models for sequence classification.
2. Compute the importance of multi-token text blocks to fine-tuned LM predictions for a given document or set of documents using a variety of methods.

In our research paper XXX we propose a novel masked sampling procedure (MSP) to explain the predictions of any text classifier.  Our method is well-suited to very long, sparse-attention LMs, and has been validated for medical documents using two physician annotators.  

### Environment

Code in subdirectories of this repository is designed to run in different Python 3.8 virtual environments using [Anaconda](https://www.anaconda.com/products/individual).  Separating environments helps to avoid dependency conflicts for the different parts of this codebase.  In general, create a new conda environment and install dependencies as follows:

```
conda create --name=<env-name> python=3.8
source activate <env-name>
pip install -r requirements.txt
```

### LM Training

All of the explainability algorithms use a text classifier and SOC (the baseline to which we compare in our paper from [Jin et al.](https://arxiv.org/pdf/1911.06194.pdf)) uses an LM that has undergone further pretraining on the dataset used for sequence classification.  To fine-tune or continue pretraining an LM, check out the code in `general/models` and associated `README.md` file.

### Generating Explanations

To use MSP or our implementation of SOC, check out the `general/explain` subdirectory, which contains general implementations of both algorithms as well as a script to generate explanations with a random algorithm.  See `general/README.md` for instructions on running these scripts.  

We used the code in `long_big_bird_exp` to generate the results in our paper and provide it in case exact reproducibility is necessary, however, we recommend using and continuing to enhance the implementation of MSP in `general`.

### Evaluation

We used expert human annotaters to validate MSP, and you can take a look at the notebooks we used to generate blind experiment data for these annotators and analyze the results at `long_big_bird_exp/blind_experiment`.  For different datasets, you might wish to generate your own annotations using a similar approach.  Alternatively, [Murdoch et al.](https://arxiv.org/abs/1801.05453) and [Jin et al.](https://arxiv.org/pdf/1911.06194.pdf) use a method to automatically assess explanations by comparing importance scores of words or phrases from an explainability algorithm with Logistic Regression.  This approach is limited in that multi-collinearity and other factors can impact coefficient estimates, and the method also assumes a linear relationship between bag-of-words or bag-of-phrase representations of text within a document and the document labels.  Still, this method provides a rough estimate of explanation fidelity. Code for this automated evaluation procedure can be found in `general/evaluate`.

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

If you use this code in your research, please cite our paper: XXX.

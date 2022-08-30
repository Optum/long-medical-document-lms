# Train LMs

Pipeline to continue pretraining and fine-tune LMs from Hugging Face Transformers

### Contents

- [About](#about)
- [Environment](#environment)
- [Data Prep](#data-prep)
- [Training](#training)

### About

[![python38](https://img.shields.io/badge/python-3.8-orange.svg)]()

This repository contains code to continue pretraining and fine-tune LMs from Hugging Face Transformers using the Transformers Python library.

### Environment

To build the Python 3.8 environment required to run this code, create a Python 3.8 virtual environment with [Anaconda](https://www.anaconda.com/products/individual) and install the dependencies in `requirements.txt`. 

```
conda create --name=long-lms python=3.8
source activate long-lms
pip install -r requirements.txt
```

### Data Prep

Data should be provided as a Hugging Face Dataset.

### Training

##### Pretraining

After modifying `pt_params.yml`, run `python pt.py` to continue pretraining.

##### Fine-Tuning 

After modifying `ft_params.yml`, run `python ft.py` to run fine-tuning.


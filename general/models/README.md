# Train LMs

[![python38](https://img.shields.io/badge/python-3.8-orange.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Pipeline to continue pretraining and fine-tune LMs from Hugging Face Transformers

### About

This subdirectory contains code to continue pretraining and fine-tune LMs from Hugging Face Transformers using the [Transformers Python library](https://github.com/huggingface/transformers).

### Environment

To build the Python 3.8 environment required to run this code, create a Python 3.8 virtual environment with [Anaconda](https://www.anaconda.com/products/individual) and install the dependencies in `requirements.txt`. 

```
conda create --name=long-lms python=3.8
source activate long-lms
pip install -r requirements.txt
```

### Data Prep

Data should be provided as a [Hugging Face Dataset](https://huggingface.co/datasets).  To create a Hugging Face Dataset, check out their documentation [here](https://huggingface.co/docs/datasets/index).

### Training

##### Pretraining

After modifying `pt_params.yml`, run `python pt.py` to continue pretraining.

##### Fine-Tuning 

After modifying `ft_params.yml`, run `python ft.py` to run fine-tuning.


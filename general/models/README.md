# Train LMs

[![python38](https://img.shields.io/badge/python-3.8-orange.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Code to continue pretraining and fine-tune LMs from Hugging Face Transformers

### About

This subdirectory contains code to continue pretraining and fine-tune LMs from Hugging Face Transformers using the [Transformers Python library](https://github.com/huggingface/transformers).  The code assumes that LMs for continued pretraining and fine-tuning exist on your file-system at paths specified in `.yml`, but can be easily modified to load models over HTTP by replacing these paths with the corresponding LM names from Hugging Face.  Click a model from their [models page](https://huggingface.co/models) and then check out their "Use in Transformers" tab, to see how to download a model over HTTP.

### Data Prep

Data should be provided as a [Hugging Face Dataset](https://huggingface.co/datasets).  To create a Hugging Face Dataset, check out their documentation [here](https://huggingface.co/docs/datasets/index).

### Training

All training scripts (except for `lr.py`) use [PyTorch Data Parallelism](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) to distribute training over a single node with multiple GPUs.  LM training scripts were tested on Azure `Standard_ND40rs_v2` machines which have 32 GB GPU RAM per GPU and 8 GPUs in total.  

See the [HuggingFace Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer) for more information on the parameters defiend in the `.yml` files and passed to the Trainer in `pt.py` and `ft.py`.

Code in `utils.py` handles the automatic creation of experiment directories on your file system to track new runs and save outputs such as model checkpoints, trainer state, and metrics to unique run directories.

##### Pretraining

After modifying `pt_params.yml`, run `python pt.py` to continue pretraining. 

##### Fine-Tuning 

After modifying `ft_params.yml`, run `python ft.py` to run fine-tuning.

##### Logisitic Regression 

Simply run `python lr.py` after modifying parameters in all caps at the top of the script to fit a logistic regression model and save the word or phrase features and corresponding coefficients.


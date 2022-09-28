# Build and Analyze Blind Experiment Data

[![python38](https://img.shields.io/badge/python-3.8-orange.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Notebooks to build datasets for blind annotation and analyze the results.

### About

These notebooks were used to analyze the results in our paper.  They contain some generally useful functions for analyzing the informativeness of text blocks surfaced by MSP and other algorithms using human annotation.

### Environment

These notebooks require basic libraries like [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/), and [Scikit-Learn](https://scikit-learn.org/).  They can be installed at the top of each notebook if you don't already have them using

```
!pip install sklearn
!pip install pandas
!pip install matplotlib
```

...which should install all necessary libraries.

### Configuring Parameters

Parameters for these notebooks are in all caps at the top of each notebook following imports.  Adjust the paths to point to your data from the explainability experiments.

### Running the Notebooks

We recommend running the notebooks cell-by-cell to ensure you are getting the expected outputs, as this code has not yet been generalized but is well-commented and should not be too hard to follow.

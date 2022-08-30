# Steps to Create Mimic 50 Dataset

### Download MIMIC Data and Run CAML Repository Pre-Processing

1. Download the following required files from the [MIMIC III]( https://physionet.org/content/mimiciii/1.4) dataset with `wget -r -N -c -np --user $1 --ask-password https://physionet.org/files/mimiciii/1.4/<file-name>`:
    - D_ICD_PROCEDURES_ICD.csv.gz
    - D_ICD_DIAGNOSES_ICD.csv.gz
    - NOTEEVENTS.csv.gz
    - PROCEDURES_ICD.csv.gz
    - DIAGNOSES_ICD.csv.gz

You will be prompted for a username and password. You must formally request access via a process documented on the [MIMIC website](https://mimic.mit.edu/) for access. There are two key steps that must be completed before access is granted:

    - Complete a recognized course in protecting human research participants that includes Health Insurance Portability and Accountability Act (HIPAA) requirements.
    - Sign a data use agreement, which outlines appropriate data usage and security standards, and forbids efforts to identify individual patients.

2. Download the [caml-mimic repo](https://github.com/jamesmullenbach/caml-mimic) follow the the steps bwloe:

  1. Place the following MIMIC III tables below in the caml repository directory `/mimicdata/mimic3`:
    - D_ICD_PROCEDURES_ICD.csv
    - D_ICD_DIAGNOSES_ICD.csv
    - NOTEEVENTS.csv
    - PROCEDURES_ICD.csv
    - DIAGNOSES_ICD.csv

  2. Follow instructions in the caml repository's `./constants.py` to update `_DIR` variables according to their location in your copy of the repository.

  3. Place the following MIMIC III tables below in the directory `./mimicdata`/:
    - D_ICD_PROCEDURES_ICD.csv
    - D_ICD_DIAGNOSES_ICD.csv

  4. Run `./notebooks/dataproc_mimic_III.ipynb`

3. Once step 2 is complete, you should have the files listed below in `./mimicdata/mimic3`:
  - train_50.csv
  - dev_50.csv
  - test_50.csv

### Format MIMIC 50 Dataset for this Repository

4. Navigate to `./source_prep` and update `constant_params.py` file with paths to the following:
  - train_50.csv
  - dev_50.csv
  - test_50.csv
  - Path to BYOL's  `./model_artifacts` - Create one if it does not exist

5. Run `create_label_dictionary.ipynb` to create the following:
  - `labels.json`
  - `/caml_top_50_codes_dict_reversed.json` - Used to map medical codes to label indices

6. Run `create_x_y_sets.ipynb` to create the following:
  - X sets and X ids
  - y sets and y ids
  - mlb.pkl,
  - goat_label2idx.pkl

import os
import pickle
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict

BASE_PATH = (
    "/mnt/azureblobshare/nlp-modernisation/database/BYOL-mimic50_exp9/model_artifacts/"
)
OUT_PATH = "/mnt/azureblobshare/hf_datasets/"

# Load Data
X_train = np.load(BASE_PATH + "X_train.npy", allow_pickle=True)
y_train = np.load(BASE_PATH + "y_train.npy", allow_pickle=True)
X_val = np.load(BASE_PATH + "X_val.npy", allow_pickle=True)
y_val = np.load(BASE_PATH + "y_val.npy", allow_pickle=True)
X_test = np.load(BASE_PATH + "X_test.npy", allow_pickle=True)
y_test = np.load(BASE_PATH + "y_test.npy", allow_pickle=True)

# Build Hugging Face Dataset
dd = DatasetDict(
    {
        "train": Dataset.from_dict({"text": X_train.tolist(), "label": y_train}),
        "val": Dataset.from_dict({"text": X_val.tolist(), "label": y_val}),
        "test": Dataset.from_dict({"text": X_test.tolist(), "label": y_test}),
    }
)

# Load label2idx dict
with open(os.path.join(BASE_PATH, "goat_label2idx.pkl"), "rb") as f:
    goat_label2idx = pickle.load(f)

# Save label2idx dict
with open(os.path.join(OUT_PATH, "mimic_goat_label2idx.pkl"), "wb") as f:
    pickle.dump(goat_label2idx, f)

# Save dataset
dd.save_to_disk(os.path.join(OUT_PATH, "mimic50.hf"))

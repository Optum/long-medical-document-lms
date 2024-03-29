"""
Module: Utils

"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

def configure_device():

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def check_empty_count_gpus():
    """
    Empty GPU cache,
    and count the number of available devices.
    """

    # Empty GPU cache
    torch.cuda.empty_cache()

    # Count available devices
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPU(s)!")


def configure_model_for_inference(model):
    """
    Prepare to use model for inference.
    """

    check_empty_count_gpus()
    device = configure_device()

    if device.find("cuda") > -1:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)

    model = model.eval()

    return model


def torch_model_predict_indiv(model, sample_input_ids, class_strategy):
    """
    Predict with a torch model.
    Apply the appropriate transformation to the logits
    depending on the classification task.
    """

    # Get device
    device = configure_device()

    # Compute probabilities of each label on the new sample
    with torch.no_grad():
        input_id_tensor = torch.tensor(sample_input_ids).to(device)
        output = model(input_id_tensor).logits

        if class_strategy == "multi_class":
            prob = torch.softmax(output, dim=1).data.cpu().numpy()
        else:
            prob = torch.sigmoid(output).data.cpu().numpy()

    return prob

def torch_model_predict(model, test_loader, class_strategy, return_data_loader_targets=False):
    """
    Given a PyTorch model whose forward method
    returns an object with a logits attribute
    and a test data loader consisting of input ID sequences
    generate and return the estimated probabilities for data in a test data loader
    as a numpy array.  The function used to convert logits to
    probabilities is one of torch.sigmoid or torch.nn.functional.softmax
    depending on if the prediction problem is
    multi-label (class_strategy='multi_label')
    or multi-class (class_strategy='multi_class').
    This function optionally,
    returns the array of targets generated by the data loader.
    :param model (pytorch model): a pytorch model
    :param test_loader (pytorch DataLoader): the data loader
    :param return_data_loader_targets(bool, False): if to return targets or not. If True, logits and targets will be returned
    :return (np.array[,np.array]): logits [and targets]
    """

    # Set model to evaluation mode
    model = configure_model_for_inference(model)

    # Get device
    device = configure_device()

    # Iterate through test data loader
    # Generate predictions with the appropriate transformation
    # Based on the class strategy
    # And optionally return targets
    probs, targets = [], []
    for data, target in test_loader:

        data = data.to(device)
        if return_data_loader_targets:
            targets.extend(target.tolist())

        with torch.no_grad():
            output = model(data).logits

        if class_strategy == "multi_class":
            prob = torch.softmax(output, dim=1).data.cpu().numpy()
        else:
            prob = torch.sigmoid(output).data.cpu().numpy()

        probs.extend(prob)

    # Delete data, target, model, and loader
    # We can empty the cuda cache even if GPUs are not available
    del data
    del target
    del model
    del test_loader
    torch.cuda.empty_cache()

    # Return predicted probabilities and potentially the target array
    if return_data_loader_targets:
        return np.array(probs), np.array(targets)
    else:
        return np.array(probs)


def convert_binary_to_multi_label(doc_y):
    """
    Expects doc_y to be a list of labels or a binary label.
    If the latter, reformats the binary label as a multi-label target.
    """

    if isinstance(doc_y, int):
        if doc_y == 0:
            doc_y = np.array([1, 0])
        elif doc_y == 1:
            doc_y = np.array([0, 1])
        else:
            raise ValueError(f"Expected doc_y to be an int equal to 0 or 1.")

        print("Converted binary labels to one hot representation...")

    return doc_y


def convert_1d_binary_labels_to_2d(labels):
    """
    Convert 1D binary labels to a 2D representation.
    """

    # Convert a 1D, binary label array to 2D
    if isinstance(labels[0], int) or isinstance(labels[0], float):

        # Check that we have a binary list
        assert len(np.array(labels).shape), "Expected labels to be 1D."
        assert all(
            x == 0 or x == 1 for x in labels
        ), "Expected only 1s and 0s in labels."

        # Convert to 2D representation
        new_labels = np.zeros(shape=(len(labels), 2))
        for i, target in enumerate(labels):
            if target == 0:
                new_labels[i] = [1, 0]
            elif target == 1:
                new_labels[i] = [0, 1]
            else:
                raise ValueError(f"Unexpected target: {target}.")

        return new_labels

    # Return 2D array
    else:

        if isinstance(labels, np.ndarray):
            return labels
        else:
            return np.array(labels)


def check_average_precision(model, data, batch_size, class_strategy, average, round_to=4):
    """
    Predict on all data and measure AP over many bootstrap iterations.
    Return strings of AP mean +- stdv formatted for the class strategy.
    """

    # Make sure we are looking at either micro or macro average
    assert average in [
        "micro",
        "macro",
        None,
    ], "Expected average to be one of: ['micro', 'macro', None]."

    # Build data loader
    ds = TensorDataset(torch.tensor(data["input_ids"], dtype=torch.int64), torch.tensor(data["label"], dtype=torch.float32))
    dataloader = DataLoader(ds, batch_size=batch_size)

    # Generate predictions in batch
    preds, y_trues = torch_model_predict(
        model=model,
        test_loader=dataloader,
        class_strategy=class_strategy,
        return_data_loader_targets=True
    )

    # Compute multi-class or multi-label metrics
    if class_strategy in ["multi_class", "multi_label"]:

        # Convert a 1D, binary label array to 2D if it's not already
        labels = convert_1d_binary_labels_to_2d(y_trues)

        # Get bootstrapped ap and stdv
        ap, stdv = get_bootstrapped_ap(labels=labels, preds=preds, average=average)

        return f"Average Precision ({average}) +- stdv: {round(ap, round_to)} +- {round(stdv, round_to)}."

    else:

        # Ensure labels are numpy array
        labels = np.array(y_trues)

        # Get bootstrapped ap and stdv
        ap, stdv = get_bootstrapped_ap(labels=labels, preds=preds, average=None)

        return f"Average Precision +- stdv: {round(ap, round_to)} +- {round(stdv, round_to)}."

def get_bootstrapped_ap(labels, preds, average, n_bootstrap=1000):
    """
    Compute AP on samples with replacement and report mean and standard deviation.
    """

    # Run bootstrap iterations
    aps = []
    for i in range(n_bootstrap):

        # Sample N records with replacement where N is the total number of records
        sample_indices = np.random.choice(len(labels), len(labels))
        sample_labels = labels[sample_indices]
        sample_preds = preds[sample_indices]

        ap = average_precision_score(
            sample_labels, sample_preds, average=average
        )
        aps.append(ap)

    # Compute and return means and stdv
    return np.mean(aps), np.std(aps)

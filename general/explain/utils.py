"""
Module: Utils

"""
import torch
import numpy as np
from sklearn.metrics import average_precision_score


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


def configure_model_for_inference(model, device):
    """
    Prepare to use model for inference.
    """

    check_empty_count_gpus()
    model = model.eval()
    model = model.to(device)

    return model


def predict_with_clf_model(model, sample_input_ids, device, class_strategy):
    """
    Predict with a torch model.
    Apply the appropriate transformation to the logits
    depending on the classification task.
    """

    # Compute probabilities of each label on the new sample
    with torch.no_grad():
        input_id_tensor = torch.tensor(sample_input_ids).to(device)
        output = model(input_id_tensor).logits

        if class_strategy == "multi_class":
            prob = torch.softmax(output, dim=1).data.cpu().numpy()
        else:
            prob = torch.sigmoid(output).data.cpu().numpy()

    return prob


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

    return doc_y


def convert_1d_binary_labels_to_2d(labels):
    """
    Convert 1D binary labels to a 2D representation.
    """

    # Convert a 1D, binary label array to 2D
    if isinstance(labels[0], int):

        # Check that we have a binary list
        assert isinstance(labels, list), "Expected labels to be a list."
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


def check_average_precision(model, data, device, class_strategy, average, round_to=4):
    """
    Loop over many documents to generate inferences.
    It would be more efficient to predict on batches,
    but the explainability methods operate at the level of a single document.
    When generating an inference, wrap the sample in [] as if providing a batch.
    """

    # Make sure we are looking at either micro or macro average
    assert average in [
        "micro",
        "macro",
        None,
    ], "Expected average to be one of: ['micro', 'macro', None]."

    # Prepare model for inference
    model = configure_model_for_inference(model, device)

    # Run inferences and collect predictions and labels
    y_preds, y_trues = [], []
    for sample, label in zip(data["input_ids"], data["label"]):
        y_pred = predict_with_clf_model(
            model=model,
            sample_input_ids=[sample],
            device=device,
            class_strategy=class_strategy,
        )[0]
        y_preds.append(y_pred)
        y_trues.append(label)

    # Convert preds to numpy
    preds = np.array(y_preds)

    # Compute multi-class or multi-label metrics
    if class_strategy in ["multi_class", "multi_label"]:

        # Convert a 1D, binary label array to 2D if it's not already
        labels = convert_1d_binary_labels_to_2d(y_trues)

        ap = round(
            average_precision_score(y_true=labels, y_score=preds, average=average),
            round_to,
        )

        print(f"Average Precision ({average}): {ap}.")

    else:

        labels = np.array(y_trues)
        ap = round(
            average_precision_score(
                y_true=labels,
                y_score=preds,
            ),
            round_to,
        )

        print(f"Average Precision: {ap}.")

"""
Module: Utils

"""
import torch
from transformers import (
    BigBirdConfig,
    BigBirdForMaskedLM,
    BigBirdForSequenceClassification,
    BigBirdTokenizerFast,
)


def get_bigbird_config(tokenizer, params, clf=None):
    """
    Model config used in "Extend and Explain."
    """

    config = {
        "vocab_size": len(tokenizer),
        "hidden_size": 768,  # 960
        "num_hidden_layers": 12,  # 16
        "num_attention_heads": 12,  # 16
        "intermediate_size": 3072,  # 3840
        "hidden_act": "gelu_fast",
        "hidden_dropout_prob": 0.1,
        "attention_dropout_prob": 0.1,
        "max_position_embeddings": params["max_seq_len"],
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "use_cache": True,
        "attention_type": "block_sparse",
        "use_bias": True,
        "rescale_embeddings": False,
        "block_size": 64,
        "num_random_blocks": 3,
        "gradient_checkpointing": True,  # set true to save memory
        "bos_token_id": tokenizer.get_vocab()["<s>"],
        "eos_token_id": tokenizer.get_vocab()["</s>"],
        "pad_token_id": tokenizer.get_vocab()["<pad>"],
        "sep_token_id": tokenizer.get_vocab()["[SEP]"],
    }

    if clf:
        config["num_labels"] = params["num_labels"]

    return config


def get_model_for_eval(model_type, params):
    """
    Return model on GPU for non-DDP inference (in evaluation model).
    """

    # Load tokenizer
    tokenizer = BigBirdTokenizerFast(
        tokenizer_file=params["tokenizer_path"] + "bpe_tokenizer.json"
    )

    # Load model
    if model_type == "mlm":
        model = BigBirdForMaskedLM(
            config=BigBirdConfig(
                **get_bigbird_config(tokenizer, clf=False, params=params)
            )
        )
    elif model_type == "clf":
        model = BigBirdForSequenceClassification(
            config=BigBirdConfig(
                **get_bigbird_config(tokenizer, clf=True, params=params)
            )
        )
    else:
        raise ValueError(
            f"Expected model type to be one of ['mlm', 'clf'] but got {model_type}."
        )

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load model checkpoint
    if model_type == "mlm":

        # Load MLM that has undergone continued pretraining using HuggingFace Trainer
        new_state_dict = torch.load(params["mlm_checkpoint_path"], map_location=device)
        model.load_state_dict(new_state_dict, strict=True)

    elif model_type == "clf":

        checkpoint = torch.load(params["ft_model_checkpoint_path"], map_location=device)

        # Update layer names for non-DDP setting
        # Layers are stored as 'model.layer_name' so we remove 'model.'
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            name = k[6:]
            new_state_dict[name] = v

        # Apply model weights from checkpoint to initialized model
        model.load_state_dict(new_state_dict, strict=True)

    else:
        raise ValueError(
            f"Expected model type to be one of ['mlm', 'clf'] but got {model_type}."
        )

    model.eval()

    return model


def predict_on_sample_with_clf_model(sample_text, params):
    """
    Predict on a single text sample using LM with CLF head.
    Applies sigmoid to output.
    """

    # Get the model to use for inference
    model = get_model_for_eval(model_type="clf", params=params)

    # Get device
    device = configure_device()

    # Put model on device
    model.to(device)

    # Compute probabilities of each label on the new sample
    with torch.no_grad():
        input_ids = torch.tensor([sample_text[0 : params["max_seq_len"]]]).to(device)
        output = model(input_ids).logits
        prob = torch.sigmoid(output).data.cpu().numpy()[0]

    return prob


def check_empty_count_gpus():
    """
    Check that GPU is available, empty the cache,
    and count the number of available devices.
    """

    # Check that a GPU is available:
    assert torch.cuda.is_available(), "No GPU found.  Please run on a GPU."

    # Empty GPU cache
    torch.cuda.empty_cache()

    # Count available devices
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPU(s)!")


def configure_device():
    """
    Return device.
    """

    return "cuda:0" if torch.cuda.is_available() else "cpu"

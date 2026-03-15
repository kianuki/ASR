from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

# Maps sequence fields to pe padded to their pre/post-processing and length extraction logic
# Add new fields here to extend batching behavior without modifying collate_fn
# Default pre/post-processing function is an identity lambda, default get_length uses len(x)
SEQUENCE_FIELDS = {
    "spectrogram": {
        "preprocess": lambda x: x.transpose(0, 1),
        "get_length": lambda x: x.shape[1],
        "postprocess": lambda x: x.transpose(2, 1),
    },
    "text_encoded": {
        "get_length": lambda x: len(x),
    },
}


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = defaultdict(list)

    for item in dataset_items:
        for key, value in item.items():
            if key in SEQUENCE_FIELDS.keys():
                result_batch[f"{key}_length"].append(
                    SEQUENCE_FIELDS[key].get("get_length", lambda x: len(x))(value)
                )
                result_batch[key].append(
                    SEQUENCE_FIELDS[key].get("preprocess", lambda x: x)(value)
                )

                continue

            result_batch[key].append(value)

    for key in SEQUENCE_FIELDS:
        result_batch[key] = SEQUENCE_FIELDS[key].get("postprocess", lambda x: x)(
            pad_sequence(result_batch[key], batch_first=True)
        )
        result_batch[f"{key}_length"] = torch.tensor(
            result_batch[f"{key}_length"], dtype=torch.long
        )

    return result_batch

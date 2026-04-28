import json
import os

from src.utils.io_utils import ROOT_PATH


def prepare_transcriptions(dataset="librispeech", partition="train-clean-100"):
    """
    Prepare transcriptions needed for training a tokenizer
    Args:
        path (str): name of the dataset, which transcriptions are going to be used.
        partition (str): name of the partition of the dataset.
    Returns:
        path (str): path where transcriptions are located.
    """
    index_path = (
        ROOT_PATH / "data" / "datasets" / dataset.lower() / f"{partition}_index.json"
    )

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index file not found for dataset {dataset} and partition {partition}."
            f"Make sure you downloaded the data needed and it's located in {index_path}"
        )

    with open(index_path) as f:
        index = json.load(f)

    texts = [item["text"] for item in index]
    transcriptions_path = ROOT_PATH / "src" / "text_encoder" / "transcriptions.txt"

    with open(transcriptions_path, "w") as f:
        f.write("\n".join(texts))

    return transcriptions_path

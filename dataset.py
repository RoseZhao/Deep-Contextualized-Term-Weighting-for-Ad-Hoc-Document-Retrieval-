from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from typing import List, Optional

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]                       # input_mask in HDCT
    target_weights: List[float]
    target_mask: List[int]
    token_type_ids: Optional[List[int]] = None      # segment_ids in HDCT


# TODO
class HDCTDataset(Dataset):

    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i) -> InputFeatures:
        """
        :return: the input features for the ith example
        """
        raise NotImplementedError


# TODO
def convert_examples_to_features() -> list[InputFeatures]:
    """
    See https://github.com/huggingface/transformers/blob/7abc1d96d114873d9c3c2f1bc81343fb1407cec4/examples/token-classification/utils_ner.py#L78
    :return: list[InputFeatures] including input_ids, input_mask, segment_ids, target_weights, target_mask, as required by HDCT
    """
    raise NotImplementedError

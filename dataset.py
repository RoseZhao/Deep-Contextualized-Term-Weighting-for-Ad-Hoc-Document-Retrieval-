import logging
import os
# !pip install tokenization
# import tokenization
import torch
from filelock import FileLock
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
# !pip install transformers
from transformers import BertTokenizerFast
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import json
from enum import Enum
import pdb

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    term_recall_dict: dict


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]  # input_mask in HDCT
    target_weights: List[float]
    target_mask: List[int]
    token_type_ids: Optional[List[int]] = None  # segment_ids in HDCT


def read_examples_from_file(data_dir) -> List[InputExample]:
    examples = []
    train_files = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f)) and not f.startswith("cached")]
    logging.info(f"train_files = {train_files}")
    for file_name in train_files:
        train_file = open(file_name)
        for i, line in enumerate(train_file):
            json_dict = json.loads(line)
            docid = json_dict["doc"]["id"]
            # doc_text = tokenization.convert_to_unicode(json_dict["doc"]["title"])
            doc_text = json_dict["doc"]["title"]
            term_recall_dict = json_dict["term_recall"]

            guid = "train-%s" % docid
            examples.append(
                InputExample(guid=guid, words=doc_text, term_recall_dict=term_recall_dict)
            )
        train_file.close()
    # random.shuffle(examples)
    return examples

    # tsv_file = open(data_dir)
    # read_tsv = csv.reader(tsv_file, delimiter="\t")
    # examples = [InputExample(guid=row[0],words=row[1]) for row in list(read_tsv)]
    # return examples


def is_test_file(data_dir, f):
    return isfile(join(data_dir, f)) and not f.startswith("cached") and (f.endswith(".tsv.1") or f.endswith(".tsv.2"))


def get_test_examples(data_dir):
    test_files = [join(data_dir, f) for f in listdir(data_dir) if is_test_file(data_dir, f)]
    examples = []
    for file_name in test_files:
        test_file = open(file_name, encoding="utf-8")
        for i, line in enumerate(test_file):
            docid, t = line.strip().split('\t')
            # doc_text = tokenization.convert_to_unicode(t)
            term_recall_dict = {}

            guid = "test-%s" % docid
            examples.append(
                InputExample(guid=guid, words=t, term_recall_dict=term_recall_dict)
            )
        test_file.close()
    return examples


def convert_examples_to_features(examples: List[InputExample],
                                 max_seq_length: int,
                                 tokenizer: BertTokenizerFast,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 # sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True
                                 ) -> List[InputFeatures]:
    """Loads a data file into a list of `InputFeatures`
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = tokenizer.tokenize(example.words)
        if len(tokens) == 0:
            tokens = ["."]

        # for word in example.words:
        #     word_tokens = tokenizer.tokenize(word)

        #     # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
        #     if len(word_tokens) > 0:
        #         tokens.extend(word_tokens)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]

        target_weights, target_mask = gen_target_token_weights(tokens, example.term_recall_dict)
        assert len(target_mask) == len(tokens)
        assert len(target_weights) == len(tokens)

        # SEP
        tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        target_weights += [0.0]
        target_mask += [0]
        # CLS
        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
            target_weights += [0.0]
            target_mask += [0]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            target_weights = [0.0] + target_weights
            target_mask = [0] + target_mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            target_weights = ([0.0] * padding_length) + target_weights
            target_mask = ([0] * padding_length) + target_mask
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            target_weights += [0.0] * padding_length
            target_mask += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(target_weights) == max_seq_length
        assert len(target_mask) == max_seq_length

        if ex_index < 5:
            logging.info(f"example = {example}")
            logging.info(f"input_ids = {input_ids}")
            logging.info(f"input_mask = {input_mask}")
            logging.info(f"token_type_ids = {segment_ids}")
            logging.info(f"target_weights = {target_weights}")
            logging.info(f"target_mask = {target_mask}")

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                target_weights=target_weights, target_mask=target_mask
            )
        )
    return features


def gen_target_token_weights(tokens, term_recall_dict):
    fulltoken = tokens[0]
    i = 1
    s = 0
    term_recall_weights = [0.0 for _ in range(len(tokens))]
    term_recall_mask = [0 for _ in range(len(tokens))]
    while i < len(tokens):
        if tokens[i].startswith('##'):
            fulltoken += tokens[i][2:]
            i += 1
            continue

        w = term_recall_dict.get(fulltoken, 0.0)
        term_recall_weights[s] = w
        #if fulltoken in stopwords:
        #    term_recall_mask[s] = 0
        #else:
        term_recall_mask[s] = 1

        # if fulltoken in term_recall_dict:
        #     w = term_recall_dict.get(fulltoken)
        #     term_recall_weights[s] = float(w)
        #     term_recall_mask[s] = 1

        fulltoken = tokens[i]
        s = i
        i += 1

    if fulltoken in term_recall_dict:
        w = term_recall_dict.get(fulltoken)
        term_recall_weights[s] = float(w)
        term_recall_mask[s] = 1
    # w = term_recall_dict.get(fulltoken, 0)
    # term_recall_weights[s] = w
    # term_recall_mask[s] = 1
    return term_recall_weights, term_recall_mask

class Split(Enum):
    train = "train"
    # dev = "dev"
    test = "test"

class HDCTDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(
            self,
            data_dir: str,
            tokenizer: BertTokenizerFast,
            model_type: str,
            mode: Split = Split.train,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
    ):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length))
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                if mode == Split.train:
                  examples = read_examples_from_file(data_dir)
                elif mode == Split.test:
                  examples = get_test_examples(data_dir)
                for i in range(5):
                    logging.info(f"example {i} = {examples[i]}")
                self.features = convert_examples_to_features(
                    examples,
                    max_seq_length,
                    tokenizer,
                    # xlnet has a cls token at the end
                    cls_token_at_end=bool(model_type in ["xlnet"]),
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                    sep_token=tokenizer.sep_token,
                    # sep_token_extra=False,
                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=bool(tokenizer.padding_side == "left"),
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        """
        :return: the input features for the ith example
        """
        return self.features[i]

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# dataset = HDCTDataset(
#         "./",
#         tokenizer,
#         "bert",
#         20,
#         Split.test
# )
from dataset import *
from transformers import AutoTokenizer
import logging

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logging.info("called")
    data_dir = "/bos/tmp10/hongqiay/hdct/mini_train"
    examples = read_examples_from_file(data_dir)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",  use_fast=True)
    convert_examples_to_features(examples, max_seq_length=128, tokenizer=tokenizer)


if __name__=="__main__":
    main()

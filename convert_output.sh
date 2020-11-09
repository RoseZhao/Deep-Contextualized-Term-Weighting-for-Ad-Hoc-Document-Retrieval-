#!/bin/bash
OUTPUT_DIR=./deepct_output/test${1}
DATA_DIR=/bos/tmp10/hongqiay/hdct/test${1}
python bert_term_sample_to_json.py \
  --smoothing="none" \
  --output_format="json" \
  /bos/tmp10/hongqiay/hdct/test4/collection_mini.tsv.1  \
  ./deepct_output/test4/test_results.tsv   \
  ./deepct_output/test4/weights.json  \
  100

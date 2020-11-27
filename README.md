# 11785-team-project

# Requirements 
The following libraries are required for this project: 
```
pytorch 1.7
transformers 3.4.0
```

Please install transformers 3.4.0 from source in order to run the code properly: 
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install transformers
```

# Dataset
Assuming the current working directory is `11785-team-project`, do the following steps to get the dataset 
```
mkdir data
wget http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/data/myalltrain.relevant.docterm_recall
wget http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/data/collection.tsv.1.zip
wget http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/data/collection.tsv.2.zip
mkdir test0 && cd test0 && unzip collection.tsv.1.zip 
mkdir test1 && cd test1 && unzip collection.tsv.2.zip
```
Download 

# Training
Please run the following bash script to train the model
```
#!/bin/sh
OUTPUT_DIR=./deepct_output
DATA_DIR=./data

python ./run_hdct.py   \
    --model_name_or_path bert-base-cased   \
    --max_seq_length 128 \
    --do_train   \
    --data_dir $DATA_DIR  \
    --per_device_eval_batch_size=32   \
    --per_device_train_batch_size=32   \
    --learning_rate 2e-5   \
    --num_train_epochs 3.0  \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir   \
    --save_steps 10000    \
    --logging_steps 100  \
    --warmup_steps 10000
```
The final model we use is the epoch2 (0 indexed) checkpoint saved under `./deepct_output`. 

# Inference
Please run the following bash script to run the inference: 
```
for i in 0 1 
do 
DATA_DIR=./data/test${i}
OUTPUT_DIR=./deepct_output/test${i}
python ./run_hdct.py   \
    --model_name_or_path=./deepct_output/epoch2-checkpoint-32910   \
    --max_seq_length 128 \
    --do_predict   \
    --data_dir $DATA_DIR  \
    --per_device_eval_batch_size=32   \
    --per_device_train_batch_size=32   \
    --learning_rate 2e-5   \
    --num_train_epochs 1.0  \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir   \
    --save_steps 10000    \
    --logging_steps 100  \
    --warmup_steps 10000
python bert_term_sample_to_json.py \
  --smoothing="none" \
  --output_format="json" \
  ./data/test${i}/collection.tsv.$((i+1))  \
  ./deepct_output/test${i}/test_results.tsv   \
  ./deepct_output/test${i}/weights${i}.json  \
  100
done 
```
`./deepct_output/test${i}/weights${i}.json` store the term weights output by the model. 

# Indexing and Retrieval 
The json files above can be input to Anserini to index and retrieve the documents. See https://github.com/castorini/anserini for instructions. Note that we use BM25 with k1=10, b=0.9

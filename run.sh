#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=50000 # Memory - Use up to 50G
#SBATCH --time=0 # No time limit
#SBATCH --mail-user=hongqiay@andrew.cmu.edu
#SBATCH --mail-type=END
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source activate hdct

OUTPUT_DIR=./deepct_output
DATA_DIR=/bos/tmp10/hongqiay/hdct

# train & eval
python ./run_hdct.py   \
    --model_name_or_path bert-base-cased   \
    --max_seq_length 128 \
    --do_train   \
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

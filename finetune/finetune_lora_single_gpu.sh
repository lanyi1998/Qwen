#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`


MODEL="Qwen/Qwen-7B-Chat-Int4" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="./data"

export CUDA_VISIBLE_DEVICES=0

python finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --fp16 True \
  --output_dir output_qwen \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora

# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json

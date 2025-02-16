#!/bin/bash

chmod +x ./s5cmd
./s5cmd sync s3://$MODEL_S3_BUCKET/Llama-2-7B-fp16/* /tmp/llama-2-7B-fp16/
#./s5cmd sync s3://$MODEL_S3_BUCKET/llama-7b-hf/* /tmp/llama-7b-hf/


cd DeepSpeed-Chat/training/step1_supervised_finetuning/

#Fix huggingface issue
#pip install --upgrade huggingface_hub
#huggingface-cli login --token hf_xxxx


#bash training_scripts/opt/single_gpu/run_1.3b.sh
#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_llama2_7b_sft_demo
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

#if you are using standard dataset on HF, use main.py
#if you are using custom dataset, use main_sft.py

deepspeed main_sft.py \
   --data_path local/jsonfile \
   --data_split 10,0,0 \
   --model_name_or_path /tmp/llama-2-7B-fp16/ \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 4096 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --print_loss \
   --offload

#   --model_name_or_path TheBloke/Llama-2-13B-fp16 \
#   --model_name_or_path /tmp/Llama-2-13B-fp16/ \
#&> $OUTPUT/training.log

#set the timer for stopping the training job
#sleep 900
#kill "$!"

if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi

#Upload the trained models to S3
./s5cmd sync $OUTPUT s3://$MODEL_S3_BUCKET/deepspeedchat-finetuned-llama-2-7B-fp16-20230928

#s3://$MODEL_S3_BUCKET/RWKV/output/$(date +%Y-%m-%d-%H-%M-%S)/

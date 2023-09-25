#!/bin/bash

chmod +x ./s5cmd
./s5cmd sync s3://$MODEL_S3_BUCKET/Llama-2-13B-fp16/* /tmp/llama-2-13B-fp16/
#./s5cmd sync s3://$MODEL_S3_BUCKET/llama-7b-hf/* /tmp/llama-7b-hf/


cd DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/

#Fix huggingface issue
pip install --upgrade huggingface_hub
huggingface-cli login --token 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


#bash training_scripts/opt/single_gpu/run_1.3b.sh
#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_llama2_7b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path Dahoas/synthetic-instruct-gptj-pairwise \
   --data_split 10,0,0 \
   --model_name_or_path /tmp/llama-2-13B-fp16/ \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 512 \
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
   --lora_dim 128 \
   --lora_module_name "layers." \
   
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
#./s5cmd sync ./llama-2-13B-fp16_lora s3://$MODEL_S3_BUCKET/llama-2-13B-fp16_lora

#s3://$MODEL_S3_BUCKET/RWKV/output/$(date +%Y-%m-%d-%H-%M-%S)/

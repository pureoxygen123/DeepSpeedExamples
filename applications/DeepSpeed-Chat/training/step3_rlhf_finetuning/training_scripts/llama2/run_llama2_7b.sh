#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
ACTOR_MODEL_PATH="OpenLLMAI/Llama-2-7b-sft-model-ocra-500k"
CRITIC_MODEL_PATH="OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt/models--OpenLLMAI--Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt"
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step3_llama
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --num_nodes=4 \
   -H hostfile1 \
    main.py \
   --data_path Dahoas/full-hh-rlhf \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 4 \
   --gradient_accumulation_steps 4 \
   --ppo_epochs 1 \
   --max_answer_seq_len 1024 \
   --max_prompt_seq_len 1024 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --actor_dropout 0.0 \
   --num_warmup_steps 0 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_hybrid_engine \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log

#    --offload \
#    --offload_reference_model \

#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
export PATH="/fs-computility/llm/shared/rl3m_env/dep/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/fs-computility/llm/shared/rl3m_env/dep/cuda-11.7/lib64":$LD_LIBR
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export C_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu:$C_INCLUDE_PATH
export TORCH_CUDNN_V8_API_DISABLED=1
# DeepSpeed Team
ACTOR_MODEL_PATH="/fs-computility/llm/chenyang2/OpenLLMAI/Llama-2-7b-sft-model-ocra-500k"
CRITIC_MODEL_PATH="/fs-computility/llm/chenyang2/hf_model/OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt/models--OpenLLMAI--Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt/snapshots/a982afeed00fac9767d53aecde5b88947b1be194"
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
   -H /fs-computility/llm/chenyang2/DeepSpeedExamples-master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/llama2/hostfile \
    main.py \
   --data_path Dahoas/full-hh-rlhf \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --gradient_accumulation_steps 1 \
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
    &> $OUTPUT/training-test-fordata.log

#    --offload \
#    --offload_reference_model \
#    --dtype bf16 \
#    --unpin_actor_parameters \
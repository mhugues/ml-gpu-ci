# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.    


export TORCHRUN=torchrun
export TRAIN_SCRIPT=./train.py

export CUDA_LAUNCH_BLOCKING=1
declare -a TORCHRUN_ARGS=(
    --nproc_per_node=1
    --nnodes=1
    --rdzv_id=6000
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)


##############################
# GPT2 117M Training Params ##
##############################

declare -a TRAINING_ARGS=(
    --max_context_width=2048
    --hidden_width=768
    --num_layers=12
    --num_heads=12
    --model_type=gpt2
    --vocab_size=50257
    --tokenizer="openai-community/gpt2"
    --checkpoint_freq=5000
    --validation_freq=500
    --max_steps=10
    --checkpoint_dir=./checkpoints
    --dataset='allenai/c4'
    --dataset_config_name='en'
    --resume_from_checkpoint=./checkpoints
    --train_batch_size=1
    --val_batch_size=1
    --sharding_strategy="full"
    --offload_activations=1
    --bf16=1
)


${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"

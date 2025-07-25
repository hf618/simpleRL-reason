#! /bin/bash

USER_ENV=`whoami`
set -x
# export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1


export PROJECT_NAME=verl_train_gpugeek
export WANDB_API_KEY=8c84ddd422687515e5df25109f349a4f2c5df884
export WANDB_OFFICIAL=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN # FLASHINFER # XFORMERS
export HDFS_DATA_PATH=/gz-data/custom/data
export HDFS_MODEL_PATH=/gz-data/Models/qwen
export HDFS_CHECKPOINT_PATH=/gz-data/custom/checkpoint
export HDFS_LOG_PATH=/gz-data/custom/log
export RUN_NAME=verl-grpo
export ARNOLD_WORKER_NUM=1 # number of nodes you want to use 

export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export CUDA_LAUNCH_BLOCKING=1  # 同步CUDA错误
export NCCL_DEBUG=INFO          # 启用NCCL详细日志
export NCCL_SOCKET_IFNAME=eth0 # 指定网卡（根据ifconfig替换）
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$HOME/.local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# export LD_PRELOAD=$HOME/.local/lib/libcuda.so
export RAY_pickling_fallback="True"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export RAY_DEBUG=legacy
export REWORD_FUNCTION_TYPE="independent"
# n这三个平凡更换
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # 只使用 GPU 0 和 1
NUM_GPUS=4
HEAD_PORT="6379" # 注意更换 6379

WORKING_DIR="."
HEAD_IP="172.17.0.2" # 注意更换为你的 head ip


# Default values
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=32
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=16
# per GPU
PPO_MICRO_BATCH_SIZE=2
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=4
ROLLOUT_N=8
KL_COEF=0.001
TOTAL_EPOCHS=20
DATASET_NAME=simplelr_qwen_level3to5
ROLLOUT_GPU_MEMORY_UTIL=0.6
MODEL_NAME=Qwen2.5-Math-7B
SAVE_FREQ=20
TEST_FREQ=5
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
MICRO_ROLLOUT_BATCH_SIZE=1024
REMOVE_PREVIOUS_CKPT=False

# Default values for new parameters
REWARD_EMA_ALPHA=""
REWARD_INDICATOR_NAMES=""
REWARD_WEIGHTS=""
REWARD_WEIGHTS_INNER=""

# <<< 修改点 1: 将 HYDRA_OVERRIDES 初始化为数组 >>>
HYDRA_OVERRIDES=()
VAL_BEFORE_TRAIN=True
VAL_SAMPLE_SIZE=-1
ENABLE_CALCULATOR=True
DIFF_STRIDE=20
ADD_REWARD=True
COMPUTE_LOG_EFFECTIVE_RANK=False

generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --val_batch_size) suffix+="_valbatch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --ppo_micro_batch_size) shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --log_prob_micro_batch_size) suffix+="_logprobbatch$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --total_epochs) suffix+="_epochs$2"; shift 2 ;;
      --rollout_gpu_memory_util) shift 2 ;;
      # --- 重点检查这一行 ---
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      # ---------------------
      --remove_clip) suffix+="_remove_clip$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      --logger_config) LOGGER_CONFIG="$2"; shift 2 ;;
      --exp_name) EXP_NAME="$2"; shift 2 ;;
      --diff_stride) suffix+="_stride$2"; shift 2 ;;
      *) shift ;;
    esac
  done

  # 如果命令行中没有提供 --dataset_name，则使用默认值
  # 因为上面设置了标志位，所以这里不会重复添加
  # if [ "$dataset_provided" = false ]; then
  #   suffix+="_$DATASET_NAME"
  # fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi
  
  echo "$suffix"
}

echo "Arguments received: $@"

# Generate a unique suffix based on the input arguments
SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"
EXP_NAME=${exp_name}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val_batch_size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --micro_rollout_batch_size) MICRO_ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --remove_clip) REMOVE_CLIP="$2"; shift 2 ;;
    --remove_previous_ckpt) REMOVE_PREVIOUS_CKPT="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    --logger_config) LOGGER_CONFIG="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    # <<< 新增参数解析 >>>
    --reward_ema_alpha) REWARD_EMA_ALPHA="$2"; shift 2 ;;
    --reward_indicator_names) REWARD_INDICATOR_NAMES="$2"; shift 2 ;;
    --reward_weights) REWARD_WEIGHTS="$2"; shift 2 ;;
    --reward_weights_inner) REWARD_WEIGHTS_INNER="$2"; shift 2 ;;
    --val_before_train) VAL_BEFORE_TRAIN="$2"; shift 2 ;;
    --val_sample_size) VAL_SAMPLE_SIZE="$2"; shift 2 ;;
    --diff_stride) DIFF_STRIDE="$2"; shift 2 ;;
    --enable_calculator) ENABLE_CALCULATOR="$2"; shift 2 ;;
    --add_reward) ADD_REWARD="$2"; shift 2 ;;
    --compute_log_effective_rank) COMPUTE_LOG_EFFECTIVE_RANK="$2"; shift 2 ;;
    # <<< 新增结束 >>>
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done


# ... (End of argument parsing while loop)

# Generate a unique suffix based on the input arguments (now without model name)
SUFFIX=$(generate_suffix "$@")

# Construct the FINAL_RUN_NAME in the desired order: {model}_{exp}_{base}{suffix}
# For example: Qwen2.5-3B_origin_verl-grpo_max_response1280...
FINAL_RUN_NAME="${MODEL_NAME}_${EXP_NAME}_${RUN_NAME}${SUFFIX}"

# Update the log file path to use the new name
LOG_FILE_PATH="$HDFS_LOG_PATH/$FINAL_RUN_NAME.log"
# The EXP_NAME variable is now part of the FINAL_RUN_NAME

# ... (echo statements)


echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Val Batch Size: $VAL_BATCH_SIZE" 
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" 
echo "Max Response Length: $MAX_RESPONSE_LENGTH" 
echo "Learning Rate: $LEARNING_RATE" 
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" 
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE" 
echo "Micro Rollout Batch Size: $MICRO_ROLLOUT_BATCH_SIZE"
echo "KL Loss Coefficient: $KL_LOSS_COEF" 
echo "KL Loss Type: $KL_LOSS_TYPE" 
echo "Temperature: $TEMPERATURE" 
echo "Rollout N: $ROLLOUT_N"
echo "Remove Previous Ckpt: $REMOVE_PREVIOUS_CKPT"
echo "LOG FILE PATH: $LOG_FILE_PATH"

max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 100)

echo -e "Training with the following parameters:\nTrain Batch Size: $TRAIN_BATCH_SIZE\nVal Batch Size: $VAL_BATCH_SIZE\nMax Prompt Length: $MAX_PROMPT_LENGTH\nMax Response Length: $MAX_RESPONSE_LENGTH\nLearning Rate: $LEARNING_RATE\nPPO Mini Batch Size: $PPO_MINI_BATCH_SIZE\nPPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE\nKL Loss Coefficient: $KL_LOSS_COEF\nKL Loss Type: $KL_LOSS_TYPE\nTemperature: $TEMPERATURE\nRollout N: $ROLLOUT_N\nKL Coefficient: $KL_COEF\nTotal Epochs: $TOTAL_EPOCHS\nDataset Name: $DATASET_NAME\nModel Name: $MODEL_NAME"
echo "Validate Before Train: $VAL_BEFORE_TRAIN"
echo "Validation Sample Size: $VAL_SAMPLE_SIZE"
echo "Calculator Diff Stride: $DIFF_STRIDE"
echo "Enable Calculator Metrics: $ENABLE_CALCULATOR"
echo "Add Reward enabled: $ADD_REWARD"
echo "Compute Log Effective Rank: $COMPUTE_LOG_EFFECTIVE_RANK"
echo "LOG FILE PATH: $LOG_FILE_PATH"
# <<< 修改点 2: 构建 hydra 参数数组 >>>
# 将覆盖参数作为独立元素添加到数组中
if [ -n "$REWARD_EMA_ALPHA" ]; then
  HYDRA_OVERRIDES+=("reward_manager.ema_alpha=$REWARD_EMA_ALPHA")
fi
if [ -n "$REWARD_INDICATOR_NAMES" ]; then
  # 这里直接使用变量，不添加额外的引号
  HYDRA_OVERRIDES+=("reward_manager.indicator_names=$REWARD_INDICATOR_NAMES")
fi
if [ -n "$REWARD_WEIGHTS" ]; then
  HYDRA_OVERRIDES+=("reward_manager.weights=$REWARD_WEIGHTS")
fi
if [ -n "$REWARD_WEIGHTS_INNER" ]; then
  HYDRA_OVERRIDES+=("reward_manager.weights_inner=$REWARD_WEIGHTS_INNER")
fi
if [ -n "$ADD_REWARD" ]; then
  HYDRA_OVERRIDES+=("reward_manager.add_reward=$ADD_REWARD")
fi
if [ -n "$COMPUTE_LOG_EFFECTIVE_RANK" ]; then
  HYDRA_OVERRIDES+=("calculator.compute_log_effective_rank=$COMPUTE_LOG_EFFECTIVE_RANK")
fi


ray job submit --address=${HEAD_IP}:${HEAD_PORT} \
  --entrypoint-num-cpus=1 \
  --runtime-env-json='{
        "working_dir": "'${WORKING_DIR}'",
        "excludes": [
          "/.git/",                    
          "/checkpoint/",
          "/custom/checkpoint/",
          "/custom/log/",
          "/custom/data/",
          "/root/simpleRL-reason/examples/simplelr_math_eval/data/tabmwp/test.jsonl"
        ],
        "env_vars": {
          "http_proxy": "",
          "https_proxy": "",
          "WANDB_API_KEY": "8c84ddd422687515e5df25109f349a4f2c5df884",
          "CUDA_LAUNCH_BLOCKING": "1",
          "NCCL_DEBUG": "INFO",
          "NCCL_SOCKET_IFNAME": "eth0",
          "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
          "REWORD_FUNCTION_TYPE": "independent",
          "RAY_DEBUG": "legacy"
        }
    }' \
  -- python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
  data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
  actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
  actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.swap_space=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
  actor_rollout_ref.rollout.micro_rollout_batch_size=$MICRO_ROLLOUT_BATCH_SIZE \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True\
  actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  algorithm.kl_ctrl.kl_coef=$KL_COEF \
  critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
  trainer.critic_warmup=0 \
  trainer.logger=$LOGGER_CONFIG \
  trainer.project_name=$PROJECT_NAME \
  trainer.remove_previous_ckpt=$REMOVE_PREVIOUS_CKPT \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=$NUM_GPUS \
  trainer.nnodes=$ARNOLD_WORKER_NUM \
  trainer.remove_clip=$REMOVE_CLIP \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$FINAL_RUN_NAME \
  "${HYDRA_OVERRIDES[@]}" \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
  trainer.val_sample_size=$VAL_SAMPLE_SIZE \
  calculator.diff_stride=$DIFF_STRIDE \
  calculator.enable=$ENABLE_CALCULATOR \
  trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH 

  

#! /bin/bash

USER_ENV=`whoami`
set -x
FULL_ARGS="$@"

# The machine-specific variables are now inherited from the calling script's environment.
# Default values are provided for key variables to ensure the script can run standalone for debugging.
PROJECT_NAME=${PROJECT_NAME:-"verl_train_gong"}
RUN_NAME=${RUN_NAME:-"verl-grpo"}
HDFS_DATA_PATH=${HDFS_DATA_PATH:-"/path/to/your/data"}
HDFS_MODEL_PATH=${HDFS_MODEL_PATH:-"/path/to/your/models"}
HDFS_CHECKPOINT_PATH=${HDFS_CHECKPOINT_PATH:-"./checkpoint"}
HDFS_LOG_PATH=${HDFS_LOG_PATH:-"./log"}
ARNOLD_WORKER_NUM=${ARNOLD_WORKER_NUM:-1}
NUM_GPUS=${NUM_GPUS:-2}
HEAD_IP=${HEAD_IP:-"127.0.0.1"}
HEAD_PORT=${HEAD_PORT:-"6379"}
WORKING_DIR=${WORKING_DIR:-"."}

# The runtime environment JSON is now also passed from the parent script
# If it's not present, we create a minimal default for standalone execution
DEFAULT_RUNTIME_ENV_JSON='{"working_dir": "'${WORKING_DIR}'", "env_vars": { "WANDB_API_KEY": "'${WANDB_API_KEY}'" }}'
RAY_RUNTIME_ENV_JSON=${RAY_RUNTIME_ENV_JSON:-$DEFAULT_RUNTIME_ENV_JSON}

DATASET_NAME=simplelr_qwen_level1to4

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
REWARD_WEIGHTS_EXPLOIT=""
METRIC_INDICES="[0,1,2]" # 默认指标索引
# <<< 修改点 1: 将 HYDRA_OVERRIDES 初始化为数组 >>>
HYDRA_OVERRIDES=()
VAL_BEFORE_TRAIN=True
VAL_SAMPLE_SIZE=-1
ENABLE_CALCULATOR=True
DIFF_STRIDE=20
ADD_REWARD=True
COMPUTE_LOG_EFFECTIVE_RANK=False
MODULATION_GAIN=1.5
OUTPUT_TOKEN_LEVEL_METRICS=False
ADV_ESTIMATOR="grpo"
CRITIC_MODEL_PATH=""
AUX_REWARD_GLOBAL_WEIGHT=1.0 
TOKEN_LEVEL_BASELINE_TYPE="internal_mean" 

# hidden states
RETURN_HIDDEN_STATES=False
RETURN_PREFILL=True
RETURN_DECODE=True

ADD_ADV=True
AUX_FIX=False
ADV_SHAPING_KAPPA=2.0
SVD_RANK=50
SVD_NITER=5

ZEROTH_ORDER_SVD_METHOD="full" 
DIFF_SVD_METHOD="lowrank" 
HYPOTHESIS_TYPE="PlanA" 

COMPUTE_GLOBAL_METRICS=False
COMPUTE_CUMULATIVE_GLOBAL_METRICS=False
GLOBAL_DIFF_STRIDE_TRAIN=${GLOBAL_DIFF_STRIDE_TRAIN:-1}
GLOBAL_DIFF_STRIDE_VAL=${GLOBAL_DIFF_STRIDE_VAL:-20}
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
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --remove_clip) suffix+="_remove_clip$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      --logger_config) LOGGER_CONFIG="$2"; shift 2 ;;
      --exp_name) EXP_NAME="$2"; shift 2 ;;
      --diff_stride) suffix+="_stride$2"; shift 2 ;;
      --reward_ema_alpha) suffix+="_ema$2"; shift 2 ;;
      --modulation_gain) suffix+="_mgain$2"; shift 2 ;;
      --aux_reward_global_weight) suffix+="_auxgw$2"; shift 2 ;;
      --adv_estimator) suffix+="_$2"; shift 2 ;;
      --critic_model_path) shift 2 ;;
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
    --reward_ema_alpha) REWARD_EMA_ALPHA="$2"; shift 2 ;;
    --reward_indicator_names) REWARD_INDICATOR_NAMES="$2"; shift 2 ;;
    --reward_weights) REWARD_WEIGHTS="$2"; shift 2 ;;
    --reward_weights_exploit) REWARD_WEIGHTS_EXPLOIT="$2"; shift 2 ;;
    --val_before_train) VAL_BEFORE_TRAIN="$2"; shift 2 ;;
    --val_sample_size) VAL_SAMPLE_SIZE="$2"; shift 2 ;;
    --diff_stride) DIFF_STRIDE="$2"; shift 2 ;;
    --enable_calculator) ENABLE_CALCULATOR="$2"; shift 2 ;;
    --add_reward) ADD_REWARD="$2"; shift 2 ;;
    --add_adv) ADD_ADV="$2"; shift 2 ;;
    --compute_log_effective_rank) COMPUTE_LOG_EFFECTIVE_RANK="$2"; shift 2 ;;
    --metric_indices) METRIC_INDICES="$2"; shift 2 ;;
    --modulation_gain) MODULATION_GAIN="$2"; shift 2 ;;
    --output_token_level_metrics) OUTPUT_TOKEN_LEVEL_METRICS="$2"; shift 2 ;;
    --adv_estimator) ADV_ESTIMATOR="$2"; shift 2 ;;
    --critic_model_path) CRITIC_MODEL_PATH="$2"; shift 2 ;;
    --aux_reward_global_weight) AUX_REWARD_GLOBAL_WEIGHT="$2"; shift 2 ;;
    --token_level_baseline_type) TOKEN_LEVEL_BASELINE_TYPE="$2"; shift 2 ;; 
    --return_hidden_states) RETURN_HIDDEN_STATES="$2"; shift 2 ;;
    --return_prefill) RETURN_PREFILL="$2"; shift 2 ;;
    --return_decode) RETURN_DECODE="$2"; shift 2 ;;
    --aux_fix) AUX_FIX="$2"; shift 2 ;;
    --adv_shaping_kappa) ADV_SHAPING_KAPPA="$2"; shift 2 ;;
    --svd_rank) SVD_RANK="$2"; shift 2 ;;
    --svd_niter) SVD_NITER="$2"; shift 2 ;;
    --zeroth_order_svd_method) ZEROTH_ORDER_SVD_METHOD="$2"; shift 2 ;; 
    --diff_svd_method) DIFF_SVD_METHOD="$2"; shift 2 ;;
    --hypothesis_type) HYPOTHESIS_TYPE="$2"; shift 2 ;;
    --compute_global_metrics) COMPUTE_GLOBAL_METRICS="$2"; shift 2 ;;
    --compute_cumulative_global_metrics) COMPUTE_CUMULATIVE_GLOBAL_METRICS="$2"; shift 2 ;;
    --global_diff_stride_train) GLOBAL_DIFF_STRIDE_TRAIN="$2"; shift 2 ;;
    --global_diff_stride_val) GLOBAL_DIFF_STRIDE_VAL="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "$ADV_ESTIMATOR" == "grpo" ]]; then
  if [[ "$ROLLOUT_N" -le 1 ]]; then
    echo "错误：当 adv_estimator 为 grpo 时, --rollout_n 必须大于 1."
    exit 1
  fi
  if [[ -n "$CRITIC_MODEL_PATH" ]]; then
    echo "警告：当 adv_estimator 为 grpo 时, --critic_model_path 将被忽略."
    CRITIC_MODEL_PATH="" # 确保不传递
  fi
elif [[ "$ADV_ESTIMATOR" == "gae" ]]; then
  if [[ "$ROLLOUT_N" -ne 1 ]]; then
    echo "错误：当 adv_estimator 为 gae (PPO模式) 时, --rollout_n 必须等于 1."
    exit 1
  fi
  if [[ -z "$CRITIC_MODEL_PATH" ]]; then
    echo "错误：当 adv_estimator 为 gae (PPO模式) 时, 必须通过 --critic_model_path 提供 Critic 模型路径."
    exit 1
  fi
else
  echo "错误：无效的 adv_estimator: $ADV_ESTIMATOR. 请选择 'grpo' 或 'gae'."
  exit 1
fi


# 检查：如果 RETURN_PREFILL 或 RETURN_DECODE 中任何一个为 True，
# 那么 RETURN_HIDDEN_STATES 也必须为 True。
if { [ "$RETURN_PREFILL" = "True" ] || [ "$RETURN_DECODE" = "True" ]; } && [ "$RETURN_HIDDEN_STATES" != "True" ]; then
  echo "-------------------------------------------------------------------"
  echo "【错误】参数逻辑冲突"
  echo "当 --return_prefill 或 --return_decode 设置为 True 时,"
  echo "意味着需要从 prefill 或 decode 阶段获取 hidden states，"
  echo "因此 --return_hidden_states 参数也必须设置为 True。"
  echo ""
  echo "请检查您的启动参数。"
  echo "-------------------------------------------------------------------"
  exit 1
fi

# Generate a unique suffix based on the input arguments (now without model name)
SUFFIX=$(generate_suffix "$@")

# Construct the FINAL_RUN_NAME in the desired order: {model}_{exp}_{base}{suffix}
# For example: Qwen2.5-3B_origin_verl-grpo_max_response1280...
if [[ "$ADV_ESTIMATOR" == "gae" ]]; then
  SUFFIX+="_critic-$(basename $CRITIC_MODEL_PATH)"
fi
FINAL_RUN_NAME="${MODEL_NAME}_${EXP_NAME}_${RUN_NAME}${SUFFIX}"

# Update the log file path to use the new name
LOG_FILE_PATH="$HDFS_LOG_PATH/$FINAL_RUN_NAME.log"
# The EXP_NAME variable is now part of the FINAL_RUN_NAME

# 1. 定义本次运行的专属目录 (与 trainer.default_local_dir 一致)
RUN_DIRECTORY="$HDFS_CHECKPOINT_PATH/$FINAL_RUN_NAME"

# 2. 确保这个目录存在 (-p 会创建所有父目录且不会因目录已存在而报错)
mkdir -p "$RUN_DIRECTORY"

# 3. 定义参数日志文件的完整路径
PARAMS_LOG_FILE="$RUN_DIRECTORY/hyperparameters.log"

# 4. 将调用此脚本的完整命令和所有参数写入日志文件
#    使用我们在一开始备份的 FULL_ARGS 变量，而不是已被消耗的 $@
echo "bash $0 $FULL_ARGS" > "$PARAMS_LOG_FILE"

echo "----------------------------------------------------"
echo "超参数配置已保存到: $PARAMS_LOG_FILE"
echo "----------------------------------------------------"


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

if [ -n "$REWARD_EMA_ALPHA" ]; then
  HYDRA_OVERRIDES+=("reward_manager.ema_alpha=$REWARD_EMA_ALPHA")
fi
if [ -n "$REWARD_INDICATOR_NAMES" ]; then
  HYDRA_OVERRIDES+=("reward_manager.indicator_names=$REWARD_INDICATOR_NAMES")
fi
if [ -n "$REWARD_WEIGHTS" ]; then
  HYDRA_OVERRIDES+=("reward_manager.weights=$REWARD_WEIGHTS")
fi
if [ -n "$REWARD_WEIGHTS_EXPLOIT" ]; then
  HYDRA_OVERRIDES+=("reward_manager.weights_exploit=$REWARD_WEIGHTS_EXPLOIT")
fi
if [ -n "$ADD_REWARD" ]; then
  HYDRA_OVERRIDES+=("reward_manager.add_reward=$ADD_REWARD")
fi
if [ -n "$COMPUTE_LOG_EFFECTIVE_RANK" ]; then
  HYDRA_OVERRIDES+=("calculator.compute_log_effective_rank=$COMPUTE_LOG_EFFECTIVE_RANK")
fi
if [ -n "$METRIC_INDICES" ]; then
  HYDRA_OVERRIDES+=("calculator.metric_indices=$METRIC_INDICES")
fi
if [ -n "$MODULATION_GAIN" ]; then
  HYDRA_OVERRIDES+=("reward_manager.modulation_gain=$MODULATION_GAIN")
fi
if [ -n "$OUTPUT_TOKEN_LEVEL_METRICS" ]; then
  HYDRA_OVERRIDES+=("calculator.output_token_level_metrics=$OUTPUT_TOKEN_LEVEL_METRICS")
fi
if [ -n "$CRITIC_MODEL_PATH" ]; then
  HYDRA_OVERRIDES+=("critic.model.path=$CRITIC_MODEL_PATH")
fi
if [ -n "$TOKEN_LEVEL_BASELINE_TYPE" ]; then
  HYDRA_OVERRIDES+=("reward_manager.token_level_baseline_type=$TOKEN_LEVEL_BASELINE_TYPE")
fi
if [ -n "$RETURN_HIDDEN_STATES" ]; then
  HYDRA_OVERRIDES+=("actor_rollout_ref.rollout.return_hidden_states=$RETURN_HIDDEN_STATES")
fi
if [ -n "$RETURN_PREFILL" ]; then
  HYDRA_OVERRIDES+=("actor_rollout_ref.rollout.return_prefill=$RETURN_PREFILL")
fi
if [ -n "$RETURN_DECODE" ]; then
  HYDRA_OVERRIDES+=("actor_rollout_ref.rollout.return_decode=$RETURN_DECODE")
fi
if [ -n "$ADD_ADV" ]; then
  HYDRA_OVERRIDES+=("algorithm.add_adv=$ADD_ADV")
fi
if [ -n "$AUX_FIX" ]; then
  HYDRA_OVERRIDES+=("reward_manager.aux_fix=$AUX_FIX")
fi
if [ -n "$ADV_SHAPING_KAPPA" ]; then
HYDRA_OVERRIDES+=("algorithm.adv_shaping_kappa=$ADV_SHAPING_KAPPA")
fi
if [ -n "$SVD_RANK" ]; then
  HYDRA_OVERRIDES+=("calculator.svd_rank=$SVD_RANK")
fi
if [ -n "$SVD_NITER" ]; then
  HYDRA_OVERRIDES+=("calculator.svd_niter=$SVD_NITER")
fi
if [ -n "$ZEROTH_ORDER_SVD_METHOD" ]; then
  HYDRA_OVERRIDES+=("calculator.zeroth_order_svd_method=$ZEROTH_ORDER_SVD_METHOD")
fi
if [ -n "$DIFF_SVD_METHOD" ]; then
  HYDRA_OVERRIDES+=("calculator.diff_svd_method=$DIFF_SVD_METHOD")
fi
if [ -n "$HYPOTHESIS_TYPE" ]; then
  HYDRA_OVERRIDES+=("reward_manager.hypothesis_type=$HYPOTHESIS_TYPE")
fi
if [ -n "$COMPUTE_GLOBAL_METRICS" ]; then 
  HYDRA_OVERRIDES+=("calculator.compute_global_metrics=$COMPUTE_GLOBAL_METRICS")
fi
if [ -n "$COMPUTE_CUMULATIVE_GLOBAL_METRICS" ]; then
  HYDRA_OVERRIDES+=("calculator.compute_cumulative_global_metrics=$COMPUTE_CUMULATIVE_GLOBAL_METRICS")
fi
if [ -n "$GLOBAL_DIFF_STRIDE_TRAIN" ]; then 
  HYDRA_OVERRIDES+=("calculator.global_diff_stride_train=$GLOBAL_DIFF_STRIDE_TRAIN")
fi
if [ -n "$GLOBAL_DIFF_STRIDE_VAL" ]; then 
  HYDRA_OVERRIDES+=("calculator.global_diff_stride_val=$GLOBAL_DIFF_STRIDE_VAL")
fi
if [ -n "$AUX_REWARD_GLOBAL_WEIGHT" ]; then
  HYDRA_OVERRIDES+=("reward_manager.aux_reward_global_weight=$AUX_REWARD_GLOBAL_WEIGHT")
fi
ray job submit --address=${HEAD_IP}:${HEAD_PORT} \
  --entrypoint-num-cpus=1 \
  --runtime-env-json="${RAY_RUNTIME_ENV_JSON}" \
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
  actor_rollout_ref.rollout.name=hidden_vllm \
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
  trainer.experiment_name=$FINAL_RUN_NAME \
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

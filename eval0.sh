#!/bin/bash

# =======================================================
#               参数配置区域
# =======================================================
DTYPE="torch.bfloat16"
HDFS_PATH="/home/root1/Fanding/simpleRL-reason/custom"
MODEL_BASE_PATH="/media/root1/4t/Models"
GPU_MEMORY_UTILIZATION=0.70
# "aime25,amc24,aime24,amc23,aqua,asdiv,carp_en,cmath,cn_middle_school,college_math,gaokao2023en,gaokao2024_I,gaokao2024_II,gaokao2024_mix,gaokao_math_cloze,gaokao_math_qa,gsm8k,math,math500,mawps,minerva_math,mmlu_stem,olympiadbench,sat_math,svamp,tabmwp" 

# --- 全局默认参数 ---
# 将所有需要测试的数据集统一放在这里
BENCHMARKS="aime25,amc24,aime24,amc23"
DEFAULT_TEMPLATE="abel"
DEFAULT_N_SAMPLING=256
DEFAULT_SPECIFIC_STEPS="140"
TEMPERATURES=(0.6)
MAX_RESPONSE_LENGTH=(1280)
TOP_P=0.95
USE_WANDB="false"
CALCULATE_METRICS="false"
METRICS_TO_CALC="Effective Rank"
METRIC_ORDERS="0,1,2"
METRIC_STRIDE=20
RUN_COLLECT_RESULTS="false"

# =======================================================
#               模型与运行配置
# =======================================================
# --- 步骤 1: 定义简短的“别名”，并用一个普通数组控制执行顺序 ---
RUN_ALIASES=(
    "llama_1B_er_grpo"
    "llama_ppo"
)

# --- 步骤 2: 使用“别名”作为Key，定义各个配置字典 ---

# 别名 -> 完整的 Run Name
declare -A RUN_NAME_MAP
RUN_NAME_MAP=(
    ["llama_1B_er_grpo"]="llama/Llama-3.2-1B-Instruct_er_adv_allfull_verl-grpo_max_response1024_grpo_batch48_ppomini24_valbatch48_rollout4_logprobbatch1_klcoef0.001_entcoef0.001_epochs2_simplelr_abel_gsm8k_level1_stride40_mgain1.0_auxgw0.5_ema0.3"
    ["llama_ppo"]="llama/Llama-3.2-1B-Instruct_originPPO_verl-grpo_max_response1024_gae_batch48_ppomini24_valbatch48_rollout1_logprobbatch1_klcoef0.001_entcoef0.001_epochs2_simplelr_abel_gsm8k_level1_stride20_mgain1.0_auxgw0.2_ema0.3_critic-DeepSeek-R1-Distill-Qwen-1.5B"
)

# 别名 -> 基础模型
declare -A BASE_MODEL_MAP
BASE_MODEL_MAP=(
    ["llama_1B_er_grpo"]="Llama-3.2-1B-Instruct"
    ["llama_ppo"]="Llama-3.2-1B-Instruct"
)

# 别名 -> 特定评测步骤
declare -A STEP_MAP
STEP_MAP=(
    ["llama_1B_er_grpo"]="140"
    ["llama_ppo"]="140"
)

# 别名 -> 特定模板
declare -A TEMPLATE_MAP
TEMPLATE_MAP=(
    ["llama_1B_er_grpo"]="abel"
    ["llama_ppo"]="abel"
)

# =======================================================
#               嵌套循环主体
# =======================================================

# 外层循环: 遍历定义好顺序的“别名”数组
for alias in "${RUN_ALIASES[@]}"
do
    # --- 使用别名从各个字典中获取配置信息 ---
    run_name="${RUN_NAME_MAP[$alias]}"
    init_model_basename="${BASE_MODEL_MAP[$alias]}"
    
    # 如果 run_name 或 basename 未定义，说明配置有误，跳过以防出错
    if [ -z "$run_name" ] || [ -z "$init_model_basename" ]; then
        echo "Warning: Configuration for alias '${alias}' is incomplete. Skipping."
        continue
    fi

    # 准备模型路径
    model_family_dir=$(dirname "${run_name}")
    init_model_relative_path="${model_family_dir}/${init_model_basename}"

    # 动态决定评测步骤
    current_specific_steps=${STEP_MAP[$alias]:-$DEFAULT_SPECIFIC_STEPS}
    
    # 动态决定是否添加 step 0
    add_step_0="false"
    if [[ "${current_specific_steps}" =~ (^|,)0(,|$) ]]; then
        add_step_0="true"
    fi

    # 动态决定使用哪个模板
    current_template=${TEMPLATE_MAP[$alias]:-$DEFAULT_TEMPLATE}

    # 循环 2: 遍历每一个 max_response_length
    for max_response_length in "${MAX_RESPONSE_LENGTH[@]}"
    do
        # 循环 3: 遍历每一个温度值
        for temp in "${TEMPERATURES[@]}"
        do
            # 由于不再循环单个数据集，n_sampling 直接使用默认值
            current_n_sampling=${DEFAULT_N_SAMPLING}

            echo "========================================================================"
            echo ">>>>>  RUNNING EVALUATION FOR ALIAS: ${alias}"
            echo ">>>>>  Run Name: ${run_name}"
            echo ">>>>>  Template: ${current_template}, N_Sampling: ${current_n_sampling}, STEPS: ${current_specific_steps}"
            echo ">>>>>  Benchmarks: ${BENCHMARKS}"
            echo "========================================================================"

            FINAL_OUTPUT_DIR="eval_results_temp_${temp}_maxlen${max_response_length}_n${current_n_sampling}_calc${CALCULATE_METRICS}"
            
            # 直接调用脚本，并传递完整的 benchmark 列表
            bash eval_math_nodes.sh \
                --run_name "${run_name}" \
                --template "${current_template}" \
                --init_model "${init_model_relative_path}" \
                --tp_size 2 \
                --add_step_0 ${add_step_0}  \
                --temperature ${temp} \
                --top_p ${TOP_P} \
                --max_tokens ${max_response_length} \
                --benchmarks "${BENCHMARKS}" \
                --n_sampling ${current_n_sampling} \
                --visible_gpus 0,1 \
                --output_dir "${FINAL_OUTPUT_DIR}" \
                --use_wandb_arg ${USE_WANDB} \
                --calculate_metrics ${CALCULATE_METRICS} \
                --metrics_to_calc "${METRICS_TO_CALC}" \
                --metric_orders "${METRIC_ORDERS}" \
                --metric_stride ${METRIC_STRIDE} \
                --specific_steps "${current_specific_steps}" \
                --num_test_sample_per_dataset -1 \
                --hdfs_home "${HDFS_PATH}" \
                --init_model_base_path "${MODEL_BASE_PATH}" \
                --dtype "${DTYPE}" \
                --run_collect_results "${RUN_COLLECT_RESULTS}" \
                --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION}
        done
    done
done

echo "========================================================================"
echo "All evaluations are complete."
echo "========================================================================"

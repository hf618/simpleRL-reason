#!/bin/bash
set -e

# =======================================================
#               可配置参数
# =======================================================

# --- 实验和模型标识 ---
RUN_NAME="er_di_2_3_verl-grpo_Qwen2.5-3B_max_response1280_batch48_rollout6_valbatch6_ppomini24_logprobbatch1_klcoef0.001_entcoef0.001_epochs1_simplelr_math_35"
INIT_MODEL="Qwen2.5-3B"
BASE_CHECKPOINT_PATH="/home/hfd24/simpleRL-reason/custom/checkpoint"
OUTPUT_DIR_NAME="eval_results_test"

# --- 可视化目标 ---
DATASET="math500"
STEPS_TO_VISUALIZE="0,100"

# --- 指标配置 ---
METRIC_NAME="Effective Rank"
METRIC_ORDER="1"
METRIC_STRIDE=20

# --- 评估时的参数（用于构造文件名） ---
TEMPLATE="qwen-boxed"
TEMPERATURE=0.01

# --- 输出文件 ---
OUTPUT_HTML="visuals/comparison_v3_${DATASET}_${METRIC_NAME}.html"

# =======================================================
#          【新】问题采样模式开关
# =======================================================
# MODE: 'direct' 或 'rule'
#   - direct: 直接使用下面的 PROBLEM_IDS_TO_VISUALIZE 列表
#   - rule:   使用下面的 SAMPLING_RULE 自动筛选问题
SAMPLING_MODE='rule'

# --- 模式1: 'direct' 配置 ---
PROBLEM_IDS_TO_VISUALIZE="1,2,3"

# --- 模式2: 'rule' 配置 ---
# 规则: 一个JSON格式的字符串。键是step, 值是布尔型 (true代表正确, false代表错误)
# 示例: 查找 "在step 0 回答正确，但在step 100 回答错误" 的问题
SAMPLING_RULE='{"0": true, "100": false}'
NUM_SAMPLES=5 # 希望从符合规则的样本中采样多少个

# =======================================================
#               脚本执行逻辑
# =======================================================
echo "开始生成对比可视化 (v3.0)..."
echo "当前采样模式: ${SAMPLING_MODE}"

# 根据阶数自动构造指标全名
FULL_METRIC_NAME=${METRIC_NAME}
if [ "$METRIC_ORDER" = "1" ]; then
    FULL_METRIC_NAME="${METRIC_NAME} diff"
elif [ "$METRIC_ORDER" = "2" ]; then
    FULL_METRIC_NAME="${METRIC_NAME} diff 2"
fi

# 创建输出目录
mkdir -p $(dirname ${OUTPUT_HTML})

# 根据模式构建不同的Python命令参数
CMD_ARGS=""
if [ "$SAMPLING_MODE" = "direct" ]; then
    CMD_ARGS="--sampling_mode direct --problem_ids ${PROBLEM_IDS_TO_VISUALIZE}"
elif [ "$SAMPLING_MODE" = "rule" ]; then
    CMD_ARGS="--sampling_mode rule --sampling_rule '${SAMPLING_RULE}' --num_samples ${NUM_SAMPLES}"
else
    echo "错误: 无效的 SAMPLING_MODE: ${SAMPLING_MODE}"
    exit 1
fi

# 调用Python脚本
# 使用 eval 来正确处理带空格和引号的参数
eval python examples/simplelr_math_eval/visualize_metrics.py \
    --base_eval_dir "'${BASE_CHECKPOINT_PATH}/${RUN_NAME}/${OUTPUT_DIR_NAME}'" \
    --steps "'${STEPS_TO_VISUALIZE}'" \
    --dataset "'${DATASET}'" \
    --metric_to_visualize "'${FULL_METRIC_NAME}'" \
    --metric_stride ${METRIC_STRIDE} \
    --init_model_path "'/home/hfd24/model/qwen/${INIT_MODEL}'" \
    --template "'${TEMPLATE}'" \
    --temperature ${TEMPERATURE} \
    --output_html_path "'${OUTPUT_HTML}'" \
    ${CMD_ARGS}

echo "所有任务完成!"
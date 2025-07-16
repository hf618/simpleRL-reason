#!/bin/bash
set -x

# 参数检查
if [ "$#" -gt 20 ]; then # 增加参数个数检查
    echo "Usage: $0 <eval_script_path> <base_checkpoint_path> <init_model_path> <template> [benchmarks] [temperature] [max_tokens] [top_p] [tp_size] [ckpt_list_file] [output_dir] [overwrite] [n_sampling] [visible_gpus] [calculate_metrics] [metrics_to_calc] [metric_stride] [metric_orders] [num_test_sample_per_dataset] [dtype]"
    exit 1
fi

# 获取参数
eval_script_path=$1
base_checkpoint_path=$2
init_model_path=$3
template=$4
benchmarks=$5
temperature=$6
max_tokens=$7
top_p=$8
tp_size=${9:-1} 
ckpt_list_file=${10:-""} 
output_dir_base=${11:-"eval_results"}
overwrite=${12:-false}
n_sampling=${13:-1}
# output_dir="${output_dir_base}_n${n_sampling}"
output_dir="${output_dir_base}"
actor_dir="actor"

visible_gpus=${14:-""}
calculate_metrics=${15:-"false"}
metrics_to_calc=${16:-""}
metric_stride=${17:-1}
metric_orders=${18:-"0,1,2"}
num_test_sample_per_dataset=${19:--1}  # 默认值为 -1，表示使用所有样本
dtype=${20:-"torch.float16"}
# visible_gpus=${14:-""} 
# # 设置可见的 GPU
# if [ -n "$visible_gpus" ]; then
#     export CUDA_VISIBLE_DEVICES="$visible_gpus"
#     echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES" >&2
# fi
# 设置可见的 GPU
VISIBLE_GPUS_ARRAY=()
if [ -n "$visible_gpus" ]; then
    IFS=',' read -r -a VISIBLE_GPUS_ARRAY <<< "$visible_gpus"
    export CUDA_VISIBLE_DEVICES="$visible_gpus"
    echo "CUDA_VISIBLE_DEVICES set to: $visible_gpus" >&2
else
    # 默认使用全部 GPU
    mapfile -t VISIBLE_GPUS_ARRAY < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
# 获取可用的GPU数量
get_visible_gpus_count() {
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | xargs
    else
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^$' | wc -l | xargs
    fi
}
NUM_GPUS=$(get_visible_gpus_count)

# NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
# NUM_GPU_GROUPS=$((NUM_GPUS / tp_size))  # 计算可用的GPU组数
NUM_GPUS=${#VISIBLE_GPUS_ARRAY[@]}
NUM_GPU_GROUPS=$((NUM_GPUS / tp_size))

# 函数：复制 tokenizer 文件
copy_tokenizer_files() {
    local ckpt_path=$1
    local init_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        "config.json"
        "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    if [ -f "$init_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi
    # 创建目标路径，确保它存在
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path" >&2
    else
        echo "Checkpoint directory already exists: $ckpt_path" >&2
    fi

    # 复制每个文件
    for filename in "${files_to_copy[@]}"; do
        src="$init_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src to $dst"
        else
            echo "Warning: $src does not exist."
        fi
    done
}

# 函数：获取所有需要评估的检查点，并过滤掉已评估的
get_checkpoints_to_evaluate() {
    local base_path="$1"
    
    if [ -n "$ckpt_list_file" ] && [ -f "$ckpt_list_file" ]; then
        # Read checkpoints from the provided file
        cat "$ckpt_list_file"
    else
        # Original logic for getting all checkpoints
        local checkpoints=()
        for ckpt_dir in "$base_path"/global_step_*; do
            if [ -d "$ckpt_dir" ]; then
                step_tag=$(basename "$ckpt_dir")
                checkpoints+=("$step_tag")
            fi
        done
        
        if [ ${#checkpoints[@]} -eq 0 ]; then
            echo ""
        else
            printf "%s\n" "${checkpoints[@]}"
        fi
    fi
}

# 函数：在指定GPU上处理单个检查点
process_checkpoint() {
    local start_idx=$((group_id * tp_size))
    local gpu_ids=""

    for ((i=0; i<tp_size; i++)); do
        physical_gpu=${VISIBLE_GPUS_ARRAY[$((start_idx + i))]}
        if [ -z "$physical_gpu" ]; then
            echo "Error: Not enough visible GPUs available for group $group_id." >&2
            exit 1
        fi
        if [ -n "$gpu_ids" ]; then
            gpu_ids="${gpu_ids},"
        fi
        gpu_ids="${gpu_ids}${physical_gpu}"
    done
    
    ckpt_path="$base_checkpoint_path/$step_tag/$actor_dir/huggingface"
    
    echo "Evaluating checkpoint $step_tag on GPUs $gpu_ids" >&2
    
    output_path_new="$base_checkpoint_path/$output_dir/$step_tag"
    mkdir -p "$output_path_new"
    
    CUDA_VISIBLE_DEVICES=$gpu_ids bash "$eval_script_path" \
        ${template} "$ckpt_path" "$output_path_new" "$temperature" \
        "$max_tokens" "$top_p" "$benchmarks" "$overwrite" "$n_sampling" \
        "$calculate_metrics" "$metrics_to_calc" "$metric_stride" "$metric_orders" "$num_test_sample_per_dataset" \
        "$dtype" 


}

# 记录当前工作目录
original_dir=$(pwd)

# 主脚本部分修改
# 获取需要评估的检查点
readarray -t checkpoints_to_evaluate < <(get_checkpoints_to_evaluate "$base_checkpoint_path")

if [ ${#checkpoints_to_evaluate[@]} -eq 0 ]; then
    echo "No new checkpoints to evaluate." >&2
    exit 0
fi

# 检查GPU数量是否满足tp_size要求
if [ $((NUM_GPUS % tp_size)) -ne 0 ]; then
    echo "Error: Number of available GPUs ($NUM_GPUS) is not divisible by tp_size ($tp_size)" >&2
    exit 1
fi

echo "Found ${#checkpoints_to_evaluate[@]} checkpoints to evaluate:" >&2
printf '%s\n' "${checkpoints_to_evaluate[@]}" >&2
total_checkpoints=${#checkpoints_to_evaluate[@]}
eval_count=0
# 并行处理检查点，按GPU组分配
for i in "${!checkpoints_to_evaluate[@]}"; do
    group_id=$((i % NUM_GPU_GROUPS))
    step_tag="${checkpoints_to_evaluate[i]}"
    
    # 在后台启动处理任务
    process_checkpoint "$step_tag" "$group_id" 
    
    # 每启动NUM_GPU_GROUPS个任务后等待它们完成
    if [ $(((i + 1) % NUM_GPU_GROUPS)) -eq 0 ]; then
        wait
    fi
    eval_count=$((eval_count + 1))
    echo "Evaluating $eval_count/$total_checkpoints checkpoints ..."
done

# 等待所有剩余的后台任务完成
wait

cd "$original_dir"
echo "All conversions and evaluations completed."
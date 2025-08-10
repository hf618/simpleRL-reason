# # 小学生守则
# 1. False / True 字符别写错了
# 2. --enable_calculator True 记得要和model_runner对应（古法炼钢，在源码里魔改）
# 3. hidden states 是否通过 norm 要明确记录, 数据类型一开始就干成 bfloat16
# 4. 是否进行effective rank 中心化
bash train_grpo_math_tune_ray.sh \
    --model_name llama/Llama-3.2-1B-Instruct --max_response_length 1024 \
    --critic_model_path "" \
    --train_batch_size 48 --ppo_mini_batch_size 24 --val_batch_size 48 --rollout_n 4 \
    --ppo_micro_batch_size 1 --log_prob_micro_batch_size 1 --micro_rollout_batch_size 1 \
    --kl_loss_coef 0.001 --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.70 \
	--logger_config "['console','wandb']" \
    --rollout_tp 1 --save_freq 20 --test_freq 5 --total_epochs 2 \
    --exp_name "er_pyr_3_new_normcen" --add_reward True --dataset_name "simplelr_abel_gsm8k_level1" \
    --val_before_train True --val_sample_size -1 \
    --enable_calculator True --metric_indices "[0,1]" \
    --reward_weights "[0.0, 0.0, 1.0]" --reward_weights_exploit "[0.0, 1.0, 0.0]" \
    --reward_indicator_names "['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']" \
    --output_token_level_metrics False --compute_log_effective_rank False \
    --diff_stride 20 --modulation_gain 1.0 --aux_reward_global_weight 0.2 --reward_ema_alpha 0.3 --adv_estimator "grpo" 

# python monitor_gpu.py -H 10 -S 2 -g 0 1 -o ./custom/log_gpu
# w1 "[1.0, 0.25, 0.0625]"
# w2 "[1.0, 0.50, 0.25]"
# w3 "[0.5, 0.25, 0.125]"
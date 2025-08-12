# # 小学生守则
# 1. False / True 字符别写错了
# 2. When enable_calculator is True, return_hidden_states and return_decode must be True
# 3. All hidden states are produced via norm layer, 数据类型一开始就干成 bfloat16 (V100s should modify this to float16)
# 4. All effective rank and entropy calculation via centered matrix
# 5. If use PPO: critic_model_path give absolute path, and adv_estimator "gae"
bash train_grpo_math_tune_ray.sh \
    --model_name llama/Llama-3.2-1B-Instruct --max_response_length 1024 \
    --critic_model_path "/media/root1/4t/Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --adv_estimator "gae" \
    --train_batch_size 4 --ppo_mini_batch_size 4 --val_batch_size 48 --rollout_n 1 \
    --ppo_micro_batch_size 1 --log_prob_micro_batch_size 1 --micro_rollout_batch_size 1 \
    --kl_loss_coef 0.001 --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.70 \
	--logger_config "['console','wandb']" \
    --rollout_tp 2 --save_freq 20 --test_freq 5 --total_epochs 2 \
    --exp_name "originPPO" --add_reward False --dataset_name "simplelr_abel_gsm8k_level1" \
    --val_before_train False --val_sample_size -1 \
    --enable_calculator True --metric_indices "[0,1]" \
    --reward_weights "[0.0, 0.0, 1.0]" --reward_weights_exploit "[0.0, 1.0, 0.0]" \
    --reward_indicator_names "['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']" \
    --output_token_level_metrics False --compute_log_effective_rank False \
    --diff_stride 20 --modulation_gain 1.0 --aux_reward_global_weight 0.2 --reward_ema_alpha 0.3 \
    --return_hidden_states True --return_prefill False --return_decode True

# python monitor_gpu.py -H 10 -S 2 -g 0 1 -o ./custom/log_gpu
# w1 "[1.0, 0.25, 0.0625]"
# w2 "[1.0, 0.50, 0.25]"
# w3 "[0.5, 0.25, 0.125]"
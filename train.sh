bash train_grpo_math_tune_ray.sh \
    --model_name Qwen2.5-7B --max_response_length 2560 \
    --train_batch_size 48 \
    --rollout_n 8 \
    --val_batch_size 48 \
    --ppo_mini_batch_size 24 \
    --ppo_micro_batch_size 2 \
    --log_prob_micro_batch_size 2 \
    --micro_rollout_batch_size 2 \
    --kl_loss_coef 0.001 \
    --entropy_coeffient 0.001 \
    --rollout_gpu_memory_util 0.58 \
	--logger_config "['console','wandb']" \
    --rollout_tp 4 --save_freq 20 --test_freq 5 --total_epochs 2 \
    --exp_name "er_pyr_3" --dataset_name "simplelr_qwen_level3to5" \
    --reward_ema_alpha 0.5 \
    --reward_weights "[1.0, 0.25, 0.0625]" \
    --reward_weights_inner "[1.0, 1e-1, 1e-5]" \
    --reward_indicator_names "['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']" \
    --val_before_train True \
    --val_sample_size -1 \
    --diff_stride 20 --enable_calculator True \
    --add_reward True --compute_log_effective_rank True 

# python monitor_gpu.py -H 10 -S 10 -g 0 1 2 3 -o ./custom/log_gpu
# "[1.0, 0.25, 0.0625]"
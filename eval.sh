bash eval_math_nodes.sh \
    --run_name verl-grpo_Qwen2.5-3B_max_response1280_batch48_rollout6_valbatch6_ppomini24_logprobbatch2_klcoef0.001_entcoef0.001_epochs3_simplelr_qwen_level3to5_en_di_2_23   \
    --init_model Qwen2.5-3B \
    --template qwen-boxed  \
    --tp_size 4 \
    --add_step_0 true  \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 900 \
    --visible_gpus 0,1,2,3 \
    --benchmarks aime24,amc23,math500,olympiadbench,gsm8k,minerva_math \
    --n_sampling 1 
    # --output_dir "eval_results_n${n_sampling}" 
    
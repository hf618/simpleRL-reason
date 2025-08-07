# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import kk
# from verl.utils.reward_score import simplelr_math
# from verl.utils.reward_score import deepseek_r1
from verl.utils.reward_score import hf_math_verify
from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
def _default_compute_score(data_source, solution_str, ground_truth):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score(solution_str, ground_truth)
    # elif data_source.lower() == "simplelr_math500" or data_source.lower() == "simplelr_aime24":
    #     return hf_math_verify.compute_accuracy(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        return math.compute_score(solution_str, ground_truth)
    
    elif "kk" in data_source:
        return kk.compute_score(solution_str, ground_truth)
    elif "simplelr" in data_source:
        return hf_math_verify.compute_score(solution_str, ground_truth)
    elif "deepseek_r1" in data_source:
        return deepseek_r1.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
    
def _custom_compute_score(data_source, solution_str, ground_truth):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score(solution_str, ground_truth)
    # elif data_source.lower() == "simplelr_math500" or data_source.lower() == "simplelr_aime24":
    #     return hf_math_verify.compute_accuracy(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        return math.compute_score(solution_str, ground_truth)
    
    elif "kk" in data_source:
        return kk.compute_score(solution_str, ground_truth)
    elif "simplelr" in data_source:
        return hf_math_verify.compute_score_custom(solution_str, ground_truth)
    elif "deepseek_r1" in data_source:
        return deepseek_r1.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError



class RewardManager():
    """
    Optimized version based on the user's final request:
    1. Uses the nuanced 'score' for both the direct reward (reward_tensor_0)
       and for updating the performance EMA that controls the global scaling factor.
    2. Includes normalization to handle the [-1, 1] range of the score.
    """
    def __init__(self, tokenizer, num_examine, compute_score=None, calculator=None,
                 ema_alpha=0.7,
                 indicator_names=None,
                 weights=None,
                 weights_exploit=None,
                 calculator_enabled=True,
                 add_reward=True,
                 modulation_gain=1.5,
                 aux_reward_global_weight=1.0,
                 adv_estimator='grpo',
                 output_token_level_metrics=False,
                 token_level_baseline_type='internal_mean'):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.calculator = calculator
        self.ema_alpha = ema_alpha
        self.indicator_names = indicator_names if indicator_names is not None else \
            ['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']
        
        self.weights_explore = weights if weights is not None else [0.0, 0.0, 1.0]
        self.weights_exploit = weights_exploit if weights_exploit is not None else [0.0, 1.0, 0.0]

        self.mids = {name: 0.0 for name in self.indicator_names}
        self.add_reward = add_reward
        self.calculator_enabled = calculator_enabled
        self.modulation_gain = modulation_gain
        self.epsilon = 1e-8
        
        # ### MODIFIED: Tracks the EMA of the nuanced score ###
        # Initialized to 0.0, representing a neutral average score.
        self.ema_performance_score = 0.0 
        self.aux_reward_global_weight = aux_reward_global_weight
        self.adv_estimator = adv_estimator
        self.output_token_level_metrics = output_token_level_metrics
        self.token_level_baseline_type = token_level_baseline_type
        print(f"[RewardManager] Initialized with token-level baseline type: {self.token_level_baseline_type}")
        

    def __call__(self, data: DataProto, is_val=False, metrics_old=None, global_step=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # correctness_tensor = torch.zeros(len(data), dtype=torch.float32)
        reward_tensor_0 = torch.zeros_like(data.batch['responses'], dtype=torch.bfloat16)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.bfloat16)
        correctness_tensor = torch.zeros(len(data), dtype=torch.bfloat16)
        calculator_tensor = torch.zeros(len(data), dtype=torch.bfloat16)
        already_print_data_sources = {}

        # ### 新增: 初始化用于存储内部指标的字典 ###
        internal_metrics = {
            'percentage_deviation': [],
            'exploit_tendency': [],
            'performance_scaling_factor': []
            # 其他指标将在循环中动态添加
        }
        layer_key = '1'
        # ### NEW: Main gatekeeper for auxiliary reward ###
        # It's only possible to calculate if it's enabled AND it's not the first step (metrics_old exists).
        use_aux_reward = self.add_reward and self.calculator_enabled and metrics_old

        performance_scaling_factor = 1.0 # Default scaling factor for step 1

        # # sigmoid = nn.Sigmoid()
        # layer_key = '1'
        # if use_aux_reward:
        #     act_func = nn.Tanh()
        #     for i in range(len(self.indicator_names)):
        #         indicator_name = self.indicator_names[i]
        #         # 关键修改：增加一层对特定键是否存在的检查
        #         metric_key = f'cal/overall/layer_{layer_key}/{indicator_name}/mean'
        #         if metrics_old and metric_key in metrics_old:
        #             v = metrics_old[metric_key]
        #             self.mids[indicator_name] = ( 1 - self.ema_alpha ) * self.mids[indicator_name] +  self.ema_alpha * v

        #     # ### KEY ADJUSTMENT: NORMALIZATION ###
        #     # 1. Normalize the performance score EMA (from [-1, 1]) to [0, 1]
        #     normalized_performance = (self.ema_performance_score + 1.0) / 2.0
            
        #     # 2. Calculate the final scaling factor based on the normalized score
        #     performance_scaling_factor = self.aux_reward_global_weight * (1.0 - normalized_performance)
        act_func = nn.Tanh() if use_aux_reward else None

        # ### 修改点1: EMA更新逻辑已移至末尾，此处不再需要 ###
        if use_aux_reward:
            # 仅计算用于当前步骤的缩放因子
            normalized_performance = (self.ema_performance_score + 1.0) / 2.0
            performance_scaling_factor = self.aux_reward_global_weight * (1.0 - normalized_performance)
            internal_metrics['performance_scaling_factor'].append(performance_scaling_factor)

     
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']


            data_source = data_item.non_tensor_batch['data_source']

            score_dict = self.compute_score(data_source=data_source, solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor_0[i, valid_response_length - 1] = score_dict['score']
            correctness_tensor[i] = score_dict['correctness']

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)    



            reward_tensor[i, valid_response_length - 1] = reward_tensor_0[i, valid_response_length - 1]

            if use_aux_reward:
                # 1. Calculate the 'Percentage Deviation' as the guidance signal
                guidance_indicator_name = self.indicator_names[0] # Diff 2
                current_guidance_value = data_item.batch['calculator_results'][layer_key][guidance_indicator_name]
                ema_baseline = self.mids[guidance_indicator_name]
                
                percentage_deviation = (current_guidance_value - ema_baseline) / (abs(ema_baseline) + self.epsilon)
                
                # We can still clamp this to prevent extreme values from having too much influence
                percentage_deviation = torch.clamp(percentage_deviation, -5.0, 5.0)


                
                # 3. Interpolate between explore and exploit weight profiles
                w_explore = torch.tensor(self.weights_explore, device=data.batch.device)
                w_exploit = torch.tensor(self.weights_exploit, device=data.batch.device)


                # --- 实验一：测试假说A (高diff 2 = 利用) ---
                # 变量名清晰地反映了它的作用
                exploit_tendency = torch.sigmoid(self.modulation_gain * percentage_deviation)
                # 当exploit_tendency趋近1时，权重偏向w_exploit
                dynamic_weights = (1.0 - exploit_tendency) * w_explore +  exploit_tendency * w_exploit


                # # --- 实验二：测试假说B (高diff 2 = 探索) ---
                # # 变量名也清晰地反映了它的作用
                # explore_tendency = torch.sigmoid(self.modulation_gain * percentage_deviation)
                # # 当explore_tendency趋近1时，权重偏向w_explore
                # dynamic_weights = explore_tendency * w_explore + (1.0 - explore_tendency) * w_exploit
                
                # Create a lookup for easier access
                weights_map = {name: weight for name, weight in zip(self.indicator_names, dynamic_weights)}

                # ### 新增: 记录 batch 的平均值 ###
                internal_metrics['percentage_deviation'].append(percentage_deviation.item())
                internal_metrics['exploit_tendency'].append(exploit_tendency.item())
                # ### 新增: 记录动态权重 ###
                for name, weight in weights_map.items():
                    log_name = f"weight_{name.replace(' ', '_').lower()}"
                    if log_name not in internal_metrics:
                        internal_metrics[log_name] = []
                    internal_metrics[log_name].append(weight.item())

                # Case 1: GAE with token-level metrics (dense reward)
                #  
                if self.adv_estimator == 'gae' and self.output_token_level_metrics:
                    aux_reward_per_token = torch.zeros(valid_response_length, device=data.batch.device, dtype=torch.bfloat16)
                    for indicator_name in self.indicator_names:
                        token_level_indicator = data_item.batch['calculator_results'][layer_key][f"{indicator_name}_token_level"]
                        valid_token_level_indicator = token_level_indicator[:valid_response_length]
                        
                        # ### 核心修改：实现分支逻辑 ###
                        baseline = 0.0
                        if self.token_level_baseline_type == 'internal_mean':
                            # --- 新思路：使用内部动态基准 ---
                            baseline = torch.mean(valid_token_level_indicator)
                        
                        elif self.token_level_baseline_type == 'external_ema':
                            # --- 老办法：使用外部历史EMA ---
                            baseline = self.mids[indicator_name]
                        else:
                            raise ValueError(f"Invalid token_level_baseline_type: {self.token_level_baseline_type}")

                        relative_deviation_tensor = (valid_token_level_indicator - baseline) / (torch.abs(baseline) + self.epsilon)
                        relative_deviation_tensor = torch.clamp(relative_deviation_tensor, -5.0, 5.0)
                        
                        
                        # (日志记录逻辑不变)
                        log_name = f"relative_deviation_{indicator_name.replace(' ', '_').lower()}"
                        if log_name not in internal_metrics: internal_metrics[log_name] = []
                        internal_metrics[log_name].append(relative_deviation_tensor.mean().item())

                        aux_reward_per_token += act_func(relative_deviation_tensor) * weights_map[indicator_name]
                    
                    final_aux_reward = aux_reward_per_token * performance_scaling_factor
                    reward_tensor[i, :valid_response_length] += final_aux_reward
                
                # Case 2: All other cases (GRPO or sequence-level metrics) (sparse reward)
                else:
                    #  
                    calculator_tensor_i = 0.0
                    for indicator_name in self.indicator_names:
                        original_indicator = data_item.batch['calculator_results'][layer_key][indicator_name]
                        relative_deviation = (original_indicator - self.mids[indicator_name]) / (abs(self.mids[indicator_name]) + self.epsilon)
                        relative_deviation = torch.clamp(relative_deviation, -5.0, 5.0)

                        # Log the scalar relative deviation
                        log_name = f"relative_deviation_{indicator_name.replace(' ', '_').lower()}"
                        if log_name not in internal_metrics: internal_metrics[log_name] = []
                        internal_metrics[log_name].append(relative_deviation.item())
                        
                        calculator_tensor_i += act_func(relative_deviation) * weights_map[indicator_name]

                    final_aux_reward = calculator_tensor_i * performance_scaling_factor
                    reward_tensor[i, valid_response_length - 1] += final_aux_reward



        
        # ### 修改点2: 将所有EMA更新逻辑集中在此处 ###
        if use_aux_reward and not is_val:
            # 1. 更新性能得分的EMA
            self.ema_performance_score = (1 - self.ema_alpha) * self.ema_performance_score + \
                                                self.ema_alpha * reward_tensor_0.sum(dim=-1).float().mean().cpu().item()
            
            # 2. 更新各个指标的EMA (即 self.mids)
            
            for indicator_name in self.indicator_names:
                metric_key = f'cal/overall/layer_{layer_key}/{indicator_name}/mean'
                if metric_key in metrics_old:
                    v = metrics_old[metric_key]
                    self.mids[indicator_name] = (1 - self.ema_alpha) * self.mids[indicator_name] + self.ema_alpha * v

  
        return {"reward_tensor": reward_tensor, 
                "correctness_tensor": correctness_tensor, 
                "reward_tensor_0": reward_tensor_0,
                "internal_metrics": internal_metrics}


class RepresentationMetricsCalculator():
    """Calculates representation quality metrics from hidden states with memory optimization."""
    
    def __init__(self, tokenizer, max_seq_len=512, svd_rank=6, compute_log_effective_rank=False, metric_indices=None, output_token_level_metrics=False):
        """
        Initializes the RepresentationMetricsCalculator.

        Args:
            tokenizer: The tokenizer object (not directly used in metric calculation, but for context).
            max_seq_len (int): Maximum sequence length to process for memory optimization. Defaults to 512.
            svd_rank (int): Number of singular values to retain for SVD-based calculations. Defaults to 6.
            compute_log_effective_rank (bool): If True, calculates and includes the log of Effective Rank
                                               and its differences. Defaults to False.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len  # Controls the maximum sequence length processed
        self.svd_rank = svd_rank        # Number of singular values retained for SVD
        self._cached_tensors = {}       # Cache for reusing intermediate results
        self.compute_log_effective_rank = compute_log_effective_rank # New flag for log effective rank
        self.output_token_level_metrics = output_token_level_metrics
        self.epsilon = 1e-8 # 添加 epsilon


        # 定义所有可用的基础指标和它们的计算函数
        all_base_metrics = [
            ("Response Entropy 1", self.calculate_response_entropy),
            # 使用 lambda 确保可以传递额外参数
            ("Effective Rank", lambda hs, mask: self.calculate_effective_rank(hs, mask, log_output=False)),
            ("Curvature", self.calculate_curvature)
        ]

        # 如果需要，动态添加 Log Effective Rank
        if self.compute_log_effective_rank:
            all_base_metrics.append(
                ("Log Effective Rank", lambda hs, mask: self.calculate_effective_rank(hs, mask, log_output=True))
            )
        
        # 根据传入的索引筛选出需要计算的指标
        if metric_indices is None:
            # 如果没有提供索引，默认使用所有指标
            self.selected_metrics = all_base_metrics
        else:
            # 从所有可用指标中，根据索引选择
            self.selected_metrics = [all_base_metrics[i] for i in metric_indices if i < len(all_base_metrics)]
        
        print(f"[RepresentationMetricsCalculator] Initialized with selected metrics: {[name for name, _ in self.selected_metrics]}")

    def __call__(self, hidden_states, attention_mask, compute_diff=False, diff_stride=1):
        with torch.inference_mode():
            batch_size, seq_len, num_layers, hidden_dim = hidden_states.shape
            results = {}
            
            for layer_idx in range(num_layers):
                layer_key = str(layer_idx + 1)
                layer_hidden = hidden_states[:, :, layer_idx, :].contiguous()
                
                # 1. 照常计算所有的 sequence-level 指标
                base_metrics = {
                    name: func(layer_hidden, attention_mask)
                    for name, func in self.selected_metrics
                }
                
                per_stride_diffs = {}
                if compute_diff:
                    final_diffs, per_stride_diffs = self.calculate_metric_diff(layer_hidden, attention_mask, diff_stride)
                    base_metrics.update(final_diffs)
                
                if self.output_token_level_metrics:
                    # ### 修正点: 遍历字典条目的一个静态列表 ###
                    # 通过 list(base_metrics.items()) 创建一个副本进行遍历
                    for name, seq_level_tensor in list(base_metrics.items()):
                        # 避免为已经是 token-level 的指标再次创建
                        if name.endswith("_token_level"):
                            continue

                        token_level_key = f"{name}_token_level"
                        
                        if name in per_stride_diffs:
                            base_metrics[token_level_key] = self._distribute_value_by_scaling(
                                seq_level_tensor, per_stride_diffs[name], attention_mask, diff_stride
                            )
                        else:
                            base_metrics[token_level_key] = self._sequence_to_token_level(
                                seq_level_tensor, attention_mask
                            )
                
                results[layer_key] = base_metrics
                self._free_memory()
                
            return results

    def _distribute_value_by_scaling(self, seq_level_tensor, per_stride_values_list, attention_mask, stride):
        """
        Implements the user's "first assign, then scale" algorithm to distribute
        a sequence-level value to the token-level.
        """
        batch_size, seq_len = attention_mask.shape
        final_token_tensor = torch.zeros_like(attention_mask, dtype=torch.float32)

        for i in range(batch_size):
            target_sum_s = seq_level_tensor[i].item()
            stride_values_d = per_stride_values_list[i]
            
            if not stride_values_d:
                continue

            # 1. Create the temporary token-level tensor
            temp_token_tensor = torch.zeros(seq_len, device=attention_mask.device)
            valid_len = attention_mask[i].sum()
            num_strides = len(stride_values_d)

            for k in range(num_strides):
                start_idx = k * stride
                end_idx = min((k + 1) * stride, valid_len)
                temp_token_tensor[start_idx:end_idx] = stride_values_d[k]

            # 2. Calculate the temporary sum
            temporary_sum = temp_token_tensor.sum()

            # 3. Calculate the scaling factor, handling the edge case of sum being zero
            if abs(temporary_sum.item()) < self.epsilon:
                if valid_len > 0:
                    per_token_value = target_sum_s / valid_len
                    final_token_tensor[i, :valid_len] = per_token_value
                continue
            
            scaling_factor = target_sum_s / temporary_sum

            # 4. Apply the scaling to get the final tensor
            final_token_tensor[i] = temp_token_tensor * scaling_factor

        return final_token_tensor

    def _sequence_to_token_level(self, seq_level_tensor, attention_mask):
        """
        Converts a sequence-level metric tensor to a token-level one by
        smearing the value across valid tokens. Used for base metrics.
        """
        valid_lengths = attention_mask.sum(dim=1).float()
        valid_lengths = torch.clamp(valid_lengths, min=1)
        per_token_value = seq_level_tensor / valid_lengths
        token_level_tensor = per_token_value.unsqueeze(1).expand_as(attention_mask)
        token_level_tensor = token_level_tensor * attention_mask.float()
        return token_level_tensor

    def calculate_metric_diff(self, hidden_states, attention_mask, stride):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        metric_calculators = {
            "Response Entropy 1": lambda h: self._single_entropy(h, 1, "gram"),
            "Effective Rank": lambda h: self._single_effective_rank(h, log_output=False),
            "Curvature": lambda h: self._single_curvature(h),
            "Log Effective Rank": lambda h: self._single_effective_rank(h, log_output=True)
        }

        selected_metric_names = [name for name, _ in self.selected_metrics]
        selected_calculators = [metric_calculators[name] for name in selected_metric_names]
        num_metrics_to_track = len(selected_metric_names)

        final_diffs = {}      # 存放最终平均值
        per_stride_diffs = {} # 存放每个样本的stride值列表

        for name in selected_metric_names:
            final_diffs[f"{name} diff"] = torch.zeros(batch_size, device=device)
            final_diffs[f"{name} diff 2"] = torch.zeros(batch_size, device=device)
            per_stride_diffs[f"{name} diff"] = [[] for _ in range(batch_size)]
            per_stride_diffs[f"{name} diff 2"] = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]
            valid_len = valid_hidden.size(0)
            
            if valid_len < 2:
                continue
            if valid_len > self.max_seq_len:
                valid_hidden = valid_hidden[-self.max_seq_len:]
                valid_len = self.max_seq_len
                
            history_sum = [0.0] * num_metrics_to_track
            history_count = 0
            prev_diff = None
            
            for t in range(1, valid_len):
                if t % stride != 0:
                    continue
                
                window_start = max(0, t - self.max_seq_len + 1)
                sub_hidden = valid_hidden[window_start:t+1]
                current_metrics = [calc(sub_hidden) for calc in selected_calculators]

                if history_count > 0:
                    hist_avg = [s / history_count for s in history_sum]
                    curr_diff = [(curr - avg) for curr, avg in zip(current_metrics, hist_avg)]
                    
                    # 记录每个stride的diff值
                    for idx, name in enumerate(selected_metric_names):
                        per_stride_diffs[f"{name} diff"][i].append(curr_diff[idx])
                    
                    if prev_diff is not None:
                        curr_diff2 = [(cd - pd) for cd, pd in zip(curr_diff, prev_diff)]
                        for idx, name in enumerate(selected_metric_names):
                            per_stride_diffs[f"{name} diff 2"][i].append(curr_diff2[idx])
                    
                    prev_diff = curr_diff
                
                history_sum = [s + curr for s, curr in zip(history_sum, current_metrics)]
                history_count += 1
                
        # 计算最终的平均值
        for i in range(batch_size):
            for name in selected_metric_names:
                diff_key = f"{name} diff"
                if per_stride_diffs[diff_key][i]:
                    final_diffs[diff_key][i] = torch.tensor(per_stride_diffs[diff_key][i]).mean()
                
                diff2_key = f"{name} diff 2"
                if len(per_stride_diffs[diff2_key][i]) > 0:
                    final_diffs[diff2_key][i] = torch.tensor(per_stride_diffs[diff2_key][i]).mean()

        return final_diffs, per_stride_diffs

    def _single_entropy(self, hidden: torch.Tensor, alpha: float = 1.0001, matrix_type: str = 'gram') -> float:
        """
        Calculate Renyi entropy using either covariance or Gram matrix for a single hidden state sequence.

        Args:
            hidden (torch.Tensor): A single sequence of hidden states (seq_len, hidden_dim).
            alpha (float): The alpha parameter for Renyi entropy. Defaults to 1.0001 (approximates Shannon entropy).
            matrix_type (str): Type of matrix to use, 'covariance' or 'gram'. Defaults to 'gram'.

        Returns:
            float: The calculated Renyi entropy.
        """
        assert matrix_type in ['covariance', 'gram'], "matrix_type must be 'covariance' or 'gram'"
        
        if hidden.size(0) < 2: # Need at least 2 tokens to form a matrix
            return 0.0

        try:
            with torch.amp.autocast(device_type='cuda'): # Use mixed precision for potential speedup
                # Center the data (critical for both methods to remove mean effect)
                centered = hidden - hidden.mean(dim=0, keepdim=True)
                
                # Build the target matrix (covariance or Gram)
                if matrix_type == 'covariance':
                    # Covariance matrix: [hidden_dim, hidden_dim]
                    matrix = centered.T @ centered / (centered.size(0) - 1)
                else:
                    # Gram matrix: [seq_len, seq_len]
                    matrix = centered @ centered.T 
                
                # Compute eigenvalues (symmetric matrix, so use eigvalsh for efficiency and stability)
                matrix = matrix.to(torch.float32)
                eigvals = torch.linalg.eigvalsh(matrix)  # Ensure numerical stability
                
                # Filter out very small eigenvalues for numerical stability
                eigvals = eigvals[eigvals > 1e-8]
                
                if len(eigvals) == 0: # No significant eigenvalues
                    return 0.0
                    
                # Normalize eigenvalues to sum to 1
                normalized = eigvals / eigvals.sum()
                
                # Compute Renyi entropy based on alpha
                if abs(alpha - 1.0) < 1e-6: # Case for Shannon entropy (alpha -> 1)
                    normalized = normalized[normalized > 1e-12] # Further safety for log(0)
                    return -torch.sum(normalized * torch.log(normalized)).item()
                else: # General Renyi entropy formula
                    return (1/(1-alpha)) * torch.log(torch.sum(normalized**alpha)).item()
        except torch._C._LinAlgError as e:
            # 捕获线性代数错误，打印警告并返回一个安全值，而不是让整个程序崩溃
            # print(f"\n[WARNING] linalg.eigh failed to converge. Returning 0.0 for this sample. Error: {e}")
            return 0.0

    def _single_effective_rank(self, hidden: torch.Tensor, log_output: bool = False) -> float:
        """
        Calculates the effective rank for a single hidden state sequence using low-rank SVD.

        Args:
            hidden (torch.Tensor): A single sequence of hidden states (seq_len, hidden_dim).
            log_output (bool): If True, returns the natural logarithm of the effective rank.
                               Defaults to False.

        Returns:
            float: The calculated effective rank or its natural logarithm.
        """
        if hidden.size(0) < 2: # Need at least 2 tokens for SVD
            return 0.0

        try:
            with torch.amp.autocast(device_type='cuda'):

                centered = hidden - hidden.mean(dim=0, keepdim=True) # 加一行这个和entropy保持一致性

                centered = centered.to( torch.float32)
                _, S, _ = torch.svd_lowrank(centered, q=min(self.svd_rank, centered.size(1)))
                
                normalized_S = S / (S.sum() + 1e-8)
                
                if log_output:
                    return -torch.sum(normalized_S * torch.log(normalized_S + 1e-8)).item()
                else:
                    return torch.exp(-torch.sum(normalized_S * torch.log(normalized_S + 1e-8))).item()
        except torch._C._LinAlgError as e:
            # print(f"[WARNING] SVD failed in _single_effective_rank. Returning 0.0. Error: {e}")
            return 0.0

    def _single_curvature(self, hidden: torch.Tensor) -> float:
        """
        Calculates the average curvature for a single hidden state sequence using cosine similarity of differences.

        Args:
            hidden (torch.Tensor): A single sequence of hidden states (seq_len, hidden_dim).

        Returns:
            float: The average curvature.
        """
        if hidden.size(0) < 3: # Need at least 3 tokens to define two difference vectors
            return 0.0
            
        # Compute differences between consecutive hidden states
        diffs = hidden[1:] - hidden[:-1] # [seq_len-1, hidden_dim]
        angles = []
        
        # Process in chunks to avoid large intermediate tensors, if diffs is very long
        chunk_size = 256
        for chunk in torch.split(diffs, chunk_size, dim=0):
            if chunk.size(0) < 2: # Need at least 2 difference vectors in a chunk
                continue
                
            # Calculate norms for normalization in cosine similarity
            norms = torch.norm(chunk, dim=1, keepdim=True)
            # Identify valid vectors (non-zero norm)
            valid = (norms > 1e-6).squeeze()
            chunk = chunk[valid] # Filter out near-zero difference vectors
            
            if chunk.size(0) < 2:
                continue
                
            # Compute cosine similarity between consecutive difference vectors
            cos_sim = F.cosine_similarity(chunk[:-1], chunk[1:], dim=1)
            # Clamp for numerical stability to ensure arccos input is in [-1, 1]
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            # Calculate angle from cosine similarity
            angles.append(torch.arccos(cos_sim))
            
        if angles:
            # Concatenate all angles and return their mean
            return torch.cat(angles).mean().item()
        return 0.0 # Return 0 if no valid angles could be computed

    def _free_tensors(self, tensors):
        """
        Explicitly frees a list of PyTorch tensors from memory.

        Args:
            tensors (list): A list of torch.Tensor objects to be deleted.
        """
        for t in tensors:
            if isinstance(t, torch.Tensor):
                del t
        # Clear CUDA cache to release GPU memory (if available)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _free_memory(self):
        """
        Clears the internal cache and explicitly frees memory.
        This is called periodically to manage memory usage.
        """
        self._cached_tensors.clear() # Clear the cache of intermediate results
        self._free_tensors([]) # Call _free_tensors with an empty list to just clear CUDA cache
    
    def calculate_response_entropy(self, 
                                hidden_states: torch.Tensor, 
                                attention_mask: torch.Tensor, 
                                alpha: float = 1.0001,
                                matrix_type: str = 'covariance') -> torch.Tensor:
        """
        Calculates Renyi entropy for each sample in a batch.

        Args:
            hidden_states (torch.Tensor): Hidden states for a single layer (batch_size, seq_len, hidden_dim).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).
            alpha (float): The alpha parameter for Renyi entropy. Defaults to 1.0001.
            matrix_type (str): Type of matrix to use, 'covariance' or 'gram'. Defaults to 'covariance'.

        Returns:
            torch.Tensor: A tensor of Renyi entropies for each sample in the batch.
        """
        assert matrix_type in ['covariance', 'gram'], "matrix_type must be 'covariance' or 'gram'"
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        entropies = torch.zeros(batch_size, device=hidden_states.device)
        
        for i in range(batch_size):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]  # Extract non-padding tokens
            entropies[i] = self._single_entropy(valid_hidden, alpha, matrix_type)
            
        return entropies
    
    def calculate_effective_rank(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, log_output: bool = False) -> torch.Tensor:
        """
        Calculates effective rank for each sample in a batch.

        Args:
            hidden_states (torch.Tensor): Hidden states for a single layer (batch_size, seq_len, hidden_dim).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).
            log_output (bool): If True, returns the natural logarithm of the effective rank.
                               Defaults to False.

        Returns:
            torch.Tensor: A tensor of effective ranks (or their logs) for each sample in the batch.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        ranks = torch.zeros(batch_size, device=hidden_states.device)
        
        for i in range(batch_size):
            # Get non-padding tokens
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]  # [valid_seq_len, hidden_dim]
            
            if valid_hidden.shape[0] == 0: # Handle empty sequences
                ranks[i] = 0.0
                continue
                
            try:
                U, S, Vh = torch.linalg.svd(valid_hidden, full_matrices=False)
                
                normalized_S = S / (S.sum() + 1e-8)
                
                if log_output:
                    ranks[i] = -torch.sum(normalized_S * torch.log(normalized_S + 1e-8))
                else:
                    ranks[i] = torch.exp(-torch.sum(normalized_S * torch.log(normalized_S + 1e-8)))
            except torch._C._LinAlgError as e:
                # print(f"[WARNING] SVD failed for a sample in calculate_effective_rank. Setting rank to 0.0. Error: {e}")
                ranks[i] = 0.0 # 为失败的样本返回安全值
            
        return ranks
    
    def calculate_curvature(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates average curvature for each sample in a batch.

        Args:
            hidden_states (torch.Tensor): Hidden states for a single layer (batch_size, seq_len, hidden_dim).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).

        Returns:
            torch.Tensor: A tensor of average curvatures for each sample in the batch.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        curvatures = torch.zeros(batch_size, device=hidden_states.device)
        
        for i in range(batch_size):
            # Get non-padding tokens
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]  # [valid_seq_len, hidden_dim]
            
            if valid_hidden.shape[0] < 3:  # Need at least 3 tokens to compute curvature
                curvatures[i] = 0.0
                continue
                
            # Compute differences between consecutive tokens
            diffs = valid_hidden[1:] - valid_hidden[:-1]  # [valid_seq_len-1, hidden_dim]
            
            # Compute angles between consecutive differences
            angles = []
            for k in range(diffs.shape[0]-1):
                v_k = diffs[k]
                v_k1 = diffs[k+1]
                
                # Handle zero vectors to avoid division by zero
                norm_v_k = torch.norm(v_k)
                norm_v_k1 = torch.norm(v_k1)

                if norm_v_k < 1e-8 or norm_v_k1 < 1e-8:
                    angle = 0.0
                else:
                    cos_theta = torch.dot(v_k, v_k1) / (norm_v_k * norm_v_k1)
                    # Clamp for numerical stability to ensure arccos input is in [-1, 1]
                    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                    angle = torch.arccos(cos_theta)
                
                angles.append(angle)
            
            if len(angles) == 0: # If no valid angles were computed
                curvatures[i] = 0.0
            else:
                curvatures[i] = torch.mean(torch.stack(angles)) # Mean of all computed angles
                
        return curvatures


import ray
import hydra

# This tells Hydra to use this function as the entry point and 
# to load the configuration from the config/ppo_trainer.yaml file (or a similar file).
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config, compute_score=_custom_compute_score)


def run_ppo(config, compute_score=None):

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote
def main_task(config, compute_score=None):
    '''
    This is the core function that performs the PPO training.
    '''
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # Purpose: This dictionary maps each role in the training process (e.g., ActorRollout, Critic, RewardModel) 
    # to the Ray worker class responsible for performing that role's computations.
    # Key: A Role enum member (e.g., Role.ActorRollout).
    # Value: A Ray remote class (e.g., ray.remote(ActorRolloutRefWorker)). 
    # This is the class that will be instantiated on the Ray cluster to perform the computations for that role.
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    calculator = RepresentationMetricsCalculator(tokenizer=tokenizer, 
                                                 compute_log_effective_rank=config.calculator.compute_log_effective_rank,
                                                 metric_indices=config.calculator.get('metric_indices', None),
                                                 output_token_level_metrics=config.calculator.output_token_level_metrics,
                                                 )


    # <<< 修改 RewardManager 的实例化过程 >>>
    # Pass parameters from the hydra config to the RewardManager
    reward_fn = RewardManager(tokenizer=tokenizer, 
                              num_examine=0, 
                              compute_score=compute_score, 
                              calculator=calculator,
                              ema_alpha=config.reward_manager.ema_alpha,
                              indicator_names=config.reward_manager.indicator_names,
                              weights=config.reward_manager.weights,
                              weights_exploit=config.reward_manager.weights_exploit,
                              calculator_enabled=config.calculator.enable,
                              add_reward=config.reward_manager.add_reward,
                              modulation_gain=config.reward_manager.modulation_gain,
                              adv_estimator=config.algorithm.adv_estimator,
                              output_token_level_metrics=config.calculator.output_token_level_metrics,
                              aux_reward_global_weight=config.reward_manager.aux_reward_global_weight,
                              token_level_baseline_type=config.reward_manager.token_level_baseline_type,
                            )
    
    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, 
                                  num_examine=1, 
                                  compute_score=None, 
                                  calculator=calculator,
                                  ema_alpha=config.reward_manager.ema_alpha,
                                  indicator_names=config.reward_manager.indicator_names,
                                  weights=config.reward_manager.weights,
                                  weights_exploit=config.reward_manager.weights_exploit,
                                  calculator_enabled=config.calculator.enable,
                                  add_reward=config.reward_manager.add_reward,
                                  modulation_gain=config.reward_manager.modulation_gain,
                                    adv_estimator=config.algorithm.adv_estimator,
                                    output_token_level_metrics=config.calculator.output_token_level_metrics,
                                    aux_reward_global_weight=config.reward_manager.aux_reward_global_weight,
                                    token_level_baseline_type=config.reward_manager.token_level_baseline_type,
                                  )
    # <<< 修改结束 >>>

    # reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, calculator=calculator)

    # # Note that we always use function-based RM for validation
    # val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, compute_score=None, calculator=calculator)

    

    # Purpose: This class manages the resource pools available on the Ray cluster and assigns roles to specific resource pools.
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            calculator=calculator)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()

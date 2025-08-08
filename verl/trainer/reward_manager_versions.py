# verl/trainer/reward_manager_versions.py
import os
import ray
from ray.util.actor_pool import ActorPool
from .metrics_calculator import RepresentationMetricsCalculator, RepresentationMetricsCalculator_parallel
import time

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
import numpy as np

# 假设 _default_compute_score 在此文件中可访问或已导入
# 如果它在 main_ppo.py 中，您需要将它也移动到这里或另一个公共的工具文件中
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


def _calculate_reward_for_single_sample(
    # 输入参数
    data_item, index, tokenizer, compute_score, use_aux_reward, indicator_names, 
    mids, weights_explore, weights_exploit, modulation_gain, epsilon, 
    performance_scaling_factor, act_func, adv_estimator, output_token_level_metrics,
    token_level_baseline_type
    ):
    """
    为单个样本计算核心奖励和辅助奖励。

    返回:
        一个包含该样本所有计算结果的字典。
    """
    layer_key = '1'
    prompt_ids, prompt_length = data_item.batch['prompts'], data_item.batch['prompts'].shape[-1]
    valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
    response_ids = data_item.batch['responses']
    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    sequences_str = tokenizer.decode(sequences)
    ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
    data_source = data_item.non_tensor_batch['data_source']
    
    score_dict = compute_score(data_source=data_source, solution_str=sequences_str, ground_truth=ground_truth)
    reward_value = score_dict['score']
    correctness = score_dict['correctness']
    internal_metrics_sample = {}
    
    # 初始化一个空的 token-level 奖励张量
    aux_reward_per_token = torch.zeros(valid_response_length, device=data_item.batch.device, dtype=torch.bfloat16)

    if use_aux_reward:
        guidance_indicator_name = indicator_names[0]
        current_guidance_value = data_item.batch['calculator_results'][layer_key][guidance_indicator_name]
        ema_baseline = mids[guidance_indicator_name]
        percentage_deviation = (current_guidance_value - ema_baseline) / (abs(ema_baseline) + epsilon)
        percentage_deviation = torch.clamp(percentage_deviation, -5.0, 5.0)
        
        w_explore = torch.tensor(weights_explore, device=data_item.batch.device)
        w_exploit = torch.tensor(weights_exploit, device=data_item.batch.device)
        exploit_tendency = torch.sigmoid(modulation_gain * percentage_deviation)
        dynamic_weights = (1.0 - exploit_tendency) * w_explore + exploit_tendency * w_exploit
        weights_map = {name: weight for name, weight in zip(indicator_names, dynamic_weights)}
        
        internal_metrics_sample['percentage_deviation'] = percentage_deviation.item()
        internal_metrics_sample['exploit_tendency'] = exploit_tendency.item()
        for name, weight in weights_map.items():
            internal_metrics_sample[f"weight_{name.replace(' ', '_').lower()}"] = weight.item()

        # Case 1: GAE with token-level metrics (dense reward)
        if adv_estimator == 'gae' and output_token_level_metrics:
            for indicator_name in indicator_names:
                token_level_indicator = data_item.batch['calculator_results'][layer_key][f"{indicator_name}_token_level"]
                # 注意：这里需要根据响应长度来切片
                valid_token_level_indicator = token_level_indicator[prompt_length : prompt_length + valid_response_length]

                baseline = 0.0
                if token_level_baseline_type == 'internal_mean':
                    baseline = torch.mean(valid_token_level_indicator)
                elif token_level_baseline_type == 'external_ema':
                    baseline = mids[indicator_name]
                else:
                    raise ValueError(f"Invalid token_level_baseline_type: {token_level_baseline_type}")

                relative_deviation_tensor = (valid_token_level_indicator - baseline) / (torch.abs(baseline) + epsilon)
                relative_deviation_tensor = torch.clamp(relative_deviation_tensor, -5.0, 5.0)
                
                log_name = f"relative_deviation_{indicator_name.replace(' ', '_').lower()}"
                internal_metrics_sample[log_name] = relative_deviation_tensor.mean().item()
                
                aux_reward_per_token += act_func(relative_deviation_tensor) * weights_map[indicator_name]
            
            aux_reward_per_token *= performance_scaling_factor

        # Case 2: All other cases (GRPO or sequence-level metrics) (sparse reward)
        else:
            calculator_tensor_i = 0.0
            for indicator_name in indicator_names:
                original_indicator = data_item.batch['calculator_results'][layer_key][indicator_name]
                relative_deviation = (original_indicator - mids[indicator_name]) / (abs(mids[indicator_name]) + epsilon)
                relative_deviation = torch.clamp(relative_deviation, -5.0, 5.0)
                
                log_name = f"relative_deviation_{indicator_name.replace(' ', '_').lower()}"
                internal_metrics_sample[log_name] = relative_deviation.item()
                
                calculator_tensor_i += act_func(relative_deviation) * weights_map[indicator_name]
            
            final_aux_reward = calculator_tensor_i * performance_scaling_factor
            # 将稀疏奖励施加在最后一个 token 上
            if valid_response_length > 0:
                aux_reward_per_token[-1] += final_aux_reward

    return {
        'index': index,
        'reward_0_value': score_dict['score'],
        'final_reward_value': reward_value, # 原始奖励，辅助奖励将在外部添加
        'aux_reward_per_token': aux_reward_per_token, # Token-level 辅助奖励
        'correctness': correctness,
        'valid_response_length': valid_response_length.item(),
        'internal_metrics_sample': internal_metrics_sample,
        'data_source': data_source,
        'sequences_str': sequences_str
    }



# ==============================================================================
#           用于并行计算的 Ray Actor (修正版)
# ==============================================================================
@ray.remote
class RewardCalculatorActor:
    """
    一个多功能的 Ray Actor，经过优化，可以批量处理计算任务。
    """
    # --- 修复点 (1/4): 在 __init__ 中接收完整的配置 ---
    def __init__(self, tokenizer, num_examine, compute_score, adv_estimator, output_token_level_metrics, token_level_baseline_type):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        # 接收并保存配置，不再使用硬编码的假设值
        self.adv_estimator = adv_estimator
        self.output_token_level_metrics = output_token_level_metrics
        self.token_level_baseline_type = token_level_baseline_type

    # --- 隐藏优化 (2/4): 修改方法以处理一批样本 (chunk) ---
    def process_sample_chunk(self, chunk_of_args):
        """
        不再处理单个样本，而是接收一个 "任务块" (一个列表)，
        并在 Actor 内部循环处理，从而大幅减少通信开销。
        """
        results_chunk = []
        # 在 Actor 内部进行循环
        for args in chunk_of_args:

            (data_item, index, use_aux_reward, indicator_names, mids, weights_explore,
             weights_exploit, modulation_gain, epsilon, performance_scaling_factor, act_func) = args
            

            single_result = _calculate_reward_for_single_sample(
                data_item=data_item,
                index=index,
                tokenizer=self.tokenizer,
                compute_score=self.compute_score,
                use_aux_reward=use_aux_reward,
                indicator_names=indicator_names,
                mids=mids,
                weights_explore=weights_explore,
                weights_exploit=weights_exploit,
                modulation_gain=modulation_gain,
                epsilon=epsilon,
                performance_scaling_factor=performance_scaling_factor,
                act_func=act_func,
                adv_estimator=self.adv_estimator,
                output_token_level_metrics=self.output_token_level_metrics,
                token_level_baseline_type=self.token_level_baseline_type
            )

            results_chunk.append(single_result)
        
        # 返回整个块的结果
        return results_chunk


# ==============================================================================
#           版本A: 原始串行 (Sequential) 的 RewardManager
# ==============================================================================

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


# ==============================================================================
#           版本B: 并行 (Parallel) 的 RewardManager (完整版)
# ==============================================================================
class RewardManager_parallel():
    """版本B：并行的 RewardManager，使用 Ray Actor Pool 加速。"""
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
        self.ema_performance_score = 0.0 
        self.aux_reward_global_weight = aux_reward_global_weight
        self.adv_estimator = adv_estimator
        self.output_token_level_metrics = output_token_level_metrics
        self.token_level_baseline_type = token_level_baseline_type
        print(f"[RewardManager] Initialized: PARALLEL version.")
        print(f" -> Token-level baseline type: {self.token_level_baseline_type}")

        available_cpus = ray.available_resources().get("CPU", 1)
        self.num_actors = min(4, int(available_cpus - 1))
        print(f" -> Ray reports {available_cpus} CPUs available. Creating Actor Pool with {self.num_actors} actors.")
        
        actors = [RewardCalculatorActor.remote(
            self.tokenizer, 
            self.num_examine, 
            self.compute_score, 
            self.adv_estimator, 
            self.output_token_level_metrics,
            self.token_level_baseline_type # 传递新参数
            ) for _ in range(self.num_actors)]
        self.actor_pool = ActorPool(actors)


    @staticmethod
    def _split_list_into_chunks(data_list: list, n: int) -> list:
        if not data_list: return []
        k, m = divmod(len(data_list), n)
        return [data_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

    def __call__(self, data: DataProto, is_val=False, metrics_old=None, global_step=None):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor_0 = torch.zeros_like(data.batch['responses'], dtype=torch.bfloat16)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.bfloat16)
        correctness_tensor = torch.zeros(len(data), dtype=torch.bfloat16)
        
        internal_metrics = { 'percentage_deviation': [], 'exploit_tendency': [] }
        already_print_data_sources = {}

        use_aux_reward = self.add_reward and self.calculator_enabled and metrics_old
        performance_scaling_factor = 1.0
        act_func = nn.Tanh() if use_aux_reward else None

        if use_aux_reward:
            normalized_performance = (self.ema_performance_score + 1.0) / 2.0
            performance_scaling_factor = self.aux_reward_global_weight * (1.0 - normalized_performance)
            internal_metrics['performance_scaling_factor'] = [performance_scaling_factor]

        all_tasks_args = []
        for i in range(len(data)):
            # **注意**: 之前的 NotImplementedError 逻辑已经被移入共享函数中，
            # 因此并行版本现在自然地支持了 GAE token-level 奖励。
            task_args = (
                data[i], i, use_aux_reward, self.indicator_names, self.mids,
                self.weights_explore, self.weights_exploit, self.modulation_gain,
                self.epsilon, performance_scaling_factor, act_func
            )
            all_tasks_args.append(task_args)
        
        task_chunks = self._split_list_into_chunks(all_tasks_args, self.num_actors)
        
        results_chunks = self.actor_pool.map_unordered(
            lambda actor, chunk: actor.process_sample_chunk.remote(chunk),
            [chunk for chunk in task_chunks if len(chunk) > 0]
        )

        for result_chunk in results_chunks:
            for result in result_chunk:
                i = result['index']
                valid_len = result['valid_response_length']
                
                if valid_len > 0:
                    reward_tensor_0[i, valid_len - 1] = result['reward_0_value']
                    
                    # 同样，最终奖励 = 基础奖励 + 辅助奖励
                    reward_tensor[i, valid_len - 1] = result['final_reward_value']
                    reward_tensor[i, :valid_len] += result['aux_reward_per_token']

                correctness_tensor[i] = result['correctness']

                for key, value in result['internal_metrics_sample'].items():
                    if key not in internal_metrics: internal_metrics[key] = []
                    internal_metrics[key].append(value)
                
                data_source = result.get('data_source', 'unknown')
                sequences_str = result.get('sequences_str', '')
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(sequences_str)

        if use_aux_reward and not is_val:
            self.ema_performance_score = (1 - self.ema_alpha) * self.ema_performance_score + \
                                                self.ema_alpha * reward_tensor_0.sum(dim=-1).float().mean().cpu().item()
            layer_key = '1'
            for indicator_name in self.indicator_names:
                metric_key = f'cal/overall/layer_{layer_key}/{indicator_name}/mean'
                if metric_key in metrics_old:
                    v = metrics_old[metric_key]
                    self.mids[indicator_name] = (1 - self.ema_alpha) * self.mids[indicator_name] + self.ema_alpha * v
  
        return {"reward_tensor": reward_tensor, 
                "correctness_tensor": correctness_tensor, 
                "reward_tensor_0": reward_tensor_0,
                "internal_metrics": internal_metrics}
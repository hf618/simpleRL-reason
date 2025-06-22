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
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, calculator=None, ema_alpha=0.7) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.calculator = calculator
        self.ema_alpha = ema_alpha
        self.indicator_names = ['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']
        self.weights = [1, 1 / 4, 1 / 16]  # 权重可以根据需要调整
        self.weights_inner = [1.0, 1e-1, 1e-5]  # 激活函数内部指标的权重
        self.mids = {name: 0.0 for name in self.indicator_names}

    def __call__(self, data: DataProto, is_val=False, metrics_old=None):
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

        # sigmoid = nn.Sigmoid()
        layer_key = '1'
        add_reward = True  #    ****************************************** 这儿记得修改***********************************************
        if add_reward:
            act_func = nn.Tanh()
            for i in range(len(self.indicator_names)):
                indicator_name = self.indicator_names[i]
                if metrics_old: # 如果字典不为空
                    v = metrics_old[f'cal/overall/layer_{layer_key}/{indicator_name}/mean']
                    self.mids[indicator_name] = ( 1 - self.ema_alpha ) * self.mids[indicator_name] +  self.ema_alpha * v

        
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

            score_dict = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            reward_tensor_0[i, valid_response_length - 1] = score_dict['score']
            correctness_tensor[i] = score_dict['correctness']

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)    

            # 算一个辅助的 reward
            if add_reward:
                calculator_tensor[i] = 0.0  # 显式清零
                for j in range(len(self.indicator_names)):
                    indicator_name = self.indicator_names[j]
                    original_indicator = data_item.batch['calculator_results'][layer_key][indicator_name] 
                    gap = original_indicator - self.mids[indicator_name]
                    calculator_tensor[i]  += act_func(gap * self.weights_inner[j]) * self.weights[j]

                # calculator_tensor[i]  = act_func(torch.log(original_indicator) - self.mid)
                # * correctness_tensor[i] 表示ne_diff_2_1
                reward_tensor[i, valid_response_length - 1] = reward_tensor_0[i, valid_response_length - 1] + calculator_tensor[i]
            else:
                reward_tensor[i, valid_response_length - 1] = score_dict['score']


        return {"reward_tensor": reward_tensor, "correctness_tensor": correctness_tensor, "reward_tensor_0": reward_tensor_0}

class RepresentationMetricsCalculator():
    """Calculates representation quality metrics from hidden states with memory optimization."""
    
    def __init__(self, tokenizer, max_seq_len=512, svd_rank=6):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len  # 控制处理的最大序列长度
        self.svd_rank = svd_rank        # SVD保留的奇异值数量
        self._cached_tensors = {}       # 重用中间结果的缓存

    def __call__(self, hidden_states, attention_mask, compute_diff=False, diff_stride=1):
        with torch.inference_mode():  # 禁用梯度且优化内存
            batch_size, seq_len, num_layers, hidden_dim = hidden_states.shape
            results = {}
            
            for layer_idx in range(num_layers):
                layer_key = str(layer_idx + 1)
                layer_hidden = hidden_states[:, :, layer_idx, :].contiguous()
                
                # 基础指标计算
                base_metrics = {
                    "Response Entropy 1": self.calculate_response_entropy(layer_hidden, attention_mask, 1, "gram"),
                    "Effective Rank": self.calculate_effective_rank(layer_hidden, attention_mask),
                    "Curvature": self.calculate_curvature(layer_hidden, attention_mask)
                }
                
                if compute_diff:
                    diff_metrics = self.calculate_metric_diff(layer_hidden, attention_mask, diff_stride)
                    base_metrics.update(diff_metrics)
                
                results[layer_key] = base_metrics
                self._free_memory()  # 显式释放内存
                
            return results

    def calculate_metric_diff(self, hidden_states, attention_mask, stride):
        """滑动窗口计算指标差异 (显存优化版)"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        diffs = {
            "Response Entropy 1 diff": torch.zeros(batch_size, device=device),
            "Effective Rank diff": torch.zeros(batch_size, device=device),
            "Curvature diff": torch.zeros(batch_size, device=device),
            "Response Entropy 1 diff 2": torch.zeros(batch_size, device=device),
            "Effective Rank diff 2": torch.zeros(batch_size, device=device),
            "Curvature diff 2": torch.zeros(batch_size, device=device)
        }
        
        for i in range(batch_size):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]
            valid_len = valid_hidden.size(0)
            
            if valid_len < 2:
                continue
                
            if valid_len > self.max_seq_len:
                valid_hidden = valid_hidden[-self.max_seq_len:]
                valid_len = self.max_seq_len
                
            # 历史累积数据
            history_sum = [0.0, 0.0, 0.0]  # [entropy, rank, curvature]
            history_count = 0
            total_diff = [0.0, 0.0, 0.0]
            total_diff2 = [0.0, 0.0, 0.0]
            valid_diff_count = 0
            prev_diff = None
            
            for t in range(1, valid_len):
                if t % stride != 0:
                    continue
                
                window_start = max(0, t - self.max_seq_len + 1)
                sub_hidden = valid_hidden[window_start:t+1]
                
                cache_key = f"{i}_{t}"
                if cache_key in self._cached_tensors:
                    current_metrics = self._cached_tensors[cache_key]
                else:
                    current_metrics = (
                        self._single_entropy(sub_hidden, 1, "gram"),
                        self._single_effective_rank(sub_hidden),
                        self._single_curvature(sub_hidden)
                    )
                    self._cached_tensors[cache_key] = current_metrics
                
                if history_count > 0:
                    # 计算历史均值
                    hist_avg = [
                        history_sum[0] / history_count,
                        history_sum[1] / history_count,
                        history_sum[2] / history_count
                    ]
                    
                    # 计算当前差异
                    curr_diff = [
                        (curr - avg) for curr, avg in zip(current_metrics, hist_avg)
                    ]
                    
                    # 累加一阶差异
                    total_diff = [sum_ + d for sum_, d in zip(total_diff, curr_diff)]
                    
                    # 计算二阶差异
                    if prev_diff is not None:
                        curr_diff2 = [
                            (curr_d - prev_d) for curr_d, prev_d in zip(curr_diff, prev_diff)
                        ]
                        total_diff2 = [sum_ + d2 for sum_, d2 in zip(total_diff2, curr_diff2)]
                    
                    prev_diff = curr_diff
                    valid_diff_count += 1
                
                # 更新历史累积
                history_sum = [sum_ + curr for sum_, curr in zip(history_sum, current_metrics)]
                history_count += 1
                
                self._free_memory()
                
            if valid_diff_count > 0:
                # 计算一阶差异均值
                avg_diff = [t / valid_diff_count for t in total_diff]
                diffs["Response Entropy 1 diff"][i] = avg_diff[0]
                diffs["Effective Rank diff"][i] = avg_diff[1]
                diffs["Curvature diff"][i] = avg_diff[2]
                
                # 计算二阶差异均值
                if valid_diff_count > 1:
                    avg_diff2 = [t / (valid_diff_count - 1) for t in total_diff2]
                    diffs["Response Entropy 1 diff 2"][i] = avg_diff2[0]
                    diffs["Effective Rank diff 2"][i] = avg_diff2[1]
                    diffs["Curvature diff 2"][i] = avg_diff2[2]
        
        return diffs

    def _single_entropy(self, hidden: torch.Tensor, alpha: float = 1.0001, matrix_type: str = 'gram') -> float:
        """Calculate Renyi entropy using either covariance or Gram matrix."""
        assert matrix_type in ['covariance', 'gram'], "matrix_type must be 'covariance' or 'gram'"
        
        if hidden.size(0) < 2:
            return 0.0

        with torch.amp.autocast(device_type='cuda'):
            # Center the data (critical for both methods)
            centered = hidden - hidden.mean(dim=0, keepdim=True)
            
            # Build the target matrix
            if matrix_type == 'covariance':
                matrix = centered.T @ centered / (centered.size(0) - 1)  # [hidden_dim, hidden_dim]
            else:
                matrix = centered @ centered.T  # [seq_len, seq_len]
            
            # Unified eigenvalue computation
            eigvals = torch.linalg.eigvalsh(matrix)
            eigvals = eigvals[eigvals > 1e-8]  # Unified threshold
            
            if len(eigvals) == 0:
                return 0.0
                
            # Normalize and compute entropy
            normalized = eigvals / eigvals.sum()
            if abs(alpha - 1.0) < 1e-6:
                normalized = normalized[normalized > 1e-12]  # Extra safety for log
                return -torch.sum(normalized * torch.log(normalized)).item()
            else:
                return (1/(1-alpha)) * torch.log(torch.sum(normalized**alpha)).item()

    def _single_effective_rank(self, hidden):
        """经济型SVD计算"""
        if hidden.size(0) < 2:
            return 0.0
            
        with torch.amp.autocast(device_type='cuda'):
            _, S, _ = torch.svd_lowrank(hidden, q=min(self.svd_rank, hidden.size(1)))
            normalized_S = S / (S.sum() + 1e-8)
            effective_rank = torch.exp(-torch.sum(normalized_S * torch.log(normalized_S + 1e-8)))
            
        self._free_tensors([S, normalized_S])
        return effective_rank.item()

    def _single_curvature(self, hidden):
        """分块角度计算"""
        if hidden.size(0) < 3:
            return 0.0
            
        diffs = hidden[1:] - hidden[:-1]
        angles = []
        
        # 分块处理避免大矩阵
        chunk_size = 256
        for chunk in torch.split(diffs, chunk_size, dim=0):
            if chunk.size(0) < 2:
                continue
                
            norms = torch.norm(chunk, dim=1, keepdim=True)
            valid = (norms > 1e-6).squeeze()
            chunk = chunk[valid]
            
            if chunk.size(0) < 2:
                continue
                
            cos_sim = F.cosine_similarity(chunk[:-1], chunk[1:], dim=1)
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            angles.append(torch.arccos(cos_sim))
            
        if angles:
            return torch.cat(angles).mean().item()
        return 0.0

    def _free_tensors(self, tensors):
        """显式释放张量"""
        for t in tensors:
            if isinstance(t, torch.Tensor):
                del t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _free_memory(self):
        """定期清理缓存"""
        self._cached_tensors.clear()
        self._free_tensors([])
    
    def calculate_response_entropy(self, 
                                hidden_states: torch.Tensor, 
                                attention_mask: torch.Tensor, 
                                alpha: float = 1.0001,
                                matrix_type: str = 'covariance') -> torch.Tensor:
        """Batch version with unified matrix selection."""
        assert matrix_type in ['covariance', 'gram'], "matrix_type must be 'covariance' or 'gram'"
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        entropies = torch.zeros(batch_size, device=hidden_states.device)
        
        for i in range(batch_size):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]  # [valid_seq_len, hidden_dim]
            entropies[i] = self._single_entropy(valid_hidden, alpha, matrix_type)
            
        return entropies
    
    def calculate_effective_rank(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Calculate effective rank for each sample in batch."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        ranks = torch.zeros(batch_size, device=hidden_states.device)
        
        for i in range(batch_size):
            # Get non-padding tokens
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]  # [valid_seq_len, hidden_dim]
            
            if valid_hidden.shape[0] == 0:
                ranks[i] = 0.0
                continue
                
            # Compute singular values
            U, S, Vh = torch.linalg.svd(valid_hidden, full_matrices=False)
            
            # Normalize singular values
            normalized_S = S / S.sum()
            
            # Compute effective rank
            effective_rank = torch.exp(-torch.sum(normalized_S * torch.log(normalized_S)))
            
            ranks[i] = effective_rank
            
        return ranks
    
    def calculate_curvature(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Calculate average curvature for each sample in batch."""
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
                
                # Handle zero vectors
                if torch.norm(v_k) < 1e-8 or torch.norm(v_k1) < 1e-8:
                    angle = 0.0
                else:
                    cos_theta = torch.dot(v_k, v_k1) / (torch.norm(v_k) * torch.norm(v_k1))
                    # Clamp for numerical stability
                    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                    angle = torch.arccos(cos_theta)
                
                angles.append(angle)
            
            if len(angles) == 0:
                curvatures[i] = 0.0
            else:
                curvatures[i] = torch.mean(torch.stack(angles))
                
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

    calculator = RepresentationMetricsCalculator(tokenizer=tokenizer)

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, calculator=calculator)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, compute_score=None, calculator=calculator)

    

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

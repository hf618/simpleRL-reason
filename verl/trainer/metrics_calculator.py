
import torch
import torch.nn.functional as F
import os
import time
import ray
from ray.util.actor_pool import ActorPool
import numpy as np
# 导入包含 Ray Actor 定义的模块
from . import metrics_utils

class RepresentationMetricsCalculator():
    """Calculates representation quality metrics from hidden states with memory optimization."""
    
    def __init__(self, tokenizer, max_seq_len=512, 
                 metric_indices=None, 
                 output_token_level_metrics=False,
                 zeroth_order_svd_method: str = 'full',
                 diff_svd_method: str = 'lowrank',
                 svd_rank: int = 6,
                 compute_log_effective_rank: bool = False,
                 ):
        """
        Initializes the RepresentationMetricsCalculator.

        Args:
            tokenizer: The tokenizer object (not directly used in metric calculation, but for context).
            max_seq_len (int): Maximum sequence length to process for memory optimization. Defaults to 512.
            compute_log_effective_rank (bool): If True, calculates and includes the log of Effective Rank and its differences. Defaults to False.
            svd_rank (int): 低秩SVD的秩. 默认为 6.
            zeroth_order_svd_method (str): SVD method for 0-order metrics ('full' or 'lowrank').
            diff_svd_method (str): SVD method for 1st/2nd order diffs ('full' or 'lowrank').
            svd_rank (int): The rank for low-rank SVD calculations.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.output_token_level_metrics = output_token_level_metrics
        self.epsilon = 1e-8
        self.compute_log_effective_rank = compute_log_effective_rank # New flag for log effective rank
        
        # 保存SVD配置
        self._cached_tensors = {}
        self.zeroth_order_svd_method = zeroth_order_svd_method
        self.diff_svd_method = diff_svd_method
        self.svd_rank = svd_rank

        # 定义所有可用的基础指标和它们的计算函数
        all_base_metrics = [
            ("Response Entropy 1", self.calculate_response_entropy),
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
        batch_size, _, _ = hidden_states.shape
        device = hidden_states.device
        selected_metric_names = [name for name, _ in self.selected_metrics]
        final_diffs = {f"{name} diff": torch.zeros(batch_size, device=device) for name in selected_metric_names}
        final_diffs.update({f"{name} diff 2": torch.zeros(batch_size, device=device) for name in selected_metric_names})
        per_stride_diffs = {f"{name} diff": [[] for _ in range(batch_size)] for name in selected_metric_names}
        per_stride_diffs.update({f"{name} diff 2": [[] for _ in range(batch_size)] for name in selected_metric_names})

        for i in range(batch_size):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]
            if valid_hidden.size(0) < 2: continue

            # 在这里传递 diff_svd_method
            per_stride_diffs_i = metrics_utils.calculate_diffs_for_single_sample(
                valid_hidden, self.max_seq_len, stride, selected_metric_names, 
                self.svd_rank, self.diff_svd_method # 使用为diff指定的SVD方法
            )
            
            # ... (聚合逻辑不变) ...
            for name in selected_metric_names:
                diff_key = f"{name} diff"
                if diff_key in per_stride_diffs_i and per_stride_diffs_i[diff_key]:
                    final_diffs[diff_key][i] = torch.tensor(per_stride_diffs_i[diff_key]).mean()
                diff2_key = f"{name} diff 2"
                if diff2_key in per_stride_diffs_i and per_stride_diffs_i[diff2_key]:
                    final_diffs[diff2_key][i] = torch.tensor(per_stride_diffs_i[diff2_key]).mean()
        return final_diffs, per_stride_diffs
    

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
    
    def calculate_response_entropy(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, alpha: float = 1.0001, matrix_type: str = 'covariance') -> torch.Tensor:
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
            
            entropies[i] = metrics_utils.compute_single_entropy(valid_hidden, alpha, matrix_type)
        return entropies
    
    def calculate_effective_rank(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, log_output: bool = False) -> torch.Tensor:
        """
        Calculates effective rank for each sample in a batch by calling the single-sample helper.

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
            # 提取有效的、非填充的token
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]  # [valid_seq_len, hidden_dim]
            
            # 为每个样本调用单一计算函数
            ranks[i] = metrics_utils.compute_single_effective_rank(
                valid_hidden, self.svd_rank, log_output, self.zeroth_order_svd_method # 使用为0阶指标指定的SVD方法
            )
        return ranks
      
    def calculate_curvature(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates average curvature for each sample in a batch by calling the single-sample helper.

        Args:
            hidden_states (torch.Tensor): Hidden states for a single layer (batch_size, seq_len, hidden_dim).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).

        Returns:
            torch.Tensor: A tensor of average curvatures for each sample in the batch.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        curvatures = torch.zeros(batch_size, device=hidden_states.device)
        
        for i in range(batch_size):
            # 提取有效的、非填充的token
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]  # [valid_seq_len, hidden_dim]
            
            # 为每个样本调用单一计算函数
            curvatures[i] = metrics_utils.compute_single_curvature(valid_hidden)
        return curvatures
    

    


class RepresentationMetricsCalculator_parallel():
    """Calculates representation quality metrics from hidden states with memory optimization."""
    
    def __init__(self, tokenizer, max_seq_len=512, svd_rank=6, 
                 compute_log_effective_rank=False, 
                 metric_indices=None, 
                 output_token_level_metrics=False,
                 computation_mode: str = "parallel"):
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


        available_cpus = ray.available_resources().get("CPU", 1)
        self.num_actors = min(2, int(available_cpus - 1))
        print(f" -> Ray reports {available_cpus} CPUs available. Creating Actor Pool with {self.num_actors} actors.")
        actors = [metrics_utils.MetricCalculatorActor.remote(self.svd_rank) for _ in range(self.num_actors)]
        self.actor_pool = ActorPool(actors)

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
                # self._free_memory()
                # 此处没有了free，Actor Pool 的生命周期由 
            return results

    @staticmethod
    def _split_list_into_chunks(data_list: list, n: int) -> list:
        """将一个列表平均分割成 n 个块。"""
        if not data_list:
            return []
        k, m = divmod(len(data_list), n)
        return [data_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

    def _run_parallel_job(self, all_tasks_with_indices, actor_method_name):
        """通用的并行任务执行器，负责分块、乱序执行、聚合和排序。"""
        if not all_tasks_with_indices: return []
        
        # --- 关键修复点 (2/2): 使用我们自己的分割函数替换 np.array_split ---
        task_chunks = self._split_list_into_chunks(all_tasks_with_indices, self.num_actors)
        

        results_chunks = self.actor_pool.map_unordered(
            lambda actor, chunk: getattr(actor, actor_method_name).remote(chunk),
            [chunk for chunk in task_chunks if len(chunk) > 0]
        )
        
        all_results_tuples = [item for sublist in results_chunks for item in sublist]
        all_results_tuples.sort(key=lambda x: x[0])
        final_results = [result for index, result in all_results_tuples]
        return final_results


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



    def calculate_response_entropy(self, hidden_states, attention_mask, alpha=1.0001, matrix_type='covariance'):
        tasks = []
        for i in range(hidden_states.shape[0]):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]
            tasks.append((i, (valid_hidden.cpu(), alpha, matrix_type)))
        
        # 调用 Actor 的 process_entropy_chunk 方法
        results = self._run_parallel_job(tasks, "process_entropy_chunk")
        return torch.tensor(results, device=hidden_states.device)

    def calculate_effective_rank(self, hidden_states, attention_mask, log_output=False):
        tasks = []
        for i in range(hidden_states.shape[0]):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]
            tasks.append((i, (valid_hidden.cpu(), log_output)))
        
        # 调用 Actor 的 process_rank_chunk 方法
        results = self._run_parallel_job(tasks, "process_rank_chunk")
        return torch.tensor(results, device=hidden_states.device)

    def calculate_curvature(self, hidden_states, attention_mask):
        tasks = []
        for i in range(hidden_states.shape[0]):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]
            tasks.append((i, (valid_hidden.cpu(),)))
        
        # 调用 Actor 的 process_curvature_chunk 方法
        results = self._run_parallel_job(tasks, "process_curvature_chunk")
        return torch.tensor(results, device=hidden_states.device)

    def calculate_metric_diff(self, hidden_states, attention_mask, stride):
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        tasks, selected_metric_names = [], [name for name, _ in self.selected_metrics]
        
        for i in range(batch_size):
            mask = attention_mask[i].bool()
            valid_hidden = hidden_states[i, mask, :]
            if valid_hidden.size(0) >= 2:
                task_args = (valid_hidden.cpu(), self.max_seq_len, stride, selected_metric_names)
                tasks.append((i, task_args))
        
        # 调用 Actor 的 process_diff_chunk 方法
        results = self._run_parallel_job(tasks, "process_diff_chunk")
        
        all_per_stride_diffs = []
        original_indices_with_tasks = {index for index, args in tasks}
        results_iter = iter(results)
        for i in range(batch_size):
            if i in original_indices_with_tasks:
                all_per_stride_diffs.append(next(results_iter))
            else:
                empty_result = {f"{name} diff": [] for name in selected_metric_names}
                empty_result.update({f"{name} diff 2": [] for name in selected_metric_names})
                all_per_stride_diffs.append(empty_result)
        
        return self._aggregate_diff_results(all_per_stride_diffs, batch_size, device, selected_metric_names)

    def _aggregate_diff_results(self, all_per_stride_diffs, batch_size, device, selected_metric_names):
        final_diffs, per_stride_diffs = {}, {}
        for name in selected_metric_names:
            final_diffs[f"{name} diff"] = torch.zeros(batch_size, device=device)
            final_diffs[f"{name} diff 2"] = torch.zeros(batch_size, device=device)
            diff_key, diff2_key = f"{name} diff", f"{name} diff 2"
            per_stride_diffs[diff_key] = [res.get(diff_key, []) for res in all_per_stride_diffs]
            per_stride_diffs[diff2_key] = [res.get(diff2_key, []) for res in all_per_stride_diffs]
        for i in range(batch_size):
            for name in selected_metric_names:
                diff_key, diff2_key = f"{name} diff", f"{name} diff 2"
                if per_stride_diffs[diff_key][i]: final_diffs[diff_key][i] = torch.tensor(per_stride_diffs[diff_key][i]).mean()
                if per_stride_diffs[diff2_key][i]: final_diffs[diff2_key][i] = torch.tensor(per_stride_diffs[diff2_key][i]).mean()
        return final_diffs, per_stride_diffs


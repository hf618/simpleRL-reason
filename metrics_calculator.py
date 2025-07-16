# metrics_calculator.py
import torch
import torch.nn.functional as F

# ##################################################################
#  将您提供的 main_ppo.py 中的 RepresentationMetricsCalculator 类的
#  完整代码粘贴到这里。为简洁起见，此处不再重复展示该类的代码。
#  请确保将整个 class RepresentationMetricsCalculator(): ... 的定义
#  都复制过来。
# ##################################################################

class RepresentationMetricsCalculator():
    """Calculates representation quality metrics from hidden states with memory optimization."""
    
    def __init__(self, tokenizer, max_seq_len=512, svd_rank=6, compute_log_effective_rank=False):
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

    def __call__(self, hidden_states, attention_mask, compute_diff=False, diff_stride=1, metrics_to_calc=None, orders_to_calc=None):
        """
        Computes representation quality metrics based on metrics_to_calc and orders_to_calc lists.
        """
        # --- 新增：设置默认阶数 ---
        if orders_to_calc is None:
            orders_to_calc = [0, 1, 2]

        if metrics_to_calc is None or not metrics_to_calc:
            metrics_to_calc = ["Response Entropy 1", "Effective Rank", "Curvature"]


        with torch.inference_mode():
            is_batched = hidden_states.dim() == 4
            
            if not is_batched:
                hidden_states = hidden_states.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            
            batch_size, seq_len, num_layers, hidden_dim = hidden_states.shape

            aggregated_results = {}
            stride_details_results = {}
            
            for layer_idx in range(num_layers):
                layer_key = str(layer_idx + 1)
                layer_hidden = hidden_states[:, :, layer_idx, :].contiguous()
                
                base_metrics = {}
                # --- 修改: 按需计算 0阶指标 ---
                if 0 in orders_to_calc:
                    if "Response Entropy 1" in metrics_to_calc:
                        base_metrics["Response Entropy 1"] = self.calculate_response_entropy(layer_hidden, attention_mask, 1, "gram")
                    if "Effective Rank" in metrics_to_calc:
                        base_metrics["Effective Rank"] = self.calculate_effective_rank(layer_hidden, attention_mask, log_output=False)
                    if "Curvature" in metrics_to_calc:
                        base_metrics["Curvature"] = self.calculate_curvature(layer_hidden, attention_mask)
                    
                    if self.compute_log_effective_rank and "Log Effective Rank" in metrics_to_calc:
                        base_metrics["Log Effective Rank"] = self.calculate_effective_rank(layer_hidden, attention_mask, log_output=True)

                stride_details = {}

                # --- 修改: 仅在需要计算1阶或2阶时才调用 diff 计算 ---
                should_compute_diff = 1 in orders_to_calc or 2 in orders_to_calc
                if should_compute_diff and metrics_to_calc:
                    # diff 计算函数需要0阶指标作为基础，所以我们传递所有可能的0阶指标
                    all_possible_base_metrics = ["Response Entropy 1", "Effective Rank", "Curvature"]
                    # 但只计算在 metrics_to_calc 中指定的那些
                    base_metric_names_to_calc_for_diff = [m for m in all_possible_base_metrics if m in metrics_to_calc]

                    if base_metric_names_to_calc_for_diff:
                        diff_metrics, stride_details = self.calculate_metric_diff(
                            layer_hidden, attention_mask, diff_stride, 
                            base_metric_names_to_calc_for_diff, 
                            orders_to_calc # <--- 传递阶数
                        )
                        base_metrics.update(diff_metrics)
                
                if not is_batched:
                    for key, tensor in base_metrics.items():
                        base_metrics[key] = tensor.squeeze(0)
                    
                    # --- 关键修复 ---
                    # history_list 是一个形如 [[2.1, 2.4, ...]] 的列表
                    # 我们直接取其内部的列表 history_list[0] 即可
                    for key, history_list in stride_details.items():
                         if history_list and isinstance(history_list[0], list):
                            stride_details[key] = history_list[0]
                         else:
                            stride_details[key] = [] # 处理空列表的情况
                    # --- 修复结束 ---

                aggregated_results[layer_key] = base_metrics
                stride_details_results[layer_key] = stride_details
                self._free_memory()
                
            return aggregated_results, stride_details_results

    def calculate_metric_diff(self, hidden_states, attention_mask, stride, base_metric_names_to_calc, orders_to_calc):
        """
        Calculates sliding window metric differences only for requested orders and metrics.
        This version contains the fix for the NameError: name 'curr_diff' is not defined.
        """
        batch_size, _, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # --- 动态生成所有需要用到的指标名称 ---
        metric_names_1_order = [f"{name} diff" for name in base_metric_names_to_calc]
        metric_names_2_order = [f"{name} diff 2" for name in base_metric_names_to_calc]
        
        # --- 动态初始化返回字典 ---
        final_diffs = {}
        if 1 in orders_to_calc:
            final_diffs.update({key: torch.zeros(batch_size, device=device) for key in metric_names_1_order})
        if 2 in orders_to_calc:
            final_diffs.update({key: torch.zeros(batch_size, device=device) for key in metric_names_2_order})

        stride_details = {key: [] for key in base_metric_names_to_calc + metric_names_1_order + metric_names_2_order}
        
        metric_calculators_map = {
            "Response Entropy 1": lambda h: self._single_entropy(h, 1, "gram"),
            "Effective Rank": lambda h: self._single_effective_rank(h, log_output=False),
            "Curvature": lambda h: self._single_curvature(h),
            "Log Effective Rank": lambda h: self._single_effective_rank(h, log_output=True),
        }
        active_calculators = [metric_calculators_map[name] for name in base_metric_names_to_calc if name in metric_calculators_map]

        if not active_calculators:
             return {}, {}

        for i in range(batch_size):
            mask = attention_mask[i].bool()
            full_valid_hidden = hidden_states[i, mask, :]
            valid_len = full_valid_hidden.size(0)

            if valid_len < stride:
                for key in stride_details: stride_details[key].append([])
                continue

            num_metrics = len(active_calculators)
            history_sum = [0.0] * num_metrics
            history_count = 0
            total_diff, total_diff2 = [0.0] * num_metrics, [0.0] * num_metrics
            valid_diff_count = 0
            prev_diff = None
            
            temp_history = {key: [] for key in stride_details.keys()}

            for t in range(stride, valid_len + 1, stride):
                sub_hidden = full_valid_hidden[:t, :]
                if sub_hidden.size(0) < 2: continue

                current_metrics = [calc(sub_hidden) for calc in active_calculators]
                
                if 0 in orders_to_calc:
                    for k, name in enumerate(base_metric_names_to_calc):
                        temp_history[name].append(current_metrics[k])
                
                # --- 逻辑重构开始 ---
                curr_diff = None  # 每一轮都初始化

                if history_count > 0:
                    # 只要有历史记录，就计算一阶差分，以备二阶差分使用
                    hist_avg = [s / history_count for s in history_sum]
                    curr_diff = [(curr - avg) for curr, avg in zip(current_metrics, hist_avg)]
                    total_diff = [s + d for s, d in zip(total_diff, curr_diff)]
                    valid_diff_count += 1
                
                # 按需记录一阶差分
                if 1 in orders_to_calc:
                    if curr_diff is not None:
                        for k, name in enumerate(metric_names_1_order):
                            temp_history[name].append(curr_diff[k])
                    else: # 第一次没有历史记录
                        for k, name in enumerate(metric_names_1_order):
                            temp_history[name].append(0.0)

                # 按需记录二阶差分
                if 2 in orders_to_calc:
                    if prev_diff is not None and curr_diff is not None:
                        curr_diff2 = [(curr_d - prev_d) for curr_d, prev_d in zip(curr_diff, prev_diff)]
                        total_diff2 = [s + d2 for s, d2 in zip(total_diff2, curr_diff2)]
                        for k, name in enumerate(metric_names_2_order):
                            temp_history[name].append(curr_diff2[k])
                    else: # 没有足够的历史记录
                        for k, name in enumerate(metric_names_2_order):
                            temp_history[name].append(0.0)
                # --- 逻辑重构结束 ---

                history_sum = [s + c for s, c in zip(history_sum, current_metrics)]
                history_count += 1
                # 更新 prev_diff 以供下一轮计算二阶差分使用
                if curr_diff is not None:
                    prev_diff = curr_diff

            for key, values in temp_history.items():
                stride_details[key].append(values)

            # 计算并填充最终的平均值
            if valid_diff_count > 0:
                if 1 in orders_to_calc:
                    avg_diff = [t / valid_diff_count for t in total_diff]
                    for k, name in enumerate(metric_names_1_order):
                        final_diffs[name][i] = avg_diff[k]
                
                if 2 in orders_to_calc and valid_diff_count > 1:
                    avg_diff2 = [t / (valid_diff_count - 1) for t in total_diff2]
                    for k, name in enumerate(metric_names_2_order):
                        final_diffs[name][i] = avg_diff2[k]
            
        return final_diffs, stride_details

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
                matrix = matrix.to(torch.float64)
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
        
        
        with torch.amp.autocast(device_type='cuda'):
            # Perform low-rank SVD for efficiency
            # q is the number of singular values to compute, capped by hidden_dim
            hidden = hidden.to(torch.float64)  # Ensure float32 for SVD stability
            _, S, _ = torch.svd_lowrank(hidden, q=min(self.svd_rank, hidden.size(1)))
            
            # Normalize singular values to sum to 1
            normalized_S = S / (S.sum() + 1e-8) # Add epsilon for stability
            
            # Compute effective rank using the formula: exp(-sum(p_i * log(p_i)))
            # Add epsilon to log argument for stability if p_i is zero
            if log_output:
                # Directly return -sum(p_i * log(p_i)) when log_output is True
                return -torch.sum(normalized_S * torch.log(normalized_S + 1e-8)).item()
            else:
                # Return the effective rank itself
                return torch.exp(-torch.sum(normalized_S * torch.log(normalized_S + 1e-8))).item()

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
                
            # Compute singular values using full SVD for batch version
            # (can be replaced with low-rank SVD if performance is critical and rank is small)
            U, S, Vh = torch.linalg.svd(valid_hidden, full_matrices=False)
            
            # Normalize singular values
            normalized_S = S / (S.sum() + 1e-8) # Add epsilon for stability
            
            # Compute effective rank
            # Add epsilon to log argument for stability if p_i is zero
            if log_output:
                # Directly return -sum(p_i * log(p_i)) when log_output is True
                ranks[i] = -torch.sum(normalized_S * torch.log(normalized_S + 1e-8))
            else:
                # Return the effective rank itself
                ranks[i] = torch.exp(-torch.sum(normalized_S * torch.log(normalized_S + 1e-8)))
            
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

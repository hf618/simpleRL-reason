# verl/trainer/metrics_utils.py


import torch
import torch.nn.functional as F
import os
import ray 

# --- 所有单一计算逻辑函数保持不变 ---
def compute_single_entropy(hidden: torch.Tensor, alpha: float = 1.0001, matrix_type: str = 'gram') -> float:
    """计算单个样本的熵"""
    assert matrix_type in ['covariance', 'gram'], "matrix_type must be 'covariance' or 'gram'"
    if hidden.size(0) < 2: return 0.0
    try:
        centered = hidden - hidden.mean(dim=0, keepdim=True)
        matrix = None
        if matrix_type == 'covariance':
            matrix = centered.T @ centered / (centered.size(0) - 1)
        else: # 'gram'
            matrix = centered @ centered.T
        
        matrix = matrix.to(torch.float32)
        eigvals = torch.linalg.eigvalsh(matrix) # 始终计算全部特征值
        eigvals = eigvals[eigvals > 1e-8]
        if len(eigvals) == 0: return 0.0
        
        normalized = eigvals / eigvals.sum()
        if abs(alpha - 1.0) < 1e-6:
            normalized = normalized[normalized > 1e-12]
            return -torch.sum(normalized * torch.log(normalized)).item()
        else:
            return (1/(1-alpha)) * torch.log(torch.sum(normalized**alpha)).item()
    except torch._C._LinAlgError:
        return 0.0

def compute_single_effective_rank(hidden: torch.Tensor, svd_rank: int, svd_niter: int, log_output: bool = False, method: str = 'lowrank') -> tuple[float, int]:
    """
    计算单个样本的有效秩和传统秩。
    高效地只执行一次SVD计算。
    返回: (effective_rank, traditional_rank)
    """
    assert method in ['lowrank', 'full'], "SVD method must be 'lowrank' or 'full'"
    if hidden.size(0) < 2: return 0.0, 0
    
    try:
        centered = hidden - hidden.mean(dim=0, keepdim=True)
        centered = centered.to(torch.float32)
        S = None
        if method == 'lowrank':
            _, S, _ = torch.svd_lowrank(centered, q=min(svd_rank, min(centered.shape)), niter=svd_niter)
        else: # 'full'
            S = torch.linalg.svdvals(centered)
            
        # --- 传统 Rank 的计算 ---
        # 只有在SVD计算成功且S非空时才计算
        traditional_rank = 0
        if S is not None and S.numel() > 0:
            # 使用PyTorch推荐的、稳健的阈值计算方法
            tol = S.max() * max(centered.shape) * torch.finfo(S.dtype).eps
            traditional_rank = torch.sum(S > tol).item()
        else:
            # 如果SVD失败或S为空，返回0
            return 0.0, 0

        # --- Effective Rank 的计算 ---
        normalized_S = S / (S.sum() + 1e-8)
        effective_rank_val = 0.0
        if log_output:
            effective_rank_val = -torch.sum(normalized_S * torch.log(normalized_S + 1e-8)).item()
        else:
            effective_rank_val = torch.exp(-torch.sum(normalized_S * torch.log(normalized_S + 1e-8))).item()
            
        return effective_rank_val, traditional_rank

    except torch._C._LinAlgError:
        return 0.0, 0

def compute_single_curvature(hidden: torch.Tensor) -> float:
    """计算单个样本的曲率"""
    if hidden.size(0) < 3: return 0.0
    diffs = hidden[1:] - hidden[:-1]
    angles = []
    chunk_size = 256
    for chunk in torch.split(diffs, chunk_size, dim=0):
        if chunk.size(0) < 2: continue
        norms = torch.norm(chunk, dim=1, keepdim=True)
        valid = (norms > 1e-6).squeeze()
        chunk = chunk[valid]
        if chunk.size(0) < 2: continue
        cos_sim = F.cosine_similarity(chunk[:-1], chunk[1:], dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angles.append(torch.arccos(cos_sim))
    if angles:
        return torch.cat(angles).mean().item()
    return 0.0

def calculate_diffs_for_single_sample(valid_hidden, max_seq_len, stride, selected_metric_names, 
                                      svd_rank, svd_niter, svd_method): # 增加 svd_niter 参数
    """为单个样本的隐藏状态计算所有选定指标的一阶和二阶差分。"""
    metric_calculators = {
        "Response Entropy 1": lambda h: compute_single_entropy(h, 1.0001, "gram"),
        "Curvature": lambda h: compute_single_curvature(h),
        "Effective Rank": lambda h: compute_single_effective_rank(h, svd_rank, svd_niter, log_output=False, method=svd_method)[0],
        "Log Effective Rank": lambda h: compute_single_effective_rank(h, svd_rank, svd_niter, log_output=True, method=svd_method)[0],
        "Traditional Rank": lambda h: compute_single_effective_rank(h, svd_rank, svd_niter, method=svd_method)[1]
    }
    # ... (函数其余部分保持不变) ...
    active_calculators = [metric_calculators[name] for name in selected_metric_names if name in metric_calculators]
    num_metrics_to_track = len(active_calculators)
    valid_len = valid_hidden.size(0)
    history_sum, history_count, prev_diff = [0.0] * num_metrics_to_track, 0, None
    per_stride_diffs_i = {f"{name} diff": [] for name in selected_metric_names}
    per_stride_diffs_i.update({f"{name} diff 2": [] for name in selected_metric_names})
    if valid_len > max_seq_len:
        valid_hidden = valid_hidden[-max_seq_len:]
        valid_len = max_seq_len
    for t in range(1, valid_len):
        if t % stride != 0: continue
        sub_hidden = valid_hidden[max(0, t - max_seq_len + 1):t+1]
        current_metrics = [calc(sub_hidden) for calc in active_calculators]
        if history_count > 0:
            hist_avg = [s / history_count for s in history_sum]
            curr_diff = [(curr - avg) for curr, avg in zip(current_metrics, hist_avg)]
            for idx, name in enumerate(selected_metric_names): 
                per_stride_diffs_i[f"{name} diff"].append(curr_diff[idx])
            if prev_diff is not None:
                curr_diff2 = [(cd - pd) for cd, pd in zip(curr_diff, prev_diff)]
                for idx, name in enumerate(selected_metric_names): 
                    per_stride_diffs_i[f"{name} diff 2"].append(curr_diff2[idx])
            prev_diff = curr_diff
        history_sum = [s + curr for s, curr in zip(history_sum, current_metrics)]
        history_count += 1
    return per_stride_diffs_i


@ray.remote
class MetricCalculatorActor:
    """一个多功能的 Ray Actor，优化后可以批量处理所有类型的计算任务。"""
    def __init__(self, svd_rank: int):
        self.svd_rank = svd_rank
        torch.set_num_threads(1)

    # (A) 熵计算：处理任务块
    def process_entropy_chunk(self, chunk_of_args):
        results = []
        for index, args in chunk_of_args:
            valid_hidden, alpha, matrix_type = args
            result = compute_single_entropy(valid_hidden, alpha, matrix_type)
            results.append((index, result))
        return results

    # (B) 有效秩计算：处理任务块
    def process_rank_chunk(self, chunk_of_args):
        results = []
        for index, args in chunk_of_args:
            valid_hidden, log_output = args
            result = compute_single_effective_rank(valid_hidden, self.svd_rank, log_output)
            results.append((index, result))
        return results

    # (C) 曲率计算：处理任务块
    def process_curvature_chunk(self, chunk_of_args):
        results = []
        for index, args in chunk_of_args:
            valid_hidden, = args
            result = compute_single_curvature(valid_hidden)
            results.append((index, result))
        return results

    # (D) 差分计算：处理任务块
    def process_diff_chunk(self, chunk_of_args):
        # 修改 6/6: 修改 - Actor现在直接调用共享函数，不再需要内部的 _process_single_diff 方法。
        # Actor的职责被简化为任务分发和结果收集。
        results = []
        for index, args in chunk_of_args:
            valid_hidden, max_seq_len, stride, selected_metric_names = args
            # 调用新的共享函数
            result = calculate_diffs_for_single_sample(
                valid_hidden, max_seq_len, stride, selected_metric_names, self.svd_rank
            )
            results.append((index, result))
        return results

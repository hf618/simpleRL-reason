# verl/trainer/metrics_utils.py


import torch
import torch.nn.functional as F
import os
import ray 

# --- 单一计算函数：提升精度以作为精确基准 ---
def compute_single_entropy(hidden: torch.Tensor, alpha: float = 1.0001, matrix_type: str = 'gram') -> float:
    """计算单个样本的熵"""
    assert matrix_type in ['covariance', 'gram'], "matrix_type must be 'covariance' or 'gram'"
    if hidden.size(0) < 2: return 0.0
    try:
        # 提前将输入转为float32，以进行高精度计算
        hidden_f32 = hidden.to(torch.float32)
        centered = hidden_f32 - hidden_f32.mean(dim=0, keepdim=True)
        
        matrix = None
        if matrix_type == 'covariance':
            matrix = centered.T @ centered / (centered.size(0) - 1)
        else: # 'gram'
            matrix = centered @ centered.T
        
        eigvals = torch.linalg.eigvalsh(matrix) # matrix 已经是 float32
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

def compute_single_effective_rank(hidden: torch.Tensor, svd_rank: int, svd_niter: int, log_output: bool = False, method: str = 'lowrank') -> tuple[float, float]:
    """计算单个样本的有效秩和传统秩。"""
    assert method in ['lowrank', 'full'], "SVD method must be 'lowrank' or 'full'"
    if hidden.size(0) < 2: return 0.0, 0
    try:
        # 提前将输入转为float32，以进行高精度中心化
        hidden_f32 = hidden.to(torch.float32)
        centered = hidden_f32 - hidden_f32.mean(dim=0, keepdim=True)

        S = None
        if method == 'lowrank':
            _, S, _ = torch.svd_lowrank(centered, q=min(svd_rank, min(centered.shape)), niter=svd_niter)
        else: # 'full'
            S = torch.linalg.svdvals(centered)
            
        traditional_rank = 0
        if S is not None and S.numel() > 0:
            tol = S.max() * max(centered.shape) * torch.finfo(S.dtype).eps
            traditional_rank = torch.sum(S > tol).item()
        else:
            return 0.0, 0.0

        normalized_S = S / (S.sum() + 1e-8)
        effective_rank_val = 0.0
        if log_output:
            effective_rank_val = -torch.sum(normalized_S * torch.log(normalized_S + 1e-8)).item()
        else:
            effective_rank_val = torch.exp(-torch.sum(normalized_S * torch.log(normalized_S + 1e-8))).item()
            
        return effective_rank_val, float(traditional_rank)
    except torch._C._LinAlgError:
        return 0.0, 0.0

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

def _get_metrics_from_eigenvalues(eigenvalues, selected_metric_names):
    """辅助函数：根据格拉姆矩阵的特征值计算所有需要的指标"""
    S = torch.sqrt(torch.relu(eigenvalues))
    results = []
    for name in selected_metric_names:
        if name == "Response Entropy 1":
            eigvals = eigenvalues[eigenvalues > 1e-8]
            if len(eigvals) == 0:
                results.append(0.0)
                continue
            normalized = eigvals / eigvals.sum()
            normalized = normalized[normalized > 1e-12]
            entropy = -torch.sum(normalized * torch.log(normalized)).item()
            results.append(entropy)
        elif name in ["Effective Rank", "Log Effective Rank", "Traditional Rank"]:
            normalized_S = S / (S.sum() + 1e-8)
            shannon_entropy_S = -torch.sum(normalized_S * torch.log(normalized_S + 1e-8)).item()
            if name == "Effective Rank":
                results.append(torch.exp(torch.tensor(shannon_entropy_S)).item())
            elif name == "Log Effective Rank":
                results.append(shannon_entropy_S)
            elif name == "Traditional Rank":
                if S.numel() > 0:
                    tol = S.max() * max(S.shape) * torch.finfo(S.dtype).eps
                    trad_rank = torch.sum(S > tol).item()
                    results.append(float(trad_rank))
                else:
                    results.append(0.0)
        elif name == "Curvature":
            results.append(0.0)
        else:
            results.append(0.0)
    return results

def calculate_diffs_for_single_sample_optimized(valid_hidden, max_seq_len, stride, selected_metric_names, 
                                                svd_rank, svd_niter, svd_method):
    """
    [最终生产版] 高效、精确、逻辑正确的优化函数。
    此版本通过在高精度下进行增量累加来确保数值稳定性和正确性。
    """
    valid_len = valid_hidden.size(0)
    if valid_len > max_seq_len:
        valid_hidden = valid_hidden[-max_seq_len:]
        valid_len = max_seq_len

    per_stride_diffs_i = {f"{name} diff": [] for name in selected_metric_names}
    per_stride_diffs_i.update({f"{name} diff 2": [] for name in selected_metric_names})
    
    if valid_len < 2 * stride:
        return per_stride_diffs_i

    history_sum = [0.0] * len(selected_metric_names)
    history_count = 0
    prev_diff = None

    # 状态变量和计算全程使用 float32
    s = torch.zeros(1, valid_hidden.shape[1], device=valid_hidden.device, dtype=torch.float32)
    U = None
    H_old = None

    for t in range(stride, valid_len, stride):
        current_window = valid_hidden[:t+1]
        current_window_f32 = current_window.to(torch.float32)
        
        new_chunk = current_window_f32[len(H_old) if H_old is not None else 0:]

        s = s + new_chunk.sum(dim=0, keepdim=True)
        
        if U is None: 
            U = new_chunk @ new_chunk.T
        else:
            C12 = H_old @ new_chunk.T
            C22 = new_chunk @ new_chunk.T
            top_part = torch.cat([U, C12], dim=1)
            bottom_part = torch.cat([C12.T, C22], dim=1)
            U = torch.cat([top_part, bottom_part], dim=0)

        k = current_window_f32.shape[0]
        mean_vec = s / k
        mean_gram = mean_vec @ mean_vec.T
        hs_T = current_window_f32 @ s.T / k
        
        ones_k = torch.ones((k, 1), device=current_window_f32.device, dtype=torch.float32)
        G = U - hs_T @ ones_k.T - ones_k @ hs_T.T + mean_gram
        
        eigenvalues = torch.linalg.eigvalsh(G)
        current_metrics = _get_metrics_from_eigenvalues(eigenvalues, selected_metric_names)

        if "Curvature" in selected_metric_names:
            current_metrics[selected_metric_names.index("Curvature")] = compute_single_curvature(current_window)

        if history_count > 0:
            hist_avg = [sm / history_count for sm in history_sum]
            curr_diff = [(curr - avg) for curr, avg in zip(current_metrics, hist_avg)]
            for idx, name in enumerate(selected_metric_names): 
                per_stride_diffs_i[f"{name} diff"].append(curr_diff[idx])
            if prev_diff is not None:
                curr_diff2 = [(cd - pd) for cd, pd in zip(curr_diff, prev_diff)]
                for idx, name in enumerate(selected_metric_names): 
                    per_stride_diffs_i[f"{name} diff 2"].append(curr_diff2[idx])
            prev_diff = curr_diff
            
        history_sum = [sm + curr for sm, curr in zip(history_sum, current_metrics)]
        history_count += 1
        
        H_old = current_window_f32
            
    return per_stride_diffs_i

def calculate_diffs_for_single_sample_original(valid_hidden, max_seq_len, stride, selected_metric_names, 
                                      svd_rank, svd_niter, svd_method):
    """为单个样本的隐藏状态计算所有选定指标的一阶和二阶差分。(基准版本)"""
    metric_calculators = {
        "Response Entropy 1": lambda h: compute_single_entropy(h, 1.0001, "gram"),
        "Curvature": lambda h: compute_single_curvature(h),
        "Effective Rank": lambda h: compute_single_effective_rank(h, svd_rank, svd_niter, log_output=False, method=svd_method)[0],
        "Log Effective Rank": lambda h: compute_single_effective_rank(h, svd_rank, svd_niter, log_output=True, method=svd_method)[0],
        "Traditional Rank": lambda h: compute_single_effective_rank(h, svd_rank, svd_niter, method=svd_method)[1]
    }
    active_calculators = [metric_calculators[name] for name in selected_metric_names if name in metric_calculators]
    num_metrics_to_track = len(active_calculators)
    valid_len = valid_hidden.size(0)
    if valid_len > max_seq_len:
        valid_hidden = valid_hidden[-max_seq_len:]
        valid_len = max_seq_len

    history_sum, history_count, prev_diff = [0.0] * num_metrics_to_track, 0, None
    per_stride_diffs_i = {f"{name} diff": [] for name in selected_metric_names}
    per_stride_diffs_i.update({f"{name} diff 2": [] for name in selected_metric_names})

    for t in range(stride, valid_len, stride):
        sub_hidden = valid_hidden[:t+1]
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
        results = []
        for index, args in chunk_of_args:
            # 假设 args 的顺序是 (valid_hidden, max_seq_len, stride, selected_metric_names, svd_niter, svd_method)
            # 这个顺序需要在调用方 (metrics_calculator.py 的并行版本) 中保证
            valid_hidden, max_seq_len, stride, selected_metric_names, svd_niter, svd_method = args
            
            # 调用最终的生产函数，并传递所有参数
            result = calculate_diffs_for_single_sample_optimized(
                valid_hidden, max_seq_len, stride, selected_metric_names, 
                self.svd_rank, svd_niter, svd_method
            )
            results.append((index, result))
        return results

import sys
import torch

# FP Quantization-------------------------------------------------------------------------------------------------------

def quantize_to_fp(x_float: torch.Tensor, n_bits: int, maxval: torch.Tensor,
                  num_mantissa_bits: torch.Tensor, sign_bits: int,
                  gpt2_conv1d=False) -> torch.Tensor:
    if gpt2_conv1d:
        x_float = x_float.T  # 转置后的形状为 (C, N)

    maxval = maxval.to(x_float.device)
    num_mantissa_bits = num_mantissa_bits.to(x_float.device)

    x_dim = x_float.dim()
    max_dim = maxval.dim()

    if max_dim >= x_dim:
        maxval = maxval
    else:
        expand_dims = x_dim - max_dim
        for _ in range(expand_dims):
            maxval = maxval.unsqueeze(-1)

    M = torch.clamp(torch.round(num_mantissa_bits), 1, n_bits - sign_bits)
    E = n_bits - sign_bits - M

    # 确保 maxval 的形状与 x_float 兼容（已通过上述 unsqueeze 处理）
    bias = 2 ** E - torch.log2(maxval) + torch.log2(2 - 2 ** (-M)) - 1
    minval = -maxval if sign_bits == 1 else torch.zeros_like(maxval)

    # 确保 x_float 和 minval/maxval 的形状兼容
    xc = torch.min(torch.max(x_float, minval), maxval)

    log_scales = torch.clamp(torch.floor(torch.log2(torch.abs(xc)) + bias).detach(), min=1.0)
    scales = 2.0 ** (log_scales - M - bias)
    result = (xc / scales).round_() * scales

    if gpt2_conv1d:
        result = result.T

    return result


def mse_range_estimator(x: torch.Tensor, n_bits: int = 8, sign_bits: int = 1,
                        num_mantissa_bits: torch.Tensor = torch.tensor(3),
                        per_channel: bool = False, gpt2_conv1d: bool = False) -> torch.Tensor:
    if per_channel:
        if gpt2_conv1d:
            x_flat = x.view(x.shape[0], -1).T.to(x.device)
        else:
            x_flat = x.view(x.shape[0], -1).to(x.device)
    else:
        x_flat = x.view(1, -1).to(x.device)

    max_abs_vals = torch.max(torch.abs(x_flat.min(dim=-1).values), torch.abs(x_flat.max(dim=-1).values))


    # 生成搜索网格（保持原逻辑）
    search_grids = []

    for mx in max_abs_vals:
        if mx == 0.0:
            grid = torch.full((120,), 1e-6, device=x.device)
        else:
            grid = torch.linspace(0.1 * mx, 1.2 * mx, 120, device=x.device)
        search_grids.append(grid)
    search_grids = torch.stack(search_grids, dim=0)  # 形状为 (C, 120)

    # 分批次处理（关键改进）
    batch_size = 30  # 每批处理11个格点（可根据显存调整）
    num_batches = (search_grids.shape[1] + batch_size - 1) // batch_size  # 向上取整
    all_mse = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size
        current_search_grids = search_grids[:, start:end]  # 当前批次的格点

        # 调整输入形状以便与 current_search_grids 广播
        current_x_float = x_flat.unsqueeze(1).expand(-1, current_search_grids.shape[1], -1)

        # 执行量化（分批次）
        quantized_batch = quantize_to_fp(x_float=current_x_float, n_bits=n_bits, maxval=current_search_grids,
                                         num_mantissa_bits=num_mantissa_bits, sign_bits=sign_bits)

        # 计算当前批次的MSE
        current_mse = torch.mean( (x_flat.unsqueeze(1) - quantized_batch) ** 2, dim=-1 )
        all_mse.append(current_mse)

        # 释放当前批次的显存（可选）
        del current_search_grids, current_x_float, quantized_batch
        torch.cuda.empty_cache()  # 清理显存（如果使用GPU）

    # 合并所有批次的MSE结果
    mse = torch.cat(all_mse, dim=1)  # 沿着格点维度拼接

    # 寻找最优的maxval
    min_indices = torch.argmin(mse, dim=1)
    selected_maxvals = search_grids[torch.arange(len(min_indices)), min_indices]

    return selected_maxvals

# FP Quantization-------------------------------------------------------------------------------------------------------


import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.utils_quant import quantize_to_fp, mse_range_estimator


class MinMaxQuantBatchNorm2d(nn.BatchNorm2d):
    """
    MinMax quantize gamma and beta parameters of BatchNorm2d with same format
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=False,
                 quant_bit=8):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.quant_bit = quant_bit
        self.g_interval = None
        self.b_interval = None
        self.q_max = 2 ** (quant_bit - 1)  # Gamma和Beta使用相同的量化范围

    def forward(self, x):
        # 自动计算量化间隔（在第一次前向传播时）
        if self.affine and (self.g_interval is None or self.b_interval is None):
            self._compute_quant_intervals()

        # 计算当前batch的均值和方差（不更新running stats）
        mean = x.mean([0, 2, 3])
        var = x.var([0, 2, 3], unbiased=False)
        x_normalized = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)

        if self.affine:
            g_sim, b_sim = self._quantize_params()
            return g_sim[None, :, None, None] * x_normalized + b_sim[None, :, None, None]
        else:
            return x_normalized

    def _compute_quant_intervals(self):
        # 计算Gamma和Beta的量化间隔
        gamma_abs_max = self.weight.abs().max()
        self.g_interval = gamma_abs_max / (self.q_max - 0.5)

        beta_abs_max = self.bias.abs().max()
        self.b_interval = beta_abs_max / (self.q_max - 0.5)

    def _quantize_params(self):
        # 量化Gamma和Beta参数
        gamma_quant = (self.weight / self.g_interval).round().clamp_(-self.q_max, self.q_max - 1)
        gamma_quant = gamma_quant * self.g_interval

        beta_quant = (self.bias / self.b_interval).round().clamp_(-self.q_max, self.q_max - 1)
        beta_quant = beta_quant * self.b_interval

        return gamma_quant, beta_quant


class MinMaxQuantLayerNorm(nn.LayerNorm):
    """
    MinMax quantize gamma and beta parameters of LayerNorm with same format
    """
    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 quant_bit: int = 8):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.quant_bit = quant_bit
        self.g_interval = None
        self.b_interval = None
        self.q_max = 2 ** (quant_bit - 1)  # Gamma和Beta使用相同的量化范围

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算量化间隔（在第一次前向传播时）
        if self.elementwise_affine and (self.g_interval is None or self.b_interval is None):
            self._compute_quant_intervals()

        # 计算归一化维度（动态确定归一化维度）
        dims = list(range(-len(self.normalized_shape), 0)) if isinstance(self.normalized_shape, (list, tuple)) else -1
        mean = x.mean(dims, keepdim=True)
        var = x.var(dims, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            gamma_quant, beta_quant = self._quantize_params()
            return gamma_quant * x_normalized + beta_quant
        else:
            return x_normalized

    def _compute_quant_intervals(self):
        # 计算Gamma和Beta的量化间隔
        gamma_abs_max = self.weight.abs().max()
        self.g_interval = gamma_abs_max / (self.q_max - 0.5)

        if self.bias is not None:
            beta_abs_max = self.bias.abs().max()
            self.b_interval = beta_abs_max / (self.q_max - 0.5)

    def _quantize_params(self):
        # 量化Gamma和Beta参数
        gamma_quant = (self.weight / self.g_interval).round().clamp_(-self.q_max, self.q_max - 1)
        gamma_quant = gamma_quant * self.g_interval

        if self.bias is not None:
            beta_quant = (self.bias / self.b_interval).round().clamp_(-self.q_max, self.q_max - 1)
            beta_quant = beta_quant * self.b_interval
            return gamma_quant, beta_quant
        else:
            return gamma_quant, None


class FPQuantBatchNorm2d(MinMaxQuantBatchNorm2d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = False,  # 禁用全局统计量
                 mode: str = 'raw',
                 n_bits: int = 8,
                 sign_bit: int = 1,
                 exponent: int = 4,
                 metric: str = "L2_norm",
                 per_channel: bool = False):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            quant_bit=n_bits  # 继承 MinMaxQuantBatchNorm2d 的量化位数
        )
        self.mode = mode
        self.n_bits = n_bits
        self.sign_bit = sign_bit
        self.exponent = exponent
        self.mantissa = self.n_bits - self.sign_bit - self.exponent
        self.mantissa = torch.tensor([float(self.mantissa)])
        self.metric = metric
        self.per_channel = per_channel
        self.gamma_maxval = None
        self.beta_maxval = None
        self.calibrated = False
        self.raw_input = None
        self.raw_out = None

    def forward(self, x):
        if self.mode == 'raw':
            return super().forward(x)  # 继承 MinMaxQuantBatchNorm2d 的手动计算逻辑
        elif self.mode == 'quant_forward':
            return self.quant_forward(x)
        elif self.mode == 'calibration_step2':
            self.calibration_step2()
            return None
        else:
            raise NotImplementedError

    def quant_forward(self, x):
        assert self.calibrated, "Need to calibrate first"
        gamma_quant = quantize_to_fp(
            self.weight,
            self.n_bits,
            self.gamma_maxval,
            self.mantissa,
            self.sign_bit
        )
        beta_quant = quantize_to_fp(
            self.bias,
            self.n_bits,
            self.beta_maxval,
            self.mantissa,
            self.sign_bit
        )

        # 手动计算均值和方差（与 MinMaxQuantBatchNorm2d 的 forward 一致）
        mean = x.mean([0, 2, 3])  # 当前 batch 的均值（通道维度）
        var = x.var([0, 2, 3], unbiased=False)  # 当前 batch 的方差
        x_normalized = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)

        # 应用量化后的 gamma 和 beta
        return gamma_quant[None, :, None, None] * x_normalized + beta_quant[None, :, None, None]

    def calibration_step2(self):
        with torch.no_grad():
            self.raw_input = self.raw_input.to(self.weight.device)
            self.raw_out = self.raw_out.to(self.weight.device)
            
            # 提前计算固定输入的均值和方差（仅计算一次）
            mean = self.raw_input.mean([0, 2, 3])  # 通道维度的均值
            var = self.raw_input.var([0, 2, 3], unbiased=False)  # 通道维度的方差
            x_normalized = (self.raw_input - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)

            optimal_mantissa = None
            min_mse = float('inf')
            for mb in range(1, self.n_bits):
                current_mantissa = torch.tensor([mb], dtype=torch.float32).to(self.weight.device)
                self.mantissa = current_mantissa
                self.gamma_maxval = mse_range_estimator(
                    self.weight, self.n_bits, self.sign_bit, current_mantissa
                )
                self.beta_maxval = mse_range_estimator(
                    self.bias, self.n_bits, self.sign_bit, current_mantissa
                )
                gamma_quant = quantize_to_fp(
                    self.weight, self.n_bits, self.gamma_maxval, current_mantissa, self.sign_bit
                )
                beta_quant = quantize_to_fp(
                    self.bias, self.n_bits, self.beta_maxval, current_mantissa, self.sign_bit
                )
                
                # 直接使用预计算的 x_normalized，仅需应用量化后的 gamma 和 beta
                out_quant = gamma_quant[None, :, None, None] * x_normalized + beta_quant[None, :, None, None]
                mse = torch.mean((self.raw_out - out_quant) ** 2)
                if mse < min_mse:
                    min_mse = mse
                    optimal_mantissa = current_mantissa
            self.mantissa = optimal_mantissa
            self.exponent = self.n_bits - self.mantissa - self.sign_bit
            self.gamma_maxval = mse_range_estimator(
                self.weight, self.n_bits, self.sign_bit, self.mantissa
            )
            self.beta_maxval = mse_range_estimator(
                self.bias, self.n_bits, self.sign_bit, self.mantissa
            )
            self.calibrated = True
            del self.raw_input, self.raw_out


class FPQuantLayerNorm(MinMaxQuantLayerNorm):
    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 mode: str = 'raw',
                 n_bits: int = 8,
                 sign_bit: int = 1,
                 exponent: int = 4,
                 em_mx: bool = True,
                 bias_mx: bool = True,
                 metric: str = "L2_norm"
                 ):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.mode = mode
        self.n_bits = n_bits
        self.sign_bit = sign_bit
        self.exponent = exponent
        self.mantissa = self.n_bits - self.sign_bit - self.exponent
        self.mantissa = torch.tensor([float(self.mantissa)])
        self.em_mx = em_mx
        self.bias_mx = bias_mx
        self.metric = metric
        self.gamma_maxval = None
        self.beta_maxval = None
        self.calibrated = False
        self.raw_input = None
        self.raw_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'raw':
            return super().forward(x)
        elif self.mode == 'quant_forward':
            return self.quant_forward(x)
        elif self.mode == 'calibration_step2':
            self.calibration_step2()
            return None
        else:
            raise NotImplementedError(f"Mode {self.mode} not supported.")

    def quant_forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.calibrated, "Need to calibrate first"

        gamma_quant = quantize_to_fp(self.weight, self.n_bits, self.gamma_maxval, self.mantissa, self.sign_bit)
        if self.bias is not None:
            beta_quant = quantize_to_fp(self.bias, self.n_bits, self.beta_maxval, self.mantissa, self.sign_bit)
        else:
            beta_quant = None
        return F.layer_norm(x, normalized_shape=self.normalized_shape, weight=gamma_quant, bias=beta_quant, eps=self.eps)

    def calibration_step2(self):

        with (torch.no_grad()):

            self.raw_input = self.raw_input.to(self.weight.device)

            self.raw_out = self.raw_out.to(self.weight.device)

            if self.em_mx:

                optimal_mantissa = None

                min_mse = float('inf')

                for mb in range(1, self.n_bits):

                    current_mantissa = torch.tensor([mb], dtype=torch.float32).to(self.weight.device)

                    self.mantissa = current_mantissa

                    self.gamma_maxval = mse_range_estimator(self.weight, self.n_bits, self.sign_bit, current_mantissa)

                    if self.bias is not None:

                        self.beta_maxval = mse_range_estimator(self.bias, self.n_bits, self.sign_bit, current_mantissa)

                    else:

                        self.beta_maxval = None

                    gamma_quant = quantize_to_fp(self.weight, self.n_bits, self.gamma_maxval, current_mantissa, self.sign_bit)

                    beta_quant = quantize_to_fp(self.bias, self.n_bits, self.beta_maxval, current_mantissa,
                                                self.sign_bit)if self.bias is not None else None

                    out_quant = F.layer_norm(self.raw_input, normalized_shape=self.normalized_shape, weight=gamma_quant,
                                             bias=beta_quant, eps=self.eps)

                    mse = torch.mean((self.raw_out - out_quant) ** 2)

                    if mse < min_mse:

                        min_mse = mse

                        optimal_mantissa = current_mantissa

                self.mantissa = optimal_mantissa

                self.exponent = self.n_bits - self.mantissa - self.sign_bit

                self.gamma_maxval = mse_range_estimator(self.weight, self.n_bits, self.sign_bit, self.mantissa)

                if self.bias is not None:

                    self.beta_maxval = mse_range_estimator(self.bias, self.n_bits, self.sign_bit, self.mantissa)

                else:

                    self.beta_maxval = None

            else:

                if self.bias_mx:

                    self.gamma_maxval = mse_range_estimator(self.weight, self.n_bits, self.sign_bit, self.mantissa)

                    if self.bias is not None:

                        self.beta_maxval = mse_range_estimator(self.bias, self.n_bits, self.sign_bit, self.mantissa)

                    else:

                        self.beta_maxval = None

                else:

                    default_maxval = (2 - 2 ** (-self.mantissa)) * 2 ** (2 ** (self.exponent - 1) - 1)

                    self.gamma_maxval = default_maxval

                    self.beta_maxval = default_maxval

            self.calibrated = True

            del self.raw_input, self.raw_out
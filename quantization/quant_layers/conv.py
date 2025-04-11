import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.pytorch_utils import Conv1D
from quantization.utils_quant import quantize_to_fp, mse_range_estimator


class MinMaxQuantConv1d(nn.Conv1d):
    """
    MinMax quantize weight and output for 1D convolution
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros', mode='raw', w_bit=8, a_bit=8, bias_bit=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.n_calibration_steps = 2
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.bias_bit = bias_bit
        assert bias_bit is None, "No support bias bit now"
        self.w_interval = None
        self.a_interval = None
        self.bias_interval = None
        self.raw_input = None
        self.raw_out = None
        self.metric = None
        self.next_nodes = []
        self.w_qmax = 2 ** (self.w_bit - 1)
        self.a_qmax = 2 ** (self.a_bit - 1)
        # self.bias_qmax=2**(self.bias_bit-1)

    def forward(self, x):
        if self.mode == 'raw':
            out = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
        elif self.mode == "calibration_step1":
            out = self.calibration_step1(x)
        elif self.mode == "calibration_step2":
            out = self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out

    def quant_weight_bias(self):
        w = (self.weight / self.w_interval).round_().clamp_(-self.w_qmax, self.w_qmax - 1)
        w_sim = w.mul_(self.w_interval)
        if self.bias is not None:
            return w_sim, self.bias
            # bias=(self.bias/self.bias_interval).round_().clamp_(-self.bias_qmax,self.bias_qmax-1)
            # bias_sim=bias*self.bias_interval
            # return w_sim,bias_sim
        else:
            return w_sim, None

    def quant_input(self, x):
        x_sim = (x / self.a_interval).round_().clamp_(-self.a_qmax, self.a_qmax - 1)
        x_sim.mul_(self.a_interval)
        return x_sim

    def quant_forward(self, x):
        assert self.calibrated is not None, f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.conv1d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out

    def calibration_step1(self, x):
        # step1: collection the FP32 values
        out = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.raw_input = x.cpu().detach()
        self.raw_out = out.cpu().detach()
        return out

    def calibration_step2(self, x):
        # step2: search for the best S^w and S^a of each layer
        self.w_interval = (self.weight.data.abs().max() / (self.w_qmax - 0.5)).detach()
        self.a_interval = (x.abs().max() / (self.a_qmax - 0.5)).detach()
        self.calibrated = True
        out = self.quant_forward(x)
        return out


class MinMaxQuantConv2d(nn.Conv2d):
    """
    MinMax quantize weight and output
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros', mode='raw', w_bit=8, a_bit=8, bias_bit=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.n_calibration_steps = 2
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.bias_bit = bias_bit
        assert bias_bit is None, "No support bias bit now"
        self.w_interval = None
        self.a_interval = None
        self.bias_interval = None
        self.raw_input = None
        self.raw_out = None
        self.metric = None
        self.next_nodes = []
        self.w_qmax = 2 ** (self.w_bit - 1)
        self.a_qmax = 2 ** (self.a_bit - 1)
        # self.bias_qmax=2**(self.bias_bit-1)

    def forward(self, x):
        if self.mode == 'raw':
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
        elif self.mode == "calibration_step1":
            out = self.calibration_step1(x)
        elif self.mode == "calibration_step2":
            out = self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out

    def quant_weight_bias(self):
        w = (self.weight / self.w_interval).round_().clamp_(-self.w_qmax, self.w_qmax - 1)
        w_sim = w.mul_(self.w_interval)
        if self.bias is not None:
            return w_sim, self.bias
            # bias=(self.bias/self.bias_interval).round_().clamp_(-self.bias_qmax,self.bias_qmax-1)
            # bias_sim=bias*self.bias_interval
            # return w_sim,bias_sim
        else:
            return w_sim, None

    def quant_input(self, x):
        x_sim = (x / self.a_interval).round_().clamp_(-self.a_qmax, self.a_qmax - 1)
        x_sim.mul_(self.a_interval)
        return x_sim

    def quant_forward(self, x):
        assert self.calibrated is not None, f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out

    def calibration_step1(self, x):
        # step1: collection the FP32 values
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.raw_input = x.cpu().detach()
        self.raw_out = out.cpu().detach()
        return out

    def calibration_step2(self, x):
        # step2: search for the best S^w and S^a of each layer
        self.w_interval = (self.weight.data.abs().max() / (self.w_qmax - 0.5)).detach()
        self.a_interval = (x.abs().max() / (self.a_qmax - 0.5)).detach()
        self.calibrated = True
        out = self.quant_forward(x)
        return out


class FPQuantConv1d(Conv1D):
    def __init__(self, nf: int,
                 nx: int,
                 mode='raw',
                 n_bits=8,
                 sign_bit=1,
                 exponent=4,
                 em_mx=True,
                 bias_mx=True,
                 bias_bit=None,
                 metric="L2_norm",
                 per_channel=True
                 ):
        super().__init__(nf, nx)
        self.bias_bit = bias_bit
        self.mode = mode
        self.n_bits = n_bits
        self.sign_bit = sign_bit
        self.exponent = exponent
        mantissa = self.n_bits - self.sign_bit - self.exponent
        self.mantissa = torch.Tensor([float(mantissa)])
        self.em_mx = em_mx
        self.bias_mx = bias_mx
        self.metric = metric
        self.per_channel = per_channel
        self.w_maxval = None
        self.a_maxval = None
        self.raw_grad = None
        self.raw_input = None
        self.raw_out = None
        self.calibrated = None
        self.gpt2_conv1d = True

    def forward(self, x):
        if self.mode == 'raw':
            out = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(*x.size()[:-1], self.nf)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
        elif self.mode == "calibration_step2":
            self.calibration_step2()
            out = None
        else:
            raise NotImplementedError
        return out

    def quant_weight_bias(self):
        w_sim = quantize_to_fp(self.weight, self.n_bits, self.w_maxval, self.mantissa, self.sign_bit, self.gpt2_conv1d)

        if self.bias is not None:
            return w_sim, self.bias
        else:
            return w_sim, None

    def quant_input(self, x):
        x_sim = quantize_to_fp(x, self.n_bits, self.a_maxval, self.mantissa, self.sign_bit)

        return x_sim

    def quant_forward(self, x):
        assert self.calibrated is not None, f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = torch.addmm(bias_sim, x_sim.view(-1, x_sim.size(-1)), w_sim).view(*x_sim.size()[:-1], self.nf)

        return out

    def calibration_step2(self):

        with torch.no_grad():

            self.raw_input = self.raw_input.to(self.weight.device)

            self.raw_out = self.raw_out.to(self.weight.device)

            if self.em_mx:

                optimal_mantissa = None

                min_mse = float('inf')

                for mb in range(1, self.n_bits):

                    current_mantissa = torch.tensor([mb], dtype=torch.float32).to(self.weight.device)

                    self.mantissa = current_mantissa

                    self.w_maxval = mse_range_estimator(self.weight, self.n_bits, self.sign_bit, current_mantissa,
                                                        self.per_channel, self.gpt2_conv1d)

                    self.a_maxval = mse_range_estimator(self.raw_input, self.n_bits, self.sign_bit, current_mantissa)

                    w_sim = quantize_to_fp(self.weight, self.n_bits, self.w_maxval, current_mantissa, self.sign_bit,
                                           self.gpt2_conv1d)

                    x_sim = quantize_to_fp(self.raw_input, self.n_bits, self.a_maxval, current_mantissa, self.sign_bit)

                    out = torch.addmm(self.bias, x_sim.view(-1, x_sim.size(-1)), w_sim).view(*x_sim.size()[:-1],
                                                                                             self.nf)

                    mse = torch.mean((self.raw_out.to(out.device) - out) ** 2)

                    if mse < min_mse:
                        min_mse = mse

                        optimal_mantissa = current_mantissa

                self.mantissa = optimal_mantissa

                self.exponent = self.n_bits - self.mantissa - self.sign_bit

                self.w_maxval = mse_range_estimator(self.weight, self.n_bits, self.sign_bit, self.mantissa,
                                                    self.per_channel, self.gpt2_conv1d)

                self.a_maxval = mse_range_estimator(self.raw_input, self.n_bits, self.sign_bit, self.mantissa)

            else:

                if self.bias_mx:

                    self.w_maxval = mse_range_estimator(self.weight, self.n_bits, self.sign_bit, self.mantissa,
                                                        self.per_channel, self.gpt2_conv1d)

                    self.a_maxval = mse_range_estimator(self.raw_input, self.n_bits, self.sign_bit, self.mantissa)

                else:

                    default_maxval = (2 - 2 ** (-self.mantissa)) * 2 ** (2 ** (self.exponent - 1) - 1)

                    self.w_maxval = default_maxval

                    self.a_maxval = default_maxval

            self.calibrated = True

            del self.raw_input, self.raw_out, self.raw_grad

    def save_quant_state(self, path):
        state_dict = {
            "n_bits": self.n_bits,
            "sign_bit": self.sign_bit,
            "exponent": self.exponent,
            "mantissa": self.mantissa,
            "w_maxval": self.w_maxval,
            "a_maxval": self.a_maxval,
            "calibrated": self.calibrated
        }
        torch.save(state_dict, path)


class FPQuantConv2d(MinMaxQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode: object = 'raw',
                 n_bits=8,
                 sign_bit=1,
                 exponent=4,
                 em_mx=True,
                 bias_mx=True,
                 metric="L2_norm",
                 per_channel=True,
                 bias_bit=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.bias_bit = bias_bit
        self.mode = mode
        self.n_bits = n_bits
        self.sign_bit = sign_bit
        self.exponent = exponent
        self.mantissa = self.n_bits - self.sign_bit - self.exponent
        self.mantissa = torch.Tensor([float(self.mantissa)])
        self.em_mx = em_mx
        self.bias_mx = bias_mx
        self.metric = metric
        self.per_channel = per_channel
        self.w_maxval = None
        self.a_maxval = None
        self.raw_grad = None
        self.raw_input = None
        self.raw_out = None
        self.calibrated = None

    def forward(self, x):
        if self.mode == 'raw':
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
        elif self.mode == "calibration_step2":
            self.calibration_step2()
            out = None
        else:
            raise NotImplementedError
        return out

    def quant_weight_bias(self):
        w_sim = quantize_to_fp(self.weight, self.n_bits, self.w_maxval, self.mantissa, self.sign_bit)
        return w_sim, self.bias

    def quant_input(self, x):
        return quantize_to_fp(x, self.n_bits, self.a_maxval, self.mantissa, self.sign_bit)

    def quant_forward(self, x):
        assert self.calibrated is not None, f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)

        return out

    def calibration_step2(self):

        with torch.no_grad():

            self.raw_input = self.raw_input.to(self.weight.device)

            self.raw_out = self.raw_out.to(self.weight.device)

            if self.em_mx:

                optimal_mantissa = None

                min_mse = float('inf')

                for mb in range(1, self.n_bits):

                    current_mantissa = torch.tensor([mb], dtype=torch.float32).to(self.weight.device)

                    self.mantissa = current_mantissa

                    self.w_maxval = mse_range_estimator(self.weight, self.n_bits, self.sign_bit, current_mantissa,
                                                        self.per_channel)

                    self.a_maxval = mse_range_estimator(self.raw_input, self.n_bits, self.sign_bit, current_mantissa)

                    w_sim = quantize_to_fp(self.weight, self.n_bits, self.w_maxval, current_mantissa, self.sign_bit)

                    x_sim = quantize_to_fp(self.raw_input, self.n_bits, self.a_maxval, current_mantissa, self.sign_bit)

                    out = F.conv2d(x_sim, w_sim, self.bias, self.stride, self.padding, self.dilation, self.groups)

                    mse = torch.mean((self.raw_out.to(out.device) - out) ** 2)

                    if mse < min_mse:
                        min_mse = mse

                        optimal_mantissa = current_mantissa

                self.mantissa = optimal_mantissa

                self.exponent = self.n_bits - self.mantissa - self.sign_bit

                self.w_maxval = mse_range_estimator(self.weight, self.n_bits, self.sign_bit, self.mantissa,
                                                    self.per_channel)

                self.a_maxval = mse_range_estimator(self.raw_input, self.n_bits, self.sign_bit, self.mantissa)

            else:

                if self.bias_mx:

                    self.w_maxval = mse_range_estimator(self.weight, self.n_bits, self.sign_bit, self.mantissa,
                                                        self.per_channel)

                    self.a_maxval = mse_range_estimator(self.raw_input, self.n_bits, self.sign_bit, self.mantissa)

                else:

                    default_maxval = (2 - 2 ** (-self.mantissa)) * 2 ** (2 ** (self.exponent - 1) - 1)

                    self.w_maxval = default_maxval

                    self.a_maxval = default_maxval

            self.calibrated = True

            del self.raw_input, self.raw_out, self.raw_grad

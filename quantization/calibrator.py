import sys
import torch
import json
import os

from transformers.pytorch_utils import Conv1D
from quantization.quant_layers import MinMaxQuantConv1d, MinMaxQuantConv2d
from quantization.quant_layers import MinMaxQuantLinear
from quantization.quant_layers import MinMaxQuantMatMul
from quantization.quant_layers import MinMaxQuantBatchNorm2d, MinMaxQuantLayerNorm
from tqdm import tqdm


class QuantCalibrator():
    """
    Modularization of quant calib.

    Notice:
    all quant modules has method "calibration_step1" that should only store raw inputs and outputs
    all quant modules has method "calibration_step2" that should only quantize its intervals
    And we assume we could feed in all calibration data in one batch, without backward propagations

    sequential calibration is memory-friendly, while parallel calibration may consume
    hundreds of GB of memory.
    """

    def __init__(self, net, wrapped_modules, calib_loader, sequential=True):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.sequential = sequential
        self.calibrated = False

    def sequential_quant_calib(self):
        """
        A quick implementation of calibration.
        Assume calibration dataset could be fed at once.
        """
        # run calibration
        n_calibration_steps = 2
        for step in range(n_calibration_steps):
            print(f"Start calibration step={step + 1}")
            for name, module in self.wrapped_modules.items():
                # corner cases for calibrated modules
                if hasattr(module, "calibrated"):
                    if step == 1:
                        module.mode = "raw"
                    elif step == 2:
                        module.mode = "quant_forward"
                else:
                    module.mode = f'calibration_step{step + 1}'
            with torch.no_grad():
                for inp, target in self.calib_loader:
                    inp = inp.cuda()
                    self.net(inp)

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = 'quant_forward'
        torch.cuda.empty_cache()  # memory footprint cleanup
        print("sequential calibration finished")

    def parallel_quant_calib(self):
        """
        A quick implementation of parallel quant calib
        Assume calibration dataset could be fed at once, and memory could hold all raw inputs/outs
        """
        # calibration step1: collect raw data
        print(f"Start calibration step=1")
        for name, module in self.wrapped_modules.items():
            # corner cases for calibrated modules
            if hasattr(module, "calibrated"):
                module.mode = "raw"
            else:
                module.mode = f'calibration_step1'
        with torch.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.cuda()
                self.net(inp)
        # calibration step2: each module run calibration with collected raw data
        for name, module in self.wrapped_modules.items():
            if hasattr(module, "calibrated"):
                continue
            else:
                module.mode = f"calibration_step2"
                with torch.no_grad():
                    if isinstance(module, MinMaxQuantLinear):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantConv2d):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantMatMul):
                        module.forward(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                    torch.cuda.empty_cache()

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = 'quant_forward'
        torch.cuda.empty_cache()  # memory footprint cleanup
        print("calibration finished")

    def quant_calib(self):
        calib_layers = []
        for name, module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        if self.sequential:
            self.sequential_quant_calib()
        else:
            self.parallel_quant_calib()
        self.calibrated = True

    def batching_quant_calib(self):
        calib_layers = []
        for name, module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start calibration")

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))

            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0, self.calib_loader.batch_size, self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st + self.batch_size].cuda()
                    self.net(inp_)
                del inp, target
                torch.cuda.empty_cache()

            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            for hook in hooks:
                hook.remove()
       # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()

            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"

        print("calibration finished")


# def grad_hook(module, grad_input, grad_output):
#     if module.raw_grad is None:
#         module.raw_grad = []
#     module.raw_grad.append(grad_output[0].cpu().detach())  # that's a tuple!


def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())


def conv1d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())


def conv2d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())


def matmul_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = [[], []]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].cpu().detach())
    module.raw_input[1].append(input[1].cpu().detach())
    module.raw_out.append(output.cpu().detach())


def batchnorm2d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())


def layernorm_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())


class HessianQuantCalibrator(QuantCalibrator):

    def __init__(self, net, tokenizer, wrapped_modules, calib_loader, sequential=False, batch_size=1,
                 dataset="imagenet"):

        super().__init__(net, wrapped_modules, calib_loader, sequential=sequential)

        self.tokenizer = tokenizer

        self.batch_size = batch_size

        self.dataset = dataset

        self._hook_mapping = {

            MinMaxQuantLinear: linear_forward_hook,

            MinMaxQuantConv1d: conv1d_forward_hook,

            Conv1D: conv1d_forward_hook,

            MinMaxQuantConv2d: conv2d_forward_hook,

            MinMaxQuantMatMul: matmul_forward_hook,

            MinMaxQuantBatchNorm2d: batchnorm2d_forward_hook,

            MinMaxQuantLayerNorm: layernorm_forward_hook,

        }

    def save_all_quant_params(self, save_path):
        quant_params = {}
        for name, module in self.wrapped_modules.items():
            params = {
                'n_bits': getattr(module, 'n_bits', None),
                'sign_bit': getattr(module, 'sign_bit', None),
                'exponent': getattr(module, 'exponent', None),
                'mantissa': getattr(module, 'mantissa', None),
                'w_maxval': getattr(module, 'w_maxval', None),
                'a_maxval': getattr(module, 'a_maxval', None),
                'A_maxval': getattr(module, 'A_maxval', None),
                'B_maxval': getattr(module, 'B_maxval', None),
                'calibrated': getattr(module, 'calibrated', None)
            }
            params = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in params.items() if v is not None}
            quant_params[name] = params

        with open(save_path, 'w') as f:
            json.dump(quant_params, f, indent=4)

    def load_all_quant_params(self, save_path):
        quant_params = torch.load(save_path)
        for name, module in self.wrapped_modules.items():
            if name in quant_params:
                params = quant_params[name]
                for param_name, param_value in params.items():
                    setattr(module, param_name, param_value)
                module.mode = "quant_forward"

    def batching_quant_calib(self, config):

        if config.QUANT.LOAD_QUANT_PARAMS:
            save_path = config.QUANT.QUANT_PARAMS_PATH
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    quant_params = json.load(f)

        else:
            calib_loader = self.calib_loader
            dataset = self.dataset
            wrapped_modules = self.wrapped_modules

            for name, module in tqdm(wrapped_modules.items(), desc="L2_Norm"):
                module_type = type(module)
                hooks = []
                hook_func = None

                for cls in module_type.mro():
                    if cls in self._hook_mapping:
                        hook_func = self._hook_mapping[cls]
                        break

                if hook_func:
                    hooks.append(module.register_forward_hook(hook_func))

                with torch.no_grad():

                    if dataset == "imagenet":
                        for batch in calib_loader:
                            inputs = batch[0].cuda()
                            _ = self.net(inputs)

                    elif dataset == "wikitext2":
                        encoding = self.tokenizer("\n\n".join(calib_loader["text"]), return_tensors="pt")
                        seq_len = encoding.input_ids.shape[1]
                        C = 1024  # context length
                        pad_token_id = self.tokenizer.pad_token_id  # 获取填充token的ID
                        for begin_loc in range(0, seq_len, C):
                            end_loc = min(begin_loc + C, seq_len)
                            current_slice = encoding.input_ids[:, begin_loc:end_loc]

                            # 计算需要填充的长度
                            pad_length = C - (end_loc - begin_loc)
                            if pad_length > 0:

                                padding = torch.full((current_slice.shape[0], pad_length), pad_token_id,
                                                     dtype=current_slice.dtype, device=current_slice.device)
                                current_slice = torch.cat([current_slice, padding], dim=1)
                            input_ids = current_slice.to('cuda')

                            with torch.no_grad():
                                _ = self.net(input_ids)

                if any(issubclass(module_type, cls) for cls in
                       (MinMaxQuantLinear, MinMaxQuantConv1d, Conv1D, MinMaxQuantConv2d, MinMaxQuantBatchNorm2d,
                        MinMaxQuantLayerNorm)):
                    module.raw_input = torch.cat(module.raw_input, dim=0)
                    module.raw_out = torch.cat(module.raw_out, dim=0)

                elif issubclass(module_type, MinMaxQuantMatMul):
                    module.raw_input = [torch.cat(tensors, dim=0) for tensors in module.raw_input]
                    module.raw_out = torch.cat(module.raw_out, dim=0)

                for hook in hooks:
                    hook.remove()

                module.calibration_step2()
                torch.cuda.empty_cache()
                module.mode = "quant_forward" if self.sequential else "raw"

                self.save_all_quant_params(config.QUANT.QUANT_PARAMS_PATH)

        for name, module in wrapped_modules.items():
            module.mode = "quant_forward"
            print(f"Module {name}: n_bits={module.n_bits}, exponent={module.n_bits - module.mantissa - module.sign_bit}, mantissa={module.mantissa}")
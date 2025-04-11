import sys

from quantization.quant_layers import (FPQuantConv1d, FPQuantConv2d, FPQuantMatMul, FPQuantBatchNorm2d, FPQuantLinear,
                                       FPQuantLayerNorm)


def wrap_modules_in_net(model, config, module_config_map=None):
    if module_config_map is None:
        module_config_map = None  # 不初始化为空字典，保持为None
    else:
        # 允许用户传入空字典，但保持逻辑一致性
        pass

    model_type = config.MODEL.TYPE
    wrapped_modules = {}
    module_dict = {}

    # 原有模块类型映射逻辑保持不变
    if model_type == 'deit':
        module_types = {
            'qkv': 'qlinear_qkv', 'proj': 'qlinear_proj',
            'fc1': 'qlinear_MLP_1', 'fc2': 'qlinear_MLP_2',
            'head': 'qlinear_classifier',
            'matmul1': 'qmatmul_qk', 'matmul2': 'qmatmul_scorev'
        }
    elif model_type == 'resnet':
        module_types = {
            'conv1': 'qconv2d_embed',
            '**.*.conv1': 'qconv2d_conv1',
            '**.conv2': 'qconv2d_conv2',
            '**.conv3': 'qconv2d_conv3',
            '**.downsample.0': 'qconv2d_downsample',
            'fc': 'qlinear_head',
        }
    elif model_type == 'gpt2':
        module_types = {
            '**.attn.c_attn': 'qconv1d_qkv', '**.attn.c_proj': 'qconv1d_proj',
            '**.mlp.c_fc': 'qconv1d_fc1', '**.mlp.c_proj': 'qconv1d_fc2',
            '**.ln_1': 'qlayernorm_ln1', '**.ln_2': 'qlayernorm_ln2',
            '**.ln_f': 'qlayernorm_lnf',
            '**.matmul1': 'qmatmul_qk', '**.matmul2': 'qmatmul_scorev'
        }
    elif model_type == 'dnn_vad':
        module_types = {
            'fc1': 'qlinear_fc1', 'fc2': 'qlinear_fc2', 'fc3': 'qlinear_fc3', 'last': 'qlinear_fc4',
        }
    else:
        raise NotImplementedError(f"Model not supported!")

        # 根据 module_config_map 是否存在决定是否排序模式
    if module_config_map is not None:
        sorted_patterns = sorted(
            module_config_map.keys(),
            key=lambda x: (-len(x.split('.')), x)
        )
    else:
        sorted_patterns = []

    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        father_name = name[:idx] if idx != -1 else ''
        father_module = module_dict.get(father_name, None)
        if not father_module:
            raise RuntimeError(f"父模块 {father_name} 未找到")

        # 检查模块类型是否匹配（如卷积、全连接等）
        matched_type = None
        for pattern in module_types:
            if _match_pattern(name, pattern):
                matched_type = module_types[pattern]
                break
        if matched_type is None:
            continue  # 类型不匹配，直接跳过

        # 收集所有匹配的配置（优先级：更具体的模式覆盖通用模式）
        specific_config = {}
        for pattern in sorted_patterns:
            if _match_pattern(name, pattern):
                specific_config.update(module_config_map[pattern])

                # 新增逻辑：如果 module_config_map 存在且没有匹配的配置，则跳过
        if module_config_map is not None and not specific_config:
            continue

            # 创建新模块并传入配置（specific_config 可能为空）
        new_m = get_module(
            module_type=matched_type,
            config=config,
            module_config=specific_config,
            in_features=getattr(m, 'in_channels', getattr(m, 'in_features', None)),
            out_features=getattr(m, 'out_channels', getattr(m, 'out_features', None)),
            nf=getattr(m, 'nf', None),
            nx=getattr(m, 'nx', None),
            kernel_size=getattr(m, 'kernel_size', None),
            stride=getattr(m, 'stride', None),
            padding=getattr(m, 'padding', None),
            dilation=getattr(m, 'dilation', None),
            groups=getattr(m, 'groups', None),
            padding_mode=getattr(m, 'padding_mode', None),
            num_features=getattr(m, 'num_features', None),
            normalized_shape=getattr(m, 'normalized_shape', None),
            eps=getattr(m, 'eps', None),
            momentum=getattr(m, 'momentum', None),
            affine=getattr(m, 'affine', None),
            elementwise_affine=getattr(m, 'elementwise_affine', None)
        )

        # 复制权重和偏置

        if hasattr(m, 'weight'):
            new_m.weight.data = m.weight.data

        if hasattr(m, 'bias'):
            new_m.bias = m.bias

            # 替换模块

        setattr(father_module, name.split('.')[-1], new_m)
        wrapped_modules[name] = new_m

    return wrapped_modules


def _match_pattern(name, pattern):
    """支持多层级通配符 `**`，例如 `**.c_attn` 可匹配任意层级的 `c_attn` 模块"""
    parts_pattern = pattern.split('.')
    parts_name = name.split('.')

    i = j = 0
    len_p, len_n = len(parts_pattern), len(parts_name)

    while i < len_p and j < len_n:
        p = parts_pattern[i]
        n = parts_name[j]

        if p == '**':
            # 处理多层级通配符
            if i + 1 < len_p:
                # 检查剩余模式是否能匹配后续路径的任意位置
                remaining_pattern = '.'.join(parts_pattern[i+1:])
                remaining_name = '.'.join(parts_name[j:])
                if _match_pattern(remaining_name, remaining_pattern):
                    return True
                else:
                    # 尝试跳过当前层级继续匹配
                    j += 1
                    continue
            else:
                # 如果剩余模式只有 **，则匹配成功
                return True
        elif p == '*':
            # 单层级通配符
            i += 1
            j += 1
        elif p == n:
            i += 1
            j += 1
        else:
            return False

    # 检查是否完全匹配
    return i == len_p and (j == len_n or (i == len_p and p == '**'))


def get_module(module_type, config, module_config=None, **kwargs):
    """根据模块类型创建量化模块，优先使用module_config中的配置"""
    def get_param(key, default):
        return module_config.get(key, getattr(config.QUANT, key, default)) if module_config else getattr(config.QUANT, key, default)

    if "qconv1d" in module_type:
        return FPQuantConv1d(
            nf=kwargs['nf'],
            nx=kwargs['nx'],
            n_bits=get_param('N_BITS', config.QUANT.N_BITS),
            sign_bit=get_param('SIGN_BIT', config.QUANT.SIGN_BIT),
            exponent=get_param('EXPONENT', config.QUANT.EXPONENT),
            em_mx=get_param('EM_MX', config.QUANT.EM_MX),
            bias_mx=get_param('BIAS_MX', config.QUANT.BIAS_MX),
            metric=get_param('METRIC', config.QUANT.METRIC),
            per_channel=get_param('PER_CHANNEL', config.QUANT.PER_CHANNEL)
        )
    elif "qconv2d" in module_type:
        return FPQuantConv2d(
            in_channels=kwargs['in_features'],
            out_channels=kwargs['out_features'],
            kernel_size=kwargs['kernel_size'],
            stride=kwargs['stride'],
            padding=kwargs['padding'],
            dilation=kwargs['dilation'],
            groups=kwargs['groups'],
            padding_mode=kwargs['padding_mode'],
            n_bits=get_param('N_BITS', config.QUANT.N_BITS),
            sign_bit=get_param('SIGN_BIT', config.QUANT.SIGN_BIT),
            exponent=get_param('EXPONENT', config.QUANT.EXPONENT),
            em_mx=get_param('EM_MX', config.QUANT.EM_MX),
            bias_mx=get_param('BIAS_MX', config.QUANT.BIAS_MX),
            metric=get_param('METRIC', config.QUANT.METRIC),
            per_channel=get_param('PER_CHANNEL', config.QUANT.PER_CHANNEL)
        )
    elif "qlinear" in module_type:
        return FPQuantLinear(
            in_features=kwargs['in_features'],
            out_features=kwargs['out_features'],
            n_bits=get_param('N_BITS', config.QUANT.N_BITS),
            sign_bit=get_param('SIGN_BIT', config.QUANT.SIGN_BIT),
            exponent=get_param('EXPONENT', config.QUANT.EXPONENT),
            em_mx=get_param('EM_MX', config.QUANT.EM_MX),
            bias_mx=get_param('BIAS_MX', config.QUANT.BIAS_MX),
            metric=get_param('METRIC', config.QUANT.METRIC),
            per_channel=get_param('PER_CHANNEL', config.QUANT.PER_CHANNEL)
        )
    elif "qmatmul" in module_type:
        return FPQuantMatMul(
            n_bits=get_param('N_BITS', config.QUANT.N_BITS),
            sign_bit=get_param('SIGN_BIT', config.QUANT.SIGN_BIT),
            exponent=get_param('EXPONENT', config.QUANT.EXPONENT),
            em_mx=get_param('EM_MX', config.QUANT.EM_MX),
            bias_mx=get_param('BIAS_MX', config.QUANT.BIAS_MX),
            metric=get_param('METRIC', config.QUANT.METRIC)
        )
    elif "qbatchnorm" in module_type:
        return FPQuantBatchNorm2d(
            num_features=kwargs['num_features'],
            eps=kwargs['eps'],
            momentum=kwargs['momentum'],
            affine=kwargs['affine'],
            n_bits = get_param('N_BITS', config.QUANT.N_BITS),
            sign_bit = get_param('SIGN_BIT', config.QUANT.SIGN_BIT),
            exponent = get_param('EXPONENT', config.QUANT.EXPONENT),
            metric = get_param('METRIC', config.QUANT.METRIC)
        )
    elif "qlayernorm" in module_type:
        return FPQuantLayerNorm(
            normalized_shape=kwargs['normalized_shape'],
            eps=kwargs['eps'],
            elementwise_affine=kwargs['elementwise_affine'],
            n_bits=get_param('N_BITS', config.QUANT.N_BITS),
            sign_bit=get_param('SIGN_BIT', config.QUANT.SIGN_BIT),
            exponent=get_param('EXPONENT', config.QUANT.EXPONENT),
            em_mx=get_param('EM_MX', config.QUANT.EM_MX),
            bias_mx=get_param('BIAS_MX', config.QUANT.BIAS_MX),
            metric=get_param('METRIC', config.QUANT.METRIC)
        )
    else:
        raise ValueError(f"Unsupported module type: {module_type}")
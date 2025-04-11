import sys
import os
import copy
import json

import torch
import subprocess
import pickle
import numpy as np


def add_common_args(parser):
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--only-cpu', action='store_true',
                        help='Perform evaluation on CPU')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    parser.add_argument('--use-sync-bn', action='store_true',
                        default=False, help='sync bn')
    parser.add_argument('--use-wandb', action='store_true',
                        default=False, help='use wandb to record log')


def load_json_config(config_path):
    if not config_path:
        return None
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('module_config_map', {})


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, grad_scaler_enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=grad_scaler_enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None and clip_grad > 0.0:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d)
                                for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    params = checkpoint['model']
    now_model_state = model.state_dict()
    mnames = ['head.weight', 'head.bias']  # (cls, 1024), (cls, )
    if mnames[-1] in params:
        ckpt_head_bias = params[mnames[-1]]
        now_model_bias = now_model_state[mnames[-1]]
        if ckpt_head_bias.shape != now_model_bias.shape:
            num_classes = 1000

            if len(ckpt_head_bias) == 21841 and len(now_model_bias) == num_classes:
                logger.info("Convert checkpoint from 21841 to 1k")
                # convert 22kto1k
                fname = './imagenet_1kto22k.txt'
                with open(fname) as fin:
                    mapping = torch.Tensor(
                        list(map(int, fin.readlines()))).to(torch.long)
                for name in mnames:
                    v = params[name]
                    shape = list(v.shape)
                    shape[0] = num_classes
                    mean_v = v[mapping[mapping != -1]].mean(0, keepdim=True)
                    v = torch.cat([v, mean_v], 0)
                    v = v[mapping]
                    params[name] = v

    msg = model.load_state_dict(params, strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE:
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            logger.info(
                f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            if 'max_accuracy' in checkpoint:
                max_accuracy = checkpoint['max_accuracy']

        if 'epoch' in checkpoint:
            config.defrost()
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
            config.freeze()

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        # local
        self._val = 0
        self._sum = 0
        self._count = 0
        # global
        self._history_avg = 0
        self._history_count = 0
        self._avg = None

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = None

    @property
    def val(self):
        return self._val

    @property
    def count(self):
        return self._count + self._history_count

    @property
    def avg(self):
        if self._avg is None:
            # compute avg
            r = self._history_count / max(1, self._history_count + self._count)
            _avg = self._sum / max(1, self._count)
            self._avg = r * self._history_avg + (1.0 - r) * _avg
        return self._avg


def load_pretrained(config, model, logger):
    logger.info(
        f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    model_state_dict = model.state_dict()

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model_state_dict[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2).permute(1, 0)

    # bicubic interpolate attention_biases if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "attention_biases" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model_state_dict[k]
        nH1, L1 = relative_position_bias_table_pretrained.size()
        nH2, L2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                    -1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                    0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                    0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                    1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-21841 weight to ImageNet-1K ......")
            map22kto1k_path = f'./imagenet_1kto22k.txt'
            with open(map22kto1k_path) as fin:
                mapping = torch.Tensor(
                    list(map(int, fin.readlines()))).to(torch.long)
            for name in ['head.weight', 'head.bias']:
                v = state_dict[name]
                mean_v = v[mapping[mapping != -1]].mean(0, keepdim=True)
                v = torch.cat([v, mean_v], 0)
                v = v[mapping]
                state_dict[name] = v
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(
                f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def is_main_process():
    return True


def run_cmd(cmd, default=None):
    try:
        return subprocess.check_output(cmd.split(), universal_newlines=True).strip()
    except:
        if default is None:
            raise
        return default





class LRSchedulerWrapper:
    """
    LR Scheduler Wrapper

    This class attaches the pre-hook on the `step` functions (including `step`, `step_update`, `step_frac`) of a lr scheduler.
    When `step` functions are called, the learning rates of all layers are updated.

    Usage:
    ```
        lr_scheduler = LRSchedulerWrapper(lr_scheduler, optimizer)
    ```
    """

    def __init__(self, lr_scheduler, optimizer):
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

    def step(self, epoch):
        self.lr_scheduler.step(epoch)
        self.update_lr()

    def step_update(self, it):
        self.lr_scheduler.step_update(it)
        self.update_lr()

    def step_frac(self, frac):
        if hasattr(self.lr_scheduler, 'step_frac'):
            self.lr_scheduler.step_frac(frac)
            self.update_lr()

    def update_lr(self):
        param_groups = self.optimizer.param_groups
        for group in param_groups:
            if 'lr_scale' in group:
                group['lr'] *= group['lr_scale']

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, *args, **kwargs):
        self.lr_scheduler.load_state_dict(*args, **kwargs)


def divide_param_groups_by_lr_scale(param_groups):
    """
    Divide parameters with different lr scale into different groups.

    Inputs
    ------
    param_groups: a list of dict of torch.nn.Parameter
    ```
    # example:
    param1.lr_scale = param2.lr_scale = param3.lr_scale = 0.6
    param4.lr_scale = param5.lr_scale = param6.lr_scale = 0.3
    param_groups = [{'params': [param1, param2, param4]},
                    {'params': [param3, param5, param6], 'weight_decay': 0.}]

    param_groups = divide_param_groups_by_lr_scale(param_groups)
    ```

    Outputs
    -------
    new_param_groups: a list of dict containing the key `lr_scale`
    ```
    param_groups = [
        {'params': [param1, param2], 'lr_scale': 0.6},
        {'params': [param3], 'weight_decay': 0., 'lr_scale': 0.6}
        {'params': [param4], 'lr_scale': 0.3},
        {'params': [param5, param6], 'weight_decay': 0., 'lr_scale': 0.3}
    ]
    ```
    """
    new_groups = []
    for group in param_groups:
        params = group.pop('params')

        '''
        divide parameters to different groups by lr_scale
        '''
        lr_scale_groups = dict()
        for p in params:
            lr_scale = getattr(p, 'lr_scale', 1.0)

            # create a list if not existed
            if lr_scale not in lr_scale_groups:
                lr_scale_groups[lr_scale] = list()

            # add the parameter with `lr_scale` into the specific group.
            lr_scale_groups[lr_scale].append(p)

        for lr_scale, params in lr_scale_groups.items():
            # copy other parameter information like `weight_decay`
            new_group = copy.copy(group)
            new_group['params'] = params
            new_group['lr_scale'] = lr_scale
            new_groups.append(new_group)
    return new_groups


def set_weight_decay(model):
    skip_list = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip_list = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def global_feature_normalize(feature, train_mean, train_std):
    mu = train_mean
    sigma = train_std
    return (feature-mu)/sigma

def calc_global_mean_std(mean_path, std_path, train_DB):
    try:
        mean = np.loadtxt(mean_path)
        mean = np.expand_dims(mean,0)
        std = np.loadtxt(std_path)
        #print("The global mean and std of train DB are loaded from saved files")
        return mean,std
    except:
        mean, std = train_mean_std(train_DB)
        np.savetxt(mean_path, mean, delimiter='\n')
        np.savetxt(std_path, std, delimiter='\n')
        print("The global mean and std of train DB are saved")
        return mean,std

def train_mean_std(train_DB):
    print("Start to calculate the global mean and std of train DB")
    """ Calculate the global mean and std of train DB """
    n_files = len(train_DB)
    train_mean = 0.
    train_std = 0.
    n_frames = 0.
    # Calculate the global mean of train DB
    for i in range(n_files):
        filename = train_DB['filename'][i]
        label_path = train_DB['label_path'][i]
        inputs, targets = read_MRCG(filename, label_path)  # input shape : (n_frames, n_dim)
        temp_n_frames = len(inputs)  # number of frames
        train_mean += np.sum(inputs, axis=0, keepdims=1)  # shape : (1, n_dim)
        n_frames += temp_n_frames
    train_mean = train_mean / n_frames
    # Calculate the global std of train DB
    for i in range(n_files):
        filename = train_DB['filename'][i]
        label_path = train_DB['label_path'][i]
        inputs, targets = read_MRCG(filename, label_path)  # input shape : (n_frames, n_dim)
        deviation = np.sum((inputs - train_mean) ** 2, axis=0, keepdims=1)  # shape : (1, n_dim)
        train_std += deviation
    train_std = train_std / (n_frames - 1)
    train_std = np.sqrt(train_std)
    return train_mean, train_std

def get_global_mean_std(config, train_DB):
    if not os.path.exists(config.DATA.VAD.MEAN_STD_PATH):
        os.makedirs(config.DATA.VAD.MEAN_STD_PATH)
    train_mean, train_std = calc_global_mean_std(config.DATA.VAD.MEAN_PATH, config.DATA.VAD.STD_PATH, train_DB)
    return train_mean, train_std

def read_MRCG(feat_path):
    with open(feat_path, 'rb') as f:
        feat_and_label = pickle.load(f, encoding='latin1')
    feature = feat_and_label['feat']  # size : (n_frames, dim=40)
    label = feat_and_label['vad_result']

    if len(feature) != len(label):
        feature = feature[0:len(label)]

    return feature, label

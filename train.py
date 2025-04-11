import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.tensorboard as tb

from collections import defaultdict
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

from config import get_config
from models import build_model
from data import build_loader_imagenet
from lr_scheduler import build_scheduler
from optimizer import build_optimizer

from logger import create_logger
from utils import add_common_args, NativeScalerWithGradNormCount, auto_resume_helper, load_checkpoint, AverageMeter,\
                  load_pretrained, is_main_process, save_checkpoint


NORM_ITER_LEN = 100


def parse_option():
    parser = argparse.ArgumentParser(
        'Model training and evaluation script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(args, config):
    # Initialize TensorBoard Writer
    writer = None
    if is_main_process():
        writer = tb.SummaryWriter(log_dir=os.path.join(config.OUTPUT, 'tensorboard'))

    dataset_train, dataset_val, data_loader_train, data_loader_val, _, mixup_fn = build_loader_imagenet(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    if not args.only_cpu:
        model.cuda()

    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    loss_scaler = NativeScalerWithGradNormCount(grad_scaler_enabled=config.AMP_ENABLE)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)

    if config.AUG.MIXUP > 0.:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model, logger)
        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if hasattr(dataset_train, 'set_epoch'):
            dataset_train.set_epoch(epoch)

        train_one_epoch(args, config, model, criterion,
                        data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, writer)
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model,
                            max_accuracy, optimizer, lr_scheduler, loss_scaler, logger)

        acc1, acc5, loss = validate(args, config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        if is_main_process() and args.use_wandb:
            writer.add_scalar('val/acc1', acc1, epoch)
            writer.add_scalar('val/acc5', acc5, epoch)
            writer.add_scalar('val/loss', loss, epoch)
            writer.add_scalar('best_acc1', max_accuracy, epoch)

    if is_main_process() and writer:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def is_valid_grad_norm(num):
    if num is None:
        return False
    return not bool(torch.isinf(num)) and not bool(torch.isnan(num))


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, writer):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        normal_global_idx = epoch * NORM_ITER_LEN + \
            (idx * NORM_ITER_LEN // num_steps)

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            original_targets = targets.argmax(dim=1)
        else:
            original_targets = targets

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if is_valid_grad_norm(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            if is_main_process() and writer:
                writer.add_scalar('train/acc1', acc1_meter.val, normal_global_idx)
                writer.add_scalar('train/acc5', acc5_meter.val, normal_global_idx)
                writer.add_scalar('train/loss', loss_meter.val, normal_global_idx)
                writer.add_scalar('train/grad_norm', norm_meter.val, normal_global_idx)
                writer.add_scalar('train/loss_scale', scaler_meter.val, normal_global_idx)
                writer.add_scalar('train/lr', lr, normal_global_idx)

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in data_loader:
            images, targets = batch
            if not config.ONLY_CPU:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))
            loss_meter.update(loss.item(), images.size(0))

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def throughput(data_loader, model, logger):
    model.eval()

    T0, T1 = 10, 60
    images, _ = next(iter(data_loader))
    batch_size, _, H, W = images.shape
    inputs = torch.randn(batch_size, 3, H, W).cuda(non_blocking=True)

    # trace model to avoid python overhead
    model = torch.jit.trace(model, inputs)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)
    timing = []
    torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    throughput = batch_size / timing.mean().item()
    logger.info(f"batch_size {batch_size} throughput {throughput}")


layer_distributions = defaultdict(lambda: {'weights': [], 'inputs': [], 'outputs': []})


if __name__ == '__main__':
    args, config = parse_option()

    if args.only_cpu:
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)
    else:
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)
        cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        logger.info(f'TensorBoard log save to {os.path.join(config.OUTPUT, "tensorboard")} ')

    logger.info(config.dump())

    main(args, config)

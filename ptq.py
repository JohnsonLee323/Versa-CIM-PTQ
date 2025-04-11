import sys
import time
import random
import os
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from logger import create_logger
from config import get_config
from utils import add_common_args, load_json_config
from utils_eval import test_imagenet, test_ppl
from data import build_loader_imagenet, build_loader_wikitext2

from quantization.calibrator import HessianQuantCalibrator
from quantization.models_quant import get_model
from quantization.net_wrap import wrap_modules_in_net


def parse_option():
    parser = argparse.ArgumentParser(
        'Transformer-based Model evaluation and quantization script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()
    config = get_config(args)

    return args, config


def main(config):
    model, tokenizer = get_model(config)

    if not config.ONLY_CPU:
        model.cuda()

    model.eval()

    # for name, module in model.named_modules():
    #     print(name, module)

    logger.info(str(model))
    
    if config.DATA.DATASET == 'imagenet':
        _, dataset_val, _, val_loader, calib_loader, _ = build_loader_imagenet(config)
    elif config.DATA.DATASET == 'wikitext2':
        val_loader, calib_loader = build_loader_wikitext2(config)
    else:
        raise NotImplementedError(f"Dataset not supported!")

    if not config.FP32:
        logger.info(f"Starting Post-Training Quantization for {config.MODEL.NAME}")

        specific_cfg = load_json_config(config.QUANT.SPECIFIC_CFG_PATH)

        wrapped_modules = wrap_modules_in_net(model, config, specific_cfg)

        logger.info(f"Model Wrapping Completed.")

        logger.info(f"Starting Post-Training Quantization Calibration.")

        calib_start_time = time.time()
        quant_calibrator = HessianQuantCalibrator(model, tokenizer, wrapped_modules, calib_loader, sequential=False,
                                                  batch_size=4, dataset=config.DATA.DATASET)
        quant_calibrator.batching_quant_calib(config)
        calib_end_time = time.time()

        logger.info(f"Post-Training Quantization Calibration Completed.")

        logger.info(f"Calibration Time : {(calib_end_time - calib_start_time) / 60}min.")

    if config.DATA.DATASET == 'imagenet':
        acc = test_imagenet(model, val_loader, description="Model Validating on imagenet")
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc:.2f}%")
    elif config.DATA.DATASET == 'wikitext2':
        ppl = test_ppl(config, model, tokenizer, val_loader)
        logger.info(f"Perplexity on wikitext-2 test set: {ppl:.2f}")


if __name__ == '__main__':
    args, config = parse_option()

    if config.ONLY_CPU:
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

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    logger.info(config.dump())

    main(config)

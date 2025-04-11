import sys
import copy
import os

import torch
import random
import numpy as np

from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import Mixup
from timm.data import create_transform
from datasets import load_dataset
from data.VAD_Dataset import read_DB_structure, VAD_Dataset, VAD_Compose, ToTensorInput, TruncatedInputfromMRCG
from utils import read_MRCG

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


# imagenet1K------------------------------------------------------------------------------------------------------------

def create_calib_loader_imagenet(config, dataset_train, calib_size):
    indices = list(range(len(dataset_train)))
    random.shuffle(indices)
    calib_indices = indices[:calib_size]
    sampler = SubsetRandomSampler(calib_indices)

    dataset_calib = copy.deepcopy(dataset_train)

    transform = build_transform_imagenet(config, is_train=False)
    dataset_calib.transform = transform

    data_loader_calib = torch.utils.data.DataLoader(
        dataset_calib,
        batch_size=config.DATA.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY
    )
    return data_loader_calib


def build_loader_imagenet(config):
    dataset_train, _ = build_dataset_imagenet(config, True)
    print(f"Successfully built train dataset")
    dataset_val, _ = build_dataset_imagenet(config, False)
    print(f"Successfully built valid dataset")
    data_loader_calib = create_calib_loader_imagenet(config, dataset_train, config.QUANT.CALIB_SIZE)
    print(f"Successfully built calib dataset")

    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY
    )

    # setup mixup / cutmix
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, data_loader_calib, mixup_fn


def build_dataset_imagenet(config, is_train=True):
    transform = build_transform_imagenet(config, is_train)
    dataset_tar_t = TimmDatasetTar

    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        # load tar dataset
        data_dir = os.path.join(config.DATA.DATA_PATH, f'{prefix}.tar')
        if os.path.exists(data_dir):
            dataset = dataset_tar_t(data_dir, transform=transform)
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform_imagenet(config, is_train):
    resize_im = config.DATA.IMG_SIZE > 32

    # RGB: mean, std
    rgbs = dict(
        default=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        inception=(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD),
        clip=((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
    )
    mean, std = rgbs[config.DATA.MEAN_AND_STD_TYPE]

    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4)

        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(
                    config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(t)
    return transform

# imagenet1K------------------------------------------------------------------------------------------------------------


# WikiText2-------------------------------------------------------------------------------------------------------------

def build_loader_wikitext2(config):
    train_dataset_hf = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_dataset_hf = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    indices = random.sample(range(len(train_dataset_hf)), config.QUANT.CALIB_SIZE)
    calib_dataset_hf = train_dataset_hf.select(indices)

    return test_dataset_hf, calib_dataset_hf

# WikiText2-------------------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author  : Jiangsi
# @Date    : 2025-12-21
# @Update  : 2025-12-21
# @File    : test_minimal_training.py
# @Desc    : 集成测试，测试训练流程

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.run_if import RunIf


def test_train_fast_dev_run_cpu(cfg_train: DictConfig) -> None:
    """
    核心集成测试：
    - 走完整train() 入口
    - 只跑 1个 train/val/test batch
    - 验证系统是否能跑通
    """
    HydraConfig().set_config(cfg_train)

    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 1
        cfg_train.trainer.enable_progress_bar = False

    train(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """
    GPU + fast_dev_run
    """
    HydraConfig().set_config(cfg_train)

    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.devices = 1
        cfg_train.trainer.enable_progress_bar = False

    train(cfg_train)


@pytest.mark.slow
def test_train_one_epoch_cpu(cfg_train: DictConfig) -> None:
    """
    非 fast_dev_run, 真实跑 1 epoch (cpu)
    用于验证：
    - loss / metric 正常
    - dataloader / scaler / inverse scaling 不出错
    """
    HydraConfig().set_config(cfg_train)

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 1
        cfg_train.trainer.enable_progress_bar = False

    train(cfg_train)
































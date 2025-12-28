# -*- coding: utf-8 -*-
# @Author  : Jiangsi
# @Date    : 2025-09-18
# @Update  : 2025-09-18
# @File    : test_modules.py
# @Desc    : 隔离测试ELECTRICITYLitModule

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from lightning import Trainer, LightningDataModule
from typing import Any, Dict, List, Tuple

from src.models.electricity_module import (
    ELECTRICITYLitModule,
    masked_mae, masked_mape, masked_mse, masked_rmse
)
from src.models.components import CauSTG

# --- 步骤 1: 创建模拟的依赖组件
class MockNet(nn.Module):
    """一个模拟的神经网络，确保输入输出维度正确"""
    def __init__(
            self, 
            input_channels=2, 
            output_channels=1, 
            num_nodes=321,
            seq_length=12
            ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.weight = nn.Parameter(torch.randn(1))
        # self.linear = nn.Linear(input_channels*num_nodes, output_channels*num_nodes)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x 的期望输入形状: (B, C_in, N, T_in) = (B, 2, 321, 12)
        batch_size = x.shape[0]
        t_in = x.shape[3]

        # 模拟模型输出 
        # FIXED：直接给mock的数据加上requiresgrad是假阳性，loss的梯度不会回传到net.parameters(),需要增加一个self.weight
        # 主输出形状: (B, C_out, N, T_out) = (B, 1, 321, 12)
        # mock_output = torch.randn(batch_size, self.output_channels, self.num_nodes, self.seq_length, requires_grad=True)
        mock_output = self.weight * torch.ones(
            batch_size, self.output_channels, self.num_nodes, self.seq_length, device=x.device
        )

        # 模拟趋势和季节性输出
        # mock_trend = torch.randn(batch_size, self.output_channels, self.num_nodes, self.seq_length, requires_grad=True)
        # mock_season = torch.randn(batch_size, self.output_channels, self.num_nodes, self.seq_length, requires_grad=True)
        mock_trend = self.weight * torch.ones(
            batch_size, self.output_channels, self.num_nodes, self.seq_length, device=x.device
        )
        mock_season = self.weight * torch.ones(
            batch_size, self.output_channels, self.num_nodes, self.seq_length, device=x.device
        )

        return mock_output, mock_trend, mock_season
    
    def decomposed_sea_tre(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """model_step 中调用了这个方法，需要模拟它，这是用于对电力负荷进行季节和趋势分解"""
        # x.shape = B out_dim N predict_length = (B 1 N 12)
        batch_size = x.shape[0]
        num_nodes = x.shape[2]
        out_dim = x.shape[1]
        seq_len = x.shape[3]

        mock_trend = torch.randn(batch_size, out_dim, num_nodes, seq_len)
        mock_season = torch.randn(batch_size, out_dim, num_nodes, seq_len)
        return mock_trend, mock_season

class MockScaler:
    """
    模拟从datamodule里获得的scaler StandardScaler。
    """
    def __init__(self, num_nodes=321, seq_len=12):
        # 模拟真实产生后的 (N * T,) 维度数组
        size = num_nodes * seq_len
        self.mean_ = np.random.rand(size).astype(np.float32)
        self.scale_ = (np.random.rand(size) + 1.0).astype(np.float32) # 另一个随机的浮点数


    def inverse_transform(self, data):
        # 因为图数据维度特殊性，无法直接使用这个方法进行反归一化，但保留它以保持接口完整性
        print("MockScaler: inverse_transform called (but should be unused in model_step).")
        # 这是一个简化的、不完全正确的 numpy 实现
        return data * self.scale_ + self.mean_
    
class MockElectricityDataModule(LightningDataModule):
    """一个模拟的数据模块，提供假的 DataLoader 和 scaler"""
    def __init__(
            self, 
            batch_size=64, 
            history_length=12, 
            predict_length=12, 
            num_nodes=321,
            in_dim=2,
            out_dim=1
            ):
        super().__init__()
        self.batch_size = batch_size
        self.history_length = history_length
        self.predict_length = predict_length
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaler = MockScaler()

    def setup(self, stage: str = None):
        # 创建一些随机数据
        num_samples = 128
        self.X = torch.randn(num_samples, self.in_dim, self.num_nodes, self.history_length) 
        self.Y = torch.randn(num_samples, self.in_dim, self.num_nodes, self.predict_length) 
        self.dataset = TensorDataset(self.X, self.Y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


# --- 步骤 3: 主测试逻辑 ---

if __name__ == '__main__':
    # 定义超参数
    BATCH_SIZE = 8
    HISTORY_LENGTH = 12
    PREDICT_LENGTH = 12
    NUM_NODES = 321
    IN_DIM = 2
    OUT_DIM = 1

    print("--- Initializing Mock Components ---")
    # 1. 初始化模拟数据模块
    mock_dm = MockElectricityDataModule(
        batch_size=BATCH_SIZE,
        history_length=HISTORY_LENGTH,
        predict_length=PREDICT_LENGTH,
        num_nodes=NUM_NODES,
        in_dim=IN_DIM,
        out_dim=OUT_DIM
    )

    # 2. 初始化模拟网络
    mock_net = MockNet(
        input_channels=IN_DIM,
        output_channels=OUT_DIM,
        num_nodes=NUM_NODES,
        seq_length=HISTORY_LENGTH
    )

    # 3. 初始化 Lightning lit Module
    lit_module = ELECTRICITYLitModule(
        net=mock_net,
        optimizer=torch.optim.AdamW,
        scheduler=None, 
        compile=False,
    )

    print('\n--- Initializing PyTorch Lightning Trainer ---')
    # 4. 初始化 Trainer 并使用 fast_dev_run
    # fast_dev_run=True 会运行一个训练批次和一个验证批次，并禁用日志记录、回调函数和检查点；这里仅测试能否运行
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='cpu',
        logger=True,
        enable_checkpointing=True
    )

    print("\n--- Starting Trainer.fit() Test ---")    
    # 5. 运行 fit 流程
    # 这将依次调用 setup() training_step validation_step
    trainer.fit(model=lit_module, datamodule=mock_dm)
    print("---Trainer.fit() Test Completed Successfully! ---")

    print("\n--- Starting Trainer.test() Test ---")
    # 6. 运行 test 流程
    # 这将调用 setup(), test_step(), on_test_epoch_end()
    trainer.test(model=lit_module, datamodule=mock_dm)
    print("--- Trainer.test() Test Completed Seccessfully! ---")



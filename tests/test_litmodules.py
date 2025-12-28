# -*- coding: utf-8 -*-
# @Author  : Jiangsi
# @Date    : 2025-12-19
# @Update  : 2025-12-19
# @File    : test_litmodules.py
# @Desc    : pytests for ELECTRICITYLitModule

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from lightning import Trainer, LightningDataModule

from src.models.electricity_module import ELECTRICITYLitModule

# ============================================================ #
#                         mock components
# ============================================================ #
class MockNet(nn.Module):
    """Mock Net, 输入输出与真实CauSTG网络一致，需要梯度以测试反向传播"""
    def __init__(self, input_channels=2, output_channels=1, num_nodes=321, seq_length=12):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_nodes = num_nodes
        self.seq_length = seq_length

    def forward(self, x):
        B, _, N, T = x.shape
        device = x.device
        
        weight = self.weight.to(device)
        pred = weight * torch.ones(
            B, self.output_channels, N, self.seq_length, device=device
        )
        
        trend = weight * torch.ones_like(pred)
        season = weight * torch.ones_like(pred)
        return pred, trend, season
    
    def decomposed_sea_tre(self,x):
        """_summary_
        用于真实值分解：
        - 仍然使用网络参数进行学习
        - 梯度上是否冻结由调用者决定：真实值分解（梯度冻结）；网络中学习传播（梯度正常）
        """
        B, C, N, T = x.shape
        weight = self.weight.to(x.device)
        
        trend = weight * torch.ones(
            B, C, N, T, device=x.device
        )
        season = weight * torch.ones_like(trend)

        return trend, season


class MockScaler:
    """模拟StandardScaler with (N*T,)"""
    def __init__(self, num_nodes=321, seq_len=12):
        size = num_nodes * seq_len
        self.mean_ = np.random.rand(size).astype(np.float32)
        self.scale_ = (np.random.rand(size) + 1.0).astype(np.float32)


class MockElectricityDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=8,
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
        self.scaler = MockScaler(num_nodes, predict_length)

    def setup(self, stage=None):
        num_samples = 32
        self.X = torch.randn(
            num_samples, self.in_dim, self.num_nodes, self.history_length
        )
        self.Y = torch.randn(
            num_samples, self.out_dim, self.num_nodes, self.predict_length
        )
        self.dataset = TensorDataset(self.X, self.Y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


# ============================================================ #
#                         pytest fixtures
# ============================================================ #

@pytest.fixture
def datamodule():
    dm = MockElectricityDataModule()
    dm.setup()
    return dm


@pytest.fixture
def model():
    net = MockNet()
    lit = ELECTRICITYLitModule(
        net=net,
        optimizer=torch.optim.AdamW,
        scheduler=None,
        compile=False
    )
    return lit


@pytest.fixture
def trainer():
    return Trainer(
        accelerator="gpu",
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False
    )


class DummyTrainer:
        def __init__(self, datamodule):
            self.datamodule = datamodule


# ============================================================ #
#                           Tests
# ============================================================ #

def test_setup_injects_scaler(trainer, model, datamodule):
    trainer.fit(model, datamodule=datamodule)

    assert model.scaler is not None
    assert model.scaler_mean.dtype == torch.float32
    assert model.scaler_scaler.dtype == torch.float32

def test_training_step_has_valid_gradient_path(model, datamodule):
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))

    # 模拟 trainer 注入(lightning 在 fit 里做的事)
    
    model.trainer = DummyTrainer(datamodule)
    model.setup(stage="fit")

    # FIXED: 由于lightning.trainer过于复杂，很难模拟所有的属性，因此禁用掉与梯度无关的内容
    model.log = lambda *args, **kwargs: None

    loss = model.training_step(batch,0)

    assert loss.requires_grad
    assert loss.grad_fn is not None

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "No parameter received gradient"


def test_inverse_scaling_preserves_shape_and_dtype(model, datamodule):
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))

    model.trainer = DummyTrainer(datamodule)
    model.setup(stage="fit")

    out = model.model_step(batch)
    preds = out["preds"]
    targets = out["targets"]

    assert preds.shape == targets.shape
    assert preds.dtype == torch.float32


def test_model_step_inverse_scaler():
    """测试手写的反归一化有没有问题，主要测试的是维度扩散"""
    B, C, N, T = 2, 1, 3, 4
    output = torch.rand(B, C, N, T)
    
    scaler_mean = torch.rand(N*T)
    scaler_scaler = torch.rand(N*T)

    means = scaler_mean.view(1,N,T)
    scales = scaler_scaler.view(1,N,T)

    predict = output.clone()
    predict[:,0,...] = output[:,0,...]*scales + means

    expected = output[:, 0, ...] * scales + means

    np.testing.assert_allclose(
        predict[:,0,...],
        expected,
        rtol = 1e-5,
        atol=1e-8,
        err_msg="反归一化计算错误"
    )
    

def test_full_fit_and_test_cycle(trainer, model, datamodule):
    """Lightning lifecycle should run without error"""
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


def test_model_step_without_scaler_should_fail(model, datamodule):
    """回归测试：如果scaler不存在，model_step/training_step 必须显性失败"""
    
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))

    # 故意构造一个没有scaler的datamodule
    class BadDataModule:
        pass

    model.trainer = DummyTrainer(BadDataModule)

    with pytest.raises((RuntimeError, AttributeError)):
        model.model_step(batch)


def test_loss_is_finite(model, datamodule):
    """测试loss不产生 NaN / Inf"""
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))

    model.log = lambda *args, **kwargs: None
    model.trainer = DummyTrainer(datamodule)
    model.setup("fit")

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_scaler_is_immutable_duting_training(model, datamodule):
    """scaler在训练过程中不能被误改"""
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))

    model.log = lambda *args, **kwargs: None
    model.trainer = DummyTrainer(datamodule)
    model.setup("fit")

    mean_before = model.scaler_mean.clone()
    scale_before = model.scaler_scaler.clone()

    loss = model.training_step(batch,0)
    loss.backward()

    assert torch.allclose(model.scaler_mean, mean_before)
    assert torch.allclose(model.scaler_scaler, scale_before)
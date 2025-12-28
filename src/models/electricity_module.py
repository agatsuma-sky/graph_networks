# -*- coding: utf-8 -*-
# @Author  : Jiangsi
# @Date    : 2025-12-17
# @Update  : 2025-12-17
# @File    : electricity_module.py
# @Desc    : 负责电力负荷数据集的 LightningModule、完整封装原始代码中的训练、验证和测试逻辑
from typing import Any, Dict, Tuple, List
import math
import torch
import torch.nn.functional as F
from torch import nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class ELECTRICITYLitModule(LightningModule):
    """负责电力负荷数据集的 LightningModule.
       这个模块完整地封装了原始代码中的训练、验证和测试逻辑
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `ELECTRICITYLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.scaler = None  # 将在 setup 钩子中被赋值

        # 损失函数
        self.loss_fn = masked_mae

        # 初始化用于在整个 epoch 中累计和平均指标的 Metric 对象
        # 训练指标
        self.train_loss = MeanMetric()
        self.train_mape = MeanMetric()
        self.train_rmse = MeanMetric()

        # 验证指标
        self.val_loss = MeanMetric()
        self.val_mape = MeanMetric()
        self.val_rmse = MeanMetric()

        # 测试指标
        self.test_loss = MeanMetric()
        self.test_mape = MeanMetric()
        self.test_rmse = MeanMetric()

        # 记录test输出
        self.test_step_outputs = []

    def diff(self, x: torch.Tensor) -> torch.Tensor:
        """计算张量在最后一个维度上的差分"""
        return x[..., 1:] - x[..., :-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """模型的前向传播
        """
        return self.net(x)
    
    def setup(self, stage: str) -> None:
        """在 fit、validate、test、predict 开始时被调用.
        我们在这里获取 ElectricityDataModule 里的scaler属性，给训练负荷数据还原真实大小
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        if stage == "fit":
            self._ensure_scaler_ready()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """展示一批数据的一次模型step

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # FIXED：确保不管是否单独调用model_step，都能获取到scaler
        self._ensure_scaler_ready()

        # NOTE: 不需要self.model.train() 进入训练模式，许多杂活都让lightning做了
        train_x, train_y = batch  # B C N T 
        B,C,N,T = train_x.shape

        # 准备真实值，dim=1的0 是电力负荷值（无归一化）
        real_phys = train_y[:,0:1,...]

        # 模型前向传播 三个网络输出都是 B C N T （输入输出全是归一化的数据）
        output, pred_trend, pred_season = self.net(train_x)
        
        # FIXED：修复model_step的量纲错误，对pred_season, pred_trend, output 都需要反归一化
        # 使用 scaler 对模型输出进行逆变换，得到真实尺度的预测值 scaler和mean的shape都是: 标量
        means = self.scaler_mean.to(output.device)
        scales = self.scaler_scaler.to(output.device)
        real_norm = (real_phys - means) / scales  # 临时归一化，因为余弦相似度在标准差状态下更准

        # 计算真实负荷的趋势分解和季节分解
        with torch.no_grad():
            real_trend_norm, real_season_norm = self.net.decomposed_sea_tre(real_norm.to(output.device))
        
        # 计算 Loss (在归一化空间下)
        loss_sea = masked_mape(pred_season, real_season_norm, 0.0)
        loss_tre = torch.mean(F.cosine_similarity(pred_trend,real_trend_norm)) + torch.mean(
            F.cosine_similarity(self.diff(pred_trend),self.diff(real_trend_norm))
        )
        loss_reg = torch.mean(F.cosine_similarity(pred_trend,pred_season))

        # 执行反归一化，只有C=0需要，因为只有电力负荷特征做了标准化
        predict_phys = output[:, 0:1, ...] * scales + means

        # 计算基础损失和总损失, base_loss无梯度 TODO: 24日下午，继续测试梯度异常的问题
        # print(f"[DEBUG] predict_phys：{predict_phys}, real_phys: {real_phys}")
        base_loss = self.loss_fn(predict_phys, real_phys, 0.0)
        
        # TODO: 按理这里的系数也应该是模型的超参数 
        lambda0, lambda1, lambda2 = 1e-5,1e-5,1e-5
        # FIXED: 向ddp工程保证，有些参数是在某些阶段没有梯度或者几乎为0的。
        total_loss = base_loss + lambda0*loss_sea + lambda1*loss_tre + lambda2*loss_reg + 0.0 * pred_trend.mean() + 0.0 * pred_season.mean()

        return {
            "loss": total_loss, 
            "preds": predict_phys, 
            "targets": real_phys}
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """展示一批训练数据的单个训练步."""

        step_output = self.model_step(batch)
        loss, preds, targets = step_output["loss"], step_output["preds"], step_output["targets"]

        print(f"Total Loss: {loss.item()}")
        print(f"Requires Grad: {loss.requires_grad}")
        print(f"Grad Fn: {loss.grad_fn}")
        
        # DEBUG：测试一下有哪些参数可能没有梯度
        loss.backward(retain_graph=True)

        for name, p in self.net.named_parameters():
            if p.requires_grad and p.grad is None:
                print("NO GRAD:", name)

        # 更新并记录指标
        self.train_loss(loss)
        self.train_mape(masked_mape(preds, targets, 0.0))
        self.train_rmse(masked_rmse(preds, targets, 0.0))

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mape", self.train_mape, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """展示一批验证数据的单个验证步"""

        step_output = self.model_step(batch)
        loss, preds, targets = step_output["loss"], step_output["preds"], step_output["targets"]

        # 更新并记录指标
        self.val_loss(loss)
        self.val_mape(masked_mape(preds, targets, 0.0))
        self.val_rmse(masked_rmse(preds, targets, 0.0))
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mape", self.val_mape, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """展示一批测试数据的单个测试步"""

        step_output = self.model_step(batch)
        output_dict = {"preds": step_output["preds"], "targets": step_output["targets"]}
        self.test_step_outputs.append(output_dict)

        return output_dict

    def on_test_epoch_end(self) -> None:
        """在测试 epoch 结束时被调用，用于计算和打印最终的12个时间步的指标"""
        # 将所有 batch 的预测和真实值拼接起来
        y_hat = torch.cat([o["preds"] for o in self.test_step_outputs], dim=0)
        realy = torch.cat([o["targets"] for o in self.test_step_outputs], dim=0)

        # 计算一些指标
        mae = masked_mae(y_hat, realy, 0.0)
        mape = masked_mape(y_hat, realy, 0.0)
        rmse = masked_rmse(y_hat, realy, 0.0)
        
        print(f"Test MAE: {mae.item():.4f}")
        print(f"Test MAPE: {mape.item():.4f}")
        print(f"Test RMSE: {rmse.item():.4f}")
        print("on_test_epoch_end finished successfully.")
        
        # 在真实测试中，你可能会把这些指标记录下来
        self.log_dict({"test/final_mae": mae, "test/final_mape": mape, "test/final_rmse": rmse})

        # 清理内存
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> Dict[str, Any]:
        """设置优化器、调度器.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def _ensure_scaler_ready(self):
        """pytest测试中，有单独调用model_step等的情况，而没有调用setup。
           我们在所有的场景下都要能正确获取scaler"""
        if not hasattr(self, "scaler_mean"):
            if self.trainer is None or self.trainer.datamodule is None:
                raise RuntimeError("Scaler 尚未注入, 且 train/datamodule 不可用")
            
            dm = self.trainer.datamodule
            if not hasattr(dm, "scaler"):
                raise AttributeError("DataModule 中不存在 scaler")
        
            self.scaler = dm.scaler
            self.scaler_mean = torch.tensor(
                self.scaler.mean_, dtype=torch.float32, device=self.device
            )
            self.scaler_scaler = torch.tensor(
                self.scaler.scale_, dtype=torch.float32, device=self.device
            )


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val) -> torch.Tensor:
    """计算 MAE, 考虑标签值中可能出现无效值或缺失值，通过创建掩码，忽略无效标签
    加权有效标签值，最后计算平均损失"""
    # print("--- EXECUTING THE NEW, CORRECT masked_mae (Version 2.0) ---")
    if isinstance(null_val, float) and math.isnan(null_val):
        # Nan无效值掩码为0
        mask = ~torch.isnan(labels) 
    else:
        # 非无效值，掩码为1
        mask = (labels != null_val)
    
    # 直接用掩码索引，获取有效的pred
    valid_preds = preds[mask]
    valid_labels = labels[mask]
    # 避免极端情况，labels全为nan，则mask为0.0，导致出现除0得到Nan。
    if valid_preds.numel() == 0:
        return torch.tensor(0.0, device=preds.device)

    # 与上面面类似，这里是由于labels、preds都有可能会出现Nan的情况，导致loss会传染保持有nan
    # BUG： loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)这里有一个巨大bug，torch.zeros_like(loss)是没有梯度的，会破坏计算图

    return torch.mean(torch.abs(valid_preds - valid_labels))

def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val) -> torch.Tensor:
    """计算 MAPE, 掩码逻辑同上"""
    if isinstance(null_val, float) and math.isnan(null_val):
        # Nan无效值掩码为0
        mask = ~torch.isnan(labels) 
    else:
        # 非无效值，掩码为1
        mask = (labels != null_val)
    mask = mask.float()
    # mask /= torch.mean(mask)
    # # 避免极端情况，labels全为nan，则mask为0.0，导致出现除0得到Nan。
    # mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # loss = torch.abs(preds - labels)/labels  # 指标计算
    # loss = loss * mask
    # # 与上面面类似，这里是由于labels、preds都有可能会出现Nan的情况，导致loss会传染保持有nan
    # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # print(f"DEBUG MAPE RAW: {torch.mean(loss).item()}")
    # return torch.mean(loss)
    # FIXED: 改用更通用的做法
    loss = torch.abs(preds - labels) / (torch.abs(labels) + 1e-5)
    loss = loss * mask

    valid_mape = torch.sum(loss) / (torch.sum(mask) + 1e-5)
    return valid_mape

def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val) -> torch.Tensor:
    """计算 RMSE"""
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val) -> torch.Tensor:
    """计算 MSE"""
    if isinstance(null_val, float) and math.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

if __name__ == "__main__":
    _ = ELECTRICITYLitModule(None, None, None, None)

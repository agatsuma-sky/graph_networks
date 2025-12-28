# -*- coding: utf-8 -*-
# @Author  : Jiangsi
# @Date    : 2025-09-03
# @Update  : 2025-09-05
# @File    : electricity_datamodule.py
# @Desc    : 构建电力负荷数据的数据模块
import os
import argparse
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, TensorDataset
from sklearn.preprocessing import StandardScaler

class ElectricityDataModule(LightningDataModule):
    """`LightningDataModule` for the electricity dataset.
    DDP是分布式数据并行技术，如果有多块GPU，会在CPU0上加载模型，下载数据等准备操作，
    随后在其他GPU上复制模型，均分数据，并行训练，每次迭代后分享模型参数，平均后再
    复制给所有GPU，重复迭代...
    """

    def __init__(
        self,
        dataset_dir: str = "data/electricity",
        batch_size: int = 64,
        # valid_batch_size: Optional[int] = None,
        # test_batch_size: Optional[int] = None,
        # train_val_test_split: Optional[Tuple[float, float, float]] = (0.7, 0.2, 0.1),
        env: int = 1,
        use_single_env: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """初始化Electricity数据集

        Args:
            x_samples (List): 历史数据样本，(num_samples, input_dim, num_nodes, input_length)
            y_samples (List): 预测数据样本，(num_samples, output_dim, num_nodes, output_length)
            x_len (int, optional): 用于预测的过去时间步. Defaults to 12.
            y_len (int, optional): 预测未来时间步. Defaults to 12.
            data_dir (str, optional): 数据路径. Defaults to "data/".
            train_val_test_split (Tuple[float, float, float], optional): 数据集划分比例. Defaults to (0.7, 0.2, 0.1).
            batch_size (int, optional): 批次大小. Defaults to 64.
            num_workers (int, optional): GPU工作数. Defaults to 0.
            pin_memory (bool, optional): 固定CPU内存，加速训练数据加载. Defaults to False.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self.valid_batch_size = valid_batch_size or batch_size
        # self.test_batch_size = test_batch_size or batch_size
    
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.scaler: Optional[StandardScaler] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """加载数据，赋予属性，stage代表生命周期，这里是为兼容写的，没有实际作用

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # 将原本的批次均分给指定的多个GPU，如果均分失败会报错
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # 加载、划分数据集
        if not self.data_train and not self.data_val and not self.data_test:
            
            # 1. Load raw data 
            raw_data = {}
            for cat in ["train", "val", "test"]:
                with np.load(os.path.join(self.hparams.dataset_dir, f"{cat}.npz")) as loader:
                    # 数组名称: 'x_train', 维度形状: (73636, 12, 321, 2), y作为标签，y_train 也一样的维度形状
                    raw_data[f"x_{cat}"] = loader['x'].astype(np.float32)
                    raw_data[f"y_{cat}"] = loader['y'].astype(np.float32)
            
            # 2. 只标准化第一个特征(负荷)
            # 修复：Standardscaler只接受1D或2D的数据
            if self.scaler is None:
                self.scaler = StandardScaler()

                # 提取用于标准化的 3D 数据，需要多一步展平
                train_load = raw_data["x_train"][:, 0, ...]
                self.scaler.fit(train_load.reshape(-1, 1))

            # 3.环境选择逻辑
            train_len = raw_data["x_train"].shape[0]
            env_piece = [0, int(train_len//4), int(train_len//4)*2, int(train_len//4)*3, train_len]

            if self.hparams.use_single_env:
                l, r = env_piece[self.hparams.env - 1], env_piece[self.hparams.env]
            else:
                l, r = env_piece[0], env_piece[-1]
            
            # 4. 对第一个特征实施标准化
            # 修复：对所有数据集用train的尺度进行标准化
            for cat in ['train', 'val', 'test']:
                key_x = f"x_{cat}"

                feat = raw_data[key_x][:,0,...]  # 形状: B,N,T
                orig_shape = feat.shape  # 记录原始形状 (B, N, T)

                # 标准化并还原形状
                feat_norm = self.scaler.transform(feat.reshape(-1, 1)).reshape(orig_shape)
                raw_data[key_x][:, 0, ...] = feat_norm
            
            # 5. 转为dataset格式 样本数量不一定，test、train、val的数据形状都为 
            # x: (num_samples, input_dim, num_nodes, history_length)     input_dim的第0特征 为电力负荷，其他特征可能有time of day例如12点 => 0.5; day of week 例如周一 => 1
            # y: (num_samples, output_dim, num_nodes, predict_length)    output_dim 与input_dim 特征一致 

            self.data_train = TensorDataset(
                torch.from_numpy(raw_data["x_train"][l:r]).float(),
                torch.from_numpy(raw_data["y_train"][l:r]).float()
            )
            self.data_val = TensorDataset(
                torch.from_numpy(raw_data["x_val"]).float(),
                torch.from_numpy(raw_data["y_val"]).float()
            )
            self.data_test = TensorDataset(
                torch.from_numpy(raw_data["x_test"]).float(),
                torch.from_numpy(raw_data["y_test"]).float()
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """返回训练数据的Dataloader

        :return: The train dataloader.
        """
        # DEBUG: 测试dataset的shape
        if self.data_train:
            print(f"[DEBUG] Total samples in data_train: {len(self.data_train)}")
            sample_x, sample_y = self.data_train[0]
            print(f"[DEBUG] Single sample X shape: {sample_x.shape}")

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """返回验证数据的Dataloader

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """返回测试数据的Dataloader

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """当保存检查点的时候调用，保存这个类里有必要的变量

        :return: A dictionary containing the datamodule state that you want to save.
        """
        if self.scaler is None:
            raise RuntimeError("scaler not initialized. Call setup() first")
        # 将 numpy 数组转为 Tensor。规避 Numpy 在恢复检查点时的被 PyTorch 拦截的风险

        return {"scaler_mean": torch.from_numpy(self.scaler.mean_), 
                "scaler_scale": torch.from_numpy(self.scaler.scale_)}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """当加载检查点时调用. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        self.scaler = StandardScaler()
        if "scaler_mean" in state_dict:
            self.scaler.mean_ = state_dict["scaler_mean"].cpu().numpy()
            self.scaler.scale_ = state_dict["scaler_scale"].cpu().numpy()

    def prepare_data(self) -> None:
        """由于官方指定不能用于保存状态例如self.data = x
           因此这里我们暂时不用这里。
           原意是用来下载数据集的，只开启一个CPU进程
        """
        pass






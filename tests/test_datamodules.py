import os
import pytest
import torch
import numpy as np
import numpy.testing as npt
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler

# from src.data.mnist_datamodule import MNISTDataModule
from src.data.electricity_datamodule import ElectricityDataModule

MOCK_SHAPE = (1000, 2, 321, 12)  # B C N T
MOCK_BATCH_SIZE = 64
# TODO: 处理pytest的这几个错误。 2025.12.16
# -------------------------- 辅助函数：创建模拟 npz 数据 ----------------------
def create_mock_npz_data(shape, mean=500.0, std=50.0):
    """创建模拟的 npz 数据字典，只对第一个特征进行高斯分布模拟"""
    num_samples, input_dim, num_nodes, input_length = shape
    data_feature = np.zeros(shape, dtype=np.float32)

    # 特征 C=0 (电力负荷): 用于标准化的主要特征, 阶梯构造来模拟季节性
    chunk_size = int(num_samples // 4)
    data_feature[0:chunk_size, 0, ...] = np.random.normal(loc=mean+10, scale=std, size=(num_samples//4, num_nodes, input_length))
    data_feature[chunk_size:2*chunk_size, 0, ...] = np.random.normal(loc=mean+20, scale=std, size=(num_samples//4, num_nodes, input_length))
    data_feature[2*chunk_size:3*chunk_size, 0, ...] = np.random.normal(loc=mean-10, scale=std, size=(num_samples//4, num_nodes, input_length))
    data_feature[3*chunk_size:, 0, ...] = np.random.normal(loc=mean-20, scale=std, size=(num_samples//4, num_nodes, input_length))

    # 特征 C=1 (例如 Time-of-day): 不应该标准化的辅助特征
    data_feature[:, 1, ...] = np.random.uniform(0, 1, size=(num_samples, num_nodes, input_length))

    # 拼接数据 B C N T
    data = data_feature

    return {
        'x': data,
        'y': data.copy()
    }

# -------------------------- 辅助函数：模拟 np.load 行为 ----------------------
def mock_np_load_side_effect(file_path, *args, **kwargs):
    """模拟 np.load 对 'train', 'val', 'test' 文件的加载行为"""
    if 'train.npz' in file_path:
        data = create_mock_npz_data((800, 2, 321, 12), mean=100, std=10)
    elif 'val.npz' in file_path:
        data = create_mock_npz_data((100, 2, 321, 12), mean=110, std=12)
    elif 'test.npz' in file_path:
        data = create_mock_npz_data((100, 2, 321, 12), mean=90, std=8)
    else:
        raise FileNotFoundError(f"Mocking failed for path: {file_path}")
    
    # MagicMock 用于模拟 npz 对象的行为
    mock_file = MagicMock()
    mock_file.files = ['x', 'y']
    mock_file.__getitem__.side_effect = lambda key: data[key]
    mock_file.get.side_effect = lambda key: data[key]
    return mock_file

@pytest.fixture
def ElectricityDataModule():
    from src.data.electricity_datamodule import ElectricityDataModule
    return ElectricityDataModule

# ====================================================================
# 1. 功能和兼容性测试
# ====================================================================

# 使用 @patch 模拟 np.load, 原数据模块的setup方法里有np.load函数读取数据，
# 我们用mock_np_load_side_effect 模拟提供测试数据。
@patch('numpy.load', side_effect=mock_np_load_side_effect)
@pytest.mark.parametrize("world_size", [1,2])  # 测试DDP兼容性
def test_datamodule_basic_functionality_and_ddp(mock_np_load, world_size: int, ElectricityDataModule):

    # --- 实例化 DataModule ---
    dm = ElectricityDataModule(
        dataset_dir='mock_data/',
        batch_size=MOCK_BATCH_SIZE,
        num_workers=0,
    )

    # 模拟 Trainer 环境，检查 DDP 批次划分逻辑是否正确
    dm.trainer = MagicMock(world_size=world_size)

    # --- 调用 setup 钩子
    dm.setup()
    
    # patch后，np.load = mock_np_load, mock_np_load 会return mock_np_load_side_effect 
    assert mock_np_load.call_count == 3  # 可以查看调用次数、调用参数或者调用顺序等

    # 1. 检查 DDP 划分批次 (防止 DDP 运行时崩溃)
    expected_batch_size = MOCK_BATCH_SIZE // world_size
    assert dm.batch_size_per_device == expected_batch_size, \
        f'[DDP Bug] 批次大小划分错误：预期 {expected_batch_size}, 实际 {dm.batch_size_per_device}'

    # 2. 检查 DataLoader 返回类型
    train_dl = dm.train_dataloader()
    assert isinstance(train_dl, DataLoader), "[类型错误] train_dataloader 应该返回 DataLoader"

    # 3. 检查 DataLoader 是否为空
    assert len(train_dl.dataset) > 0, "[数据为空 Bug] 训练集不应为空"


# ====================================================================
# 2. 数据泄漏和归一化测试
# ====================================================================

@patch('numpy.load', side_effect=mock_np_load_side_effect)
def test_datamodule_scaler_and_normalization(mock_np_load, ElectricityDataModule):

    # --- 1. 实例化 DataModule ---
    dm = ElectricityDataModule(dataset_dir="mock_data/", batch_size=MOCK_BATCH_SIZE, use_single_env=False)
    dm.trainer = MagicMock(world_size=1)
    dm.setup()
    assert mock_np_load.call_count == 3  # 可以查看调用次数、调用参数或者调用顺序等

    # --- 2. Scaler 存在性和正确性检查 (防止训练/测试数据泄露)
    assert dm.scaler is not None, "[Scaler Bug] Scaler 应该被初始化"
    # 由于 train/val/test 的 mock 均值不同, 确保 scaler 均值不在 val/test 均值附近
    # FIXED：numpy数组不能与数字比较
    # assert 95 < dm.scaler.mean_ < 105, "[数据泄露Bug] Scaler均值应基于训练集(约100)计算"
    is_mean_in_range = np.all((dm.scaler.mean_ > 98) & (dm.scaler.mean_ < 102))
    assert is_mean_in_range, "[数据泄露Bug] Scaler均值应基于训练集(约100)计算"

    # --- 3. Batch 输出形状、类型和归一化检查 ---
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    # print(x.shape, y.shape)
    
    # A. 维度检查 (防止复杂的 Numpy 转置错误)
    C, N, T = (2, 321, 12)
    assert x.shape[1:] == (C,N,T), f"[维度 Bug] X 批次形状错误: {x.shape} 后三个维度的形状有误"
    assert x.shape[0] <= MOCK_BATCH_SIZE, f"[维度 Bug] X 批次形状错误: {x.shape} ,批次维度大小错误"
    assert y.shape[1:] == (C,N,T), f"[维度 Bug] X 批次形状错误: {y.shape} 后三个维度的形状有误"
    assert y.shape[0] <= MOCK_BATCH_SIZE, f"[维度 Bug] X 批次形状错误: {y.shape} 批次维度大小错误"
    assert x.dtype == torch.float32 and y.dtype == torch.float32, "[类型 Bug] dtype 应该为torch.float32"

    # B. 归一化检查 (防止归一化错误，这是模型不收敛的常见原因)
    # 特征 C=0 (电力负荷): 应该被归一化到 N(0, 1)
    load_feature = x[:, 0, ...].cpu().numpy().flatten()
    assert np.isclose(np.mean(load_feature), 0.0, atol=0.1), "[归一化Bug] 负荷特征均值应该接近 0"
    assert np.isclose(np.std(load_feature), 1.0, atol=0.1), "[归一化Bug] 负荷特征均值应该接近 1"

    # C. 非负荷特征检查 (C=1): 应该保持原始范围 (例如 [0, 1])
    non_load_feature = x[:, 1, ...].cpu().numpy().flatten()
    assert np.max(non_load_feature) <= 1.05 and np.min(non_load_feature)>= -0.05, \
        "[特征污染 Bug] 非负荷特征的值域被意外修改 (应在 [0, 1] 附近)"


# ====================================================================
# 3. 检查点保存与加载
# ====================================================================

@patch('numpy.load', side_effect=mock_np_load_side_effect)
def test_datamodule_checkpoint(mock_np_load, ElectricityDataModule):

    dm_orig = ElectricityDataModule(dataset_dir="mock_data/", batch_size=MOCK_BATCH_SIZE)
    dm_orig.trainer = MagicMock(world_size=1)

    # 必须先调用 setup() 来初始化 scaler
    dm_orig.setup()
    assert mock_np_load.call_count == 3  # 可以查看调用次数、调用参数或者调用顺序等

    state = dm_orig.state_dict()

    # 检查 state_dict 中是否包含关键状态 (防止模型加载时无法反归一化)
    assert "scaler_mean" in state and "scaler_scale" in state, "[Checkpoint Bug] state_dict 缺少 scaler 关键键"

    # 加载状态检查
    dm_new = ElectricityDataModule(dataset_dir="mock_data/", batch_size=MOCK_BATCH_SIZE)
    dm_new.load_state_dict(state)

    # 检查加载后的值是否精确匹配
    # FIXED：scaler.mean_ 都是（B, N * T）的2D数组，数组间的比较不能用 "=="
    # assert dm_new.scaler.mean_ == state["scaler_mean"], "[Checkpoint Bug] load_state_dict 失败 (mean)"
    # assert dm_new.scaler.scale_ == state["scaler_scale"], "[Checkpoint Bug] load_state_dict 失败 (scale)"
    mean_orig = dm_orig.scaler.mean_
    mean_new = dm_new.scaler.mean_

    npt.assert_allclose(
        mean_new,
        state["scaler_mean"],
        rtol = 1e-5,
        err_msg="[Checkpoint Bug] load_state_dict 失败 (mean)"
    )

    npt.assert_allclose(
        dm_new.scaler.scale_,
        state["scaler_scale"],
        rtol = 1e-5,
        err_msg="[Checkpoint Bug] load_state_dict 失败 (scale)"
    )

# ====================================================================
# 4. 测试 数据加载器是否按照要求打乱或者不打乱
# ====================================================================
    
@patch('numpy.load', side_effect=mock_np_load_side_effect)
def test_datamodule_shuffle_behavior(mock_np_load, ElectricityDataModule):

    dm = ElectricityDataModule(
        dataset_dir='mock_data/',
        batch_size=MOCK_BATCH_SIZE
    )
    dm.trainer = MagicMock(world_size=1)
    dm.setup()

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    from torch.utils.data import SequentialSampler, RandomSampler

    assert isinstance(train_dl.sampler, RandomSampler), \
        "[Train Loader Bug] train_dataloader 应该使用shuffle"
    
    assert isinstance(val_dl.sampler, SequentialSampler), \
        "[Val Loader Bug] val_dataloader 不应该使用 shuffle"
    
    assert isinstance(test_dl.sampler, SequentialSampler), \
        "[Test Loader Bug] test_dataloader 不应该使用 shuffle"

# ====================================================================
# 5. 测试 环境划分是否正确： 不同环境的数据来自不同时间段
# ====================================================================
# 外部存储：用于收集所有 env 的归一化均值（作为指纹）
COLLECTED_NORM_MEANS = {}
# 基于 Mock 数据 (原始均值 110, 120, 90, 80; 整体均值 M_all=100; 整体标准差 S_all≈18.7)
# 假设我们计算出准确的 M_all ≈ 100.0，S_all ≈ 18.708 (根据您之前的数据计算)
S_ALL_MOCK = 18.708 # 使用更精确的预计算值
M_ALL_MOCK = 100.0
EXPECTED_NORM_MEANS_MAP = {
    1: (110.0 - M_ALL_MOCK) / S_ALL_MOCK, # 10 / 18.708 ≈ 0.534
    2: (120.0 - M_ALL_MOCK) / S_ALL_MOCK, # 20 / 18.708 ≈ 1.069
    3: (90.0 - M_ALL_MOCK) / S_ALL_MOCK,  # -10 / 18.708 ≈ -0.534
    4: (80.0 - M_ALL_MOCK) / S_ALL_MOCK  # -20 / 18.708 ≈ -1.069
}
@patch('numpy.load', side_effect=mock_np_load_side_effect)
@pytest.mark.parametrize("env, expected_mean", [
    (1, 110.0),  # Env 0 预期均值 100.0
    (2, 120.0),
    (3, 90.0),
    (4, 80.0),
])
def test_env_slicing_by_expected_mean(mock_np_load, env, expected_mean, ElectricityDataModule):

    dm = ElectricityDataModule(
        dataset_dir="mock_data/",
        batch_size=MOCK_BATCH_SIZE,
        env=env,
        use_single_env=True
    )
    dm.trainer = MagicMock(world_size=1)
    dm.setup()

    # 验证原切片逻辑
    normalized_mean = torch.mean(dm.data_train.tensors[0][:, 0, ...]).item()

    # 用归一化的均值来断言
    expected_norm_mean = EXPECTED_NORM_MEANS_MAP[env]

    assert np.isclose(normalized_mean, expected_norm_mean, atol=0.1), \
        f"Env {env} 切片错误! 归一化均值 {normalized_mean:.2f} 不接近预期 {expected_norm_mean:.2f}"

@patch('numpy.load', side_effect=mock_np_load_side_effect)
def test_multi_env_contains_all_samples(mock_np_load, ElectricityDataModule):
    dm = ElectricityDataModule(
        dataset_dir='mock_data/',
        batch_size=MOCK_BATCH_SIZE,
        use_single_env=False
    )
    dm.trainer = MagicMock(world_size=1)
    dm.setup()

    normalized_mean = torch.mean(dm.data_train.tensors[0][:, 0, ...]).item()
    expected_norm_mean = 0.0

    assert np.isclose(normalized_mean, expected_norm_mean, atol=0.1), \
        f"多环境训练集的归一化为：{normalized_mean}，错误！实际应接近为 {expected_norm_mean}"

# ====================================================================
# 6. 测试 环境泛化与消融实验（交叉验证）
# ====================================================================

@patch('numpy.load', side_effect=mock_np_load_side_effect)
@pytest.mark.parametrize("env", [1, 2, 3, 4])
def test_single_env_mean_differs_after_normalization(mock_np_load, env, ElectricityDataModule):
    """
    测试目标 A：验证 DataModule 成功切片出具有正确归一化均值（指纹）的数据子集。
    """
    dm = ElectricityDataModule(
        dataset_dir="mock_data/",
        batch_size=MOCK_BATCH_SIZE,
        env=env,
        use_single_env=True
    )
    dm.trainer = MagicMock(world_size=1)
    dm.setup()
    
    # 1. 计算归一化后的均值 (指纹)
    # 使用整个训练集的均值，因为这代表了该环境的整体分布
    # 确保 data_train.tensors 存在且非空
    assert dm.data_train is not None and len(dm.data_train.tensors) > 0, "训练数据未加载成功"
    
    # 提取第0个特征通道的均值
    normalized_mean = torch.mean(dm.data_train.tensors[0][:, 0, ...]).item()

    # 2. 健全性检查
    assert not np.isnan(normalized_mean), "归一化均值计算结果为 NaN"
    
    # 3. 验证切片准确性：断言归一化后的均值是否与预期的理论值接近
    expected_norm_mean = EXPECTED_NORM_MEANS_MAP[env]
    
    # 容忍度 atol=0.01 是合理的，考虑浮点数精度和 mock 数据的微小随机性
    assert np.isclose(normalized_mean, expected_norm_mean, atol=0.01), \
        f"[Slicing/Distribution Bug] Env {env} 归一化均值 ({normalized_mean:.3f}) 不符合预期 ({expected_norm_mean:.3f})。\n" \
        f"这表明数据切片或归一化计算存在错误。"

    # 4. 收集指纹（为后续测试做准备）
    COLLECTED_NORM_MEANS[env] = normalized_mean
    

def test_all_env_fingerprints_show_stat_difference():
    """
    测试目标 B：断言所有环境的指纹（归一化均值）两两之间存在统计学差异，以验证 DG 假设。
    """
    # 确保所有环境的指纹都已收集
    expected_envs = {1, 2, 3, 4}
    assert set(COLLECTED_NORM_MEANS.keys()) == expected_envs, \
        "指纹收集不完整。请确保 test_single_env_mean_differs_after_normalization 已运行并成功收集所有环境的指纹。"
        
    fingerprints = list(COLLECTED_NORM_MEANS.values())
    
    # 核心判断：最小差异阈值。我们预期差异至少应为 (110-100)/18.708 * 2 ≈ 1.068
    # 我们将阈值设为一个保守的值，例如 0.5。
    MIN_DIFFERENCE_THRESHOLD = 0.5 

    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            diff = abs(fingerprints[i] - fingerprints[j])
            
            # 使用 float 类型断言，确保比较的准确性
            assert diff >= MIN_DIFFERENCE_THRESHOLD, \
                f"[DG Assumption Failed] 环境指纹差异过小：Env {i+1} 和 Env {j+1} 的差异为 {diff:.3f}，小于要求的 {MIN_DIFFERENCE_THRESHOLD}。\n" \
                f"指纹值: {fingerprints[i]:.3f} vs {fingerprints[j]:.3f}。"











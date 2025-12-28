import torch
import pytest

from src.models.components.CauSTG import (
    nconv,
    gcn,
    Decomposed_temporal_evolution_extractor,
    CauSTG
)


def has_grad(module: torch.nn.Module):
    return any(
        p.grad is not None for p in module.parameters() if p.requires_grad
    )


# ============================================================ #
#                           fixture
# ============================================================ #

@ pytest.fixture
def small_CauSTG():
    return CauSTG(
        device="cpu",
        num_nodes=5,
        dropout=0.0,
        use_gcn=True,
        adaptive_adj=True,
        init_adj=None,
        in_dim=2,
        out_dim=1,
        residual_channels=4,
        dilation_channels=4,
        skip_channels=8,
        end_channels=16,
        kernel_size=3,
        blocks=1,
        layers=1
    )


# ============================================================ #
#                           forward 测试
# ============================================================ #

def test_CauSTG_forward_shapes(small_CauSTG):
    B, N, Tin, Tout = 2, 5, 12, 12
    x = torch.randn(B, 2, N, Tin)

    y, trend, season = small_CauSTG(x)

    assert y.shape == (B, 1, N, Tout)
    assert trend.shape == y.shape
    assert season.shape == y.shape


def test_CauSTG_preserves_dtype_and_device(small_CauSTG):
    x = torch.randn(2,2,5,12,dtype=torch.float32)

    y,trend,season = small_CauSTG(x)

    assert y.dtype == x.dtype
    assert trend.dtype == x.dtype
    assert season.dtype == x.dtype
    assert y.device == x.device


# ============================================================ #
#                           梯度路径测试
# ============================================================ #

def test_CauSTG_has_gradient_flow(small_CauSTG):
    x = torch.randn(2,2,5,12)

    y,trend,season = small_CauSTG(x)
    loss = y.mean() + trend.mean() + season.mean()
    loss.backward()

    grads = [p.grad for p in small_CauSTG.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "没有可学习参数收到梯度!"


def test_core_modules_has_grad(small_CauSTG):
    x = torch.randn(2,2,5,12)
    small_CauSTG.train()

    y,trend, season = small_CauSTG(x)

    # 强制把 adp 引入 loss
    adp = torch.mm(small_CauSTG.nodevec1, small_CauSTG.nodevec2)
    loss = y.mean()+ 0.01 * adp.mean()

    loss.backward()

    assert small_CauSTG.nodevec1.grad is not None
    assert small_CauSTG.nodevec2.grad is not None
    assert has_grad(small_CauSTG.start_conv)
    assert has_grad(small_CauSTG.end_conv_2)


# ============================================================ #
#                      nconv / gcn 结构测试
# ============================================================ #

def test_nconv_shape():
    B, C, N, T = 2, 4, 5, 6
    x = torch.randn(B, C, N, T)
    A = torch.randn(N, N)

    conv = nconv()
    y = conv(x, A)

    assert y.shape == (B, C, N, T)


def test_gcn_preserves_time_dimension():
    B, C, N, T = 2, 4, 5, 6
    x = torch.randn(B, C, N, T)
    support = [torch.randn(N, N)]

    g = gcn(c_in=C, c_out=8, dropout=0.0, support_len=1)
    y = g(x, support)

    assert y.shape[-1] == T


# ============================================================ #
#                      分解模块 测试
# ============================================================ #

def test_decomposition_shape_and_grad():
    B, N, T = 2, 5, 12
    x = torch.randn(B, 1, N, T, requires_grad=True)

    d = Decomposed_temporal_evolution_extractor()
    trend, season = d(x)

    assert trend.shape == x.shape
    assert season.shape == x.shape

    (trend.mean() + season.mean()).backward()
    assert x.grad is not None


# ============================================================ #
#                       数值稳定性 测试
# ============================================================ #

def test_zero_input_no_nan(small_CauSTG):
    x = torch.zeros(2, 2, 5, 12)

    y, trend, season = small_CauSTG(x)

    assert torch.isfinite(y).all()
    assert torch.isfinite(trend).all()
    assert torch.isfinite(season).all()





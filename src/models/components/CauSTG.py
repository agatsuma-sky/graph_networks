import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
from typing import Dict, List, Any, Optional, Union

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    
class Bi_directional_spatial_relation_learner(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3):
        super(Bi_directional_spatial_relation_learner,self).__init__()

        self.pos_gcn = gcn(c_in=c_in,c_out=c_out,dropout=dropout,support_len=support_len)
        self.neg_gcn = gcn(c_in=c_in,c_out=c_out,dropout=dropout,support_len=support_len)
        
    def forward(self,x,support):
        # TODO：inverse作为x的倒数，能否正确反应负相关关系有待观察
        inverse = 1/(x+100)    # 10为平滑项，防止出现nan值
        x = self.pos_gcn(x,support)
        inverse = self.neg_gcn(inverse,support)
        out = x+0.01*inverse #k0x k1x k2x 中的系数k0 k1 k2，都在各自的gcn中的mlp实现
        
        # TODO: 可能用负x更能表达负相关的意思
        # pos_out = self.pos_gcn(x, support)
        # neg_out = self.neg_gcn(-x, support)
        # return pos_out + neg_out
        
        return out
class Decomposed_temporal_evolution_extractor(nn.Module):   
    def __init__(self):
        super(Decomposed_temporal_evolution_extractor,self).__init__()
        self.seasonal_layer = nn.ModuleList()
        self.seasonal_number = 3
        # 卷积核大小
        for i in range(self.seasonal_number):
            # 计算 padding 的大小
            kernel_size = 2*i+1  # i=1, k=3, p=1
            padding = i
            # 定义卷积层，并设置 padding 参数
            self.seasonal_layer.append(nn.Conv2d(in_channels=1,
                                out_channels=1,
                                kernel_size=(1, kernel_size),
                                padding=(0,padding)))
        self.trend_layer = nn.Linear(12,12)   
    def forward(self,x):
        # x.shape = B out_dim N predict_length = (B 1 N 12)
        y_trend,y_seasonal=0.0,0.0
        for i in range(self.seasonal_number):
            #print(self.seasonal_layer[i](x).shape)
            y_seasonal = y_seasonal + self.seasonal_layer[i](x) 
        y_trend = self.trend_layer(x)
        return y_trend, y_seasonal
class CauSTG(nn.Module):
    def __init__(
            self, 
            device: str, 
            num_nodes: int, 
            dropout: float, 
            # supports: Optional[List], 
            use_gcn: bool, 
            adaptive_adj: bool, 
            init_adj: Optional[List], 
            in_dim: int,
            out_dim: int,
            residual_channels: int,
            dilation_channels: int,
            skip_channels: int,
            end_channels: int,
            kernel_size: int,
            blocks: int,
            layers: int
            ):
        super(CauSTG, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.use_gcn = use_gcn
        self.adaptive_adj = adaptive_adj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        # self.supports = supports

        # 时间感受野（能看到的历史步数）
        self.receptive_field = 1 

        # 已有先验图数量，或者初始化空列表
        self.supports_len = 0
        # if supports is not None:
        #     self.supports_len += len(supports)
        # else:
        #     self.supports = []

        # 学习自适应邻接矩阵，不提供aptinit时，直接初始化两组可训练节点嵌入
        if use_gcn and adaptive_adj:
            if init_adj is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                m, p, n = torch.svd(init_adj)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        # FIXED： 假设我们要感受野精确对齐到12，那么就需要一些定制修改。用下面这个kernel_size, dilation, layer, block
        custom_dilations = [1, 2, 5]
        custom_kernels = [3, 3, 2]
        for b in range(blocks):
            
            for i in range(self.layers):
                # dilated convolutions，卷积在input_length上进行，不影响节点间信息, 计算保持时序维度不变所需要的padding
                # padding_value = (kernel_size - 1) * new_dilation // 2  # 时序维度卷积的输出维度计算公式为： T_out = (T_in + 2*Padding - dilation_n * (kernel_n - 1) -1) / 步长 + 1
                new_dilation = custom_dilations[i]
                curr_kernel = custom_kernels[i]
                self.receptive_field = self.receptive_field + (curr_kernel -1 ) * new_dilation

                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,curr_kernel),dilation=new_dilation, padding='same'))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, curr_kernel), dilation=new_dilation, padding='same'))

                # 1x1 convolution for residual connection
                # NOTE：作者模型的一大败笔，本意是为了在选择use_gcn=False的时候作为兜底机制，作为自适应邻接矩阵的替代
                # NOTE：但是在ddp环境下，因为这个网络的参数有梯度但是没有参与反向传播，会使得ddp报错。
                # self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                #                                      out_channels=residual_channels,
                #                                      kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                # NOTE: 作者模型用不到最后一层的bn和gconv，所以我们不生成了，免得ddp报错
                if i < self.layers-1:
                    self.bn.append(nn.BatchNorm2d(residual_channels))

                if self.use_gcn and i < self.layers-1:
                    self.gconv.append(Bi_directional_spatial_relation_learner(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.decomposed_sea_tre = Decomposed_temporal_evolution_extractor()


    def forward(self, input):
        # input.shape = (num_samples, in_dim, nodes, input_length)
        in_len = input.size(3)

        # 显性检查，最后的感受野与输入的T预测尺度是否匹配
        if in_len != self.receptive_field:
            raise ValueError(f"输入的D4为:{in_len} 并不匹配模型最后的感受野: {self.receptive_field}")

        # x.shape = (num_samples, residual_channels, nodes, input_length)
        x = self.start_conv(input)
        skip = 0

        # 计算迭代邻接矩阵
        new_supports = None
        if self.use_gcn and self.adaptive_adj:  # and self.supports is not None
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = [adp]  # + self.supports

        # encoder
        for i in range(self.blocks * self.layers):
            residual = x

            # 空洞卷积，门控机制，为每个特征通道生成一个动态权重，使信息选择性传递。
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)

            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # skip connection
            s = self.skip_convs[i](x)
            skip = skip + s

            # 权重参数学习节点间空间关系一致性
            if i < self.layers -1:
                if self.use_gcn:  # and self.supports is not None
                    if self.adaptive_adj:
                        x = self.gconv[i](x, new_supports)
                    else:
                        raise RuntimeError(f"[error] 参数设置错误，没有开启自适应邻接矩阵")
                else:
                    raise RuntimeError(f"选择use_gcn=False，此处需要修改！")

                # residual connection
                x = x + residual

                # batch norm
                x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))

        # x最终维度 = B out_dim N predict_length
        x = self.end_conv_2(x) #
        
        # trend, season.shape = B 1 N 12
        trend,season = self.decomposed_sea_tre(x) 
        return x,trend,season

# ====================================================================
# 以下是数据流动追踪脚本
# ====================================================================

if __name__ == "__main__":
    print("--- 开始追踪数据流 ---")
    
    # 1. 模拟输入参数
    device = "cpu"
    batch = 2000 # 批次大小
    history_length = 12 # 输入时序长度
    nodes = 321 # 节点数量
    in_dim = 2 # 输入特征数
    out_dim = 1 # 最终输出特征维度，只要电力负荷
    
    # 模型其他参数 隐藏层维度32
    residual_channels = 32
    dilation_channels = 32
    skip_channels = 256
    end_channels = 512
    kernel_size = 2
    blocks = 4
    layers = 2
    dropout = 0.3
    
    # 2. 实例化模型
    model = CauSTG(
        device=device,
        num_nodes=nodes,
        dropout=dropout,
        use_gcn=True,
        adaptive_adj=True,
        init_adj=None,
        in_dim=in_dim, # 注意：这里的in_dim是通道数
        out_dim=out_dim,
        residual_channels=residual_channels,
        dilation_channels=dilation_channels,
        skip_channels=skip_channels,
        end_channels=end_channels,
        kernel_size=kernel_size,
        blocks=blocks,
        layers=layers
    ).to(device)
    model.eval()

    # 3. 模拟输入数据
    # 你的数据处理过程是 
    input_data = torch.randn(batch, in_dim, nodes, history_length).to(device)
    print(f"原始输入张量维度: {input_data.shape}") # [2000, 2, 321, 12]

    # 4. 逐步执行并打印中间张量维度
    print("\n--- 第一步: start_conv ---")
    x = model.start_conv(input_data)
    print(f"start_conv后的维度: {x.shape}") # 预期: [B, residual_channels, N, T] = [2000, 32, 321, 12]

    # 5. 追踪主循环 (encoder)
    print("\n--- 第二步: 主编码器循环 ---")
    skip = 0
    new_supports = None
    if model.use_gcn and model.adaptive_adj:
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        new_supports = [adp]
    
    for i in range(model.blocks * model.layers):
        residual = x
        print(f"\n--- 循环 {i+1} ---")
        
        # 时序卷积 (扩张卷积)
        filter_tensor = model.filter_convs[i](residual)
        gate_tensor = model.gate_convs[i](residual)
        x = torch.tanh(filter_tensor) * torch.sigmoid(gate_tensor)
        print(f"时序卷积和门控后维度: {x.shape}") # 预期: [B, dilation_channels, N, T] = [2000, 32, 321, 12]
        
        # 跳跃连接
        s = model.skip_convs[i](x)
        try:
            skip = skip[:, :, :,  -s.size(3):]
        except:
            skip = 0
        skip = s + skip
        print(f"跳跃连接 s 维度: {s.shape}") # 预期: [B, skip_channels, N, T] = [2000, 256, 321, 12]

        # 空间卷积 (GCN)
        if model.use_gcn:
            x = model.gconv[i](x, new_supports)
            print(f"GCN后维度: {x.shape}") # 预期: [B, residual_channels, N, T] = [2000, 32, 321, 12]
        else:
            x = model.residual_convs[i](x)
            print(f"无GCN时维度: {x.shape}") # 预期: [B, residual_channels, N, T] = [2000, 32, 321, 12]

        x = x + residual[:, :, :, -x.size(3):]
        x = model.bn[i](x)
        print(f"残差连接和BN后维度: {x.shape}") # 预期: [B, residual_channels, N, T] = [2000, 32, 321, 12]

    # 6. 追踪末端层
    print("\n--- 第三步: 末端层 ---")
    x = F.relu(skip)
    x = F.relu(model.end_conv_1(x))
    print(f"end_conv_1后维度: {x.shape}") # 预期: [B, end_channels, N, T] = [2000, 512, 321, 12]

    output = model.end_conv_2(x)
    print(f"end_conv_2后最终维度: {output.shape}") # 预期: [B, out_dim, N, T] = [2000, 1, 321, 12]

    # 7. 追踪分解模块
    print("\n--- 第四步: 分解模块 ---")
    # 输入维度为 [B, out_dim, N, T]，其中out_dim = 1， T = 12
    trend, season = model.decomposed_sea_tre(output) # 对预测的得到的电力负荷进行分解，以便后续通过时间层面的损失来增强网络性能
    print(f"分解后趋势分量维度: {trend.shape}") # 预期: [B, 1, N, 12] = [2000, 1, 321, 12]
    print(f"分解后季节分量维度: {season.shape}") # 预期: [B, 1, N, 12] = [2000, 1, 321, 12]

    print("\n--- 数据流追踪完成 ---")


"""
感受野计算 RF receptive field  block=1， layer=3 kernel_size=3
RF_n = RF_n-1 + (k_n - 1) * d_n * (s_1 *...* s_n-1) 由于s_i 为第i层的步长始终为1，因此简化为
RF_n = RF_n-1 + (k_n - 1) * d_n； k_n 为第n层卷积核的大小，d_n 为第n层的空洞率
初始值 RF_0 = 1, k_n = 3, d_n = 1

Block 1
- 初始感受野 RF_0 = 1
- Layer 1: k1 = 3, d1 = 1
RF_1 = 1+(3-1)*1 = 3
- Layer 2: k2 = 3, d2 = 2
RF_2 = 3 + (3-1) * 2 = 7
- Layer 3: k2 = 3, d2 = 4
RF_3 = 7 + (3-1) * 4 = 15
"""









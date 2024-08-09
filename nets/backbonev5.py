# import torch
# import torch.nn as nn

# # 定义卷积块Conv
# def conv(in_channels, out_channels, kernel_size, stride, padding, depthwise=False, act=None):
#     layers = []
#     layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels if depthwise else 1, bias=False))
#     layers.append(nn.BatchNorm2d(out_channels))
#     if depthwise:
#         layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
#         layers.append(nn.BatchNorm2d(out_channels))
#     if act == "silu":
#         layers.append(nn.SiLU())
#     elif act == "relu":
#         layers.append(nn.ReLU())
#     return nn.Sequential(*layers)

# # 定义C3模块
# class C3(nn.Module):
#     def __init__(self, in_channels, out_channels, n=1, shortcut=True, depthwise=False, act=None):
#         super().__init__()
#         self.conv1 = conv(in_channels, out_channels, 1, 1, 0, depthwise=depthwise, act=act)
#         self.conv2 = conv(out_channels, out_channels * 2, 3, 1, 1, depthwise=depthwise, act=act)
#         self.conv3 = conv(out_channels * 2, out_channels, 1, 1, 0, depthwise=False, act=act)  # 这里不使用深度可分离卷积
#         self.shortcut = shortcut and in_channels == out_channels

#     def forward(self, x):
#         return x + self.conv3(self.conv2(self.conv1(x))) if self.shortcut else self.conv3(self.conv2(self.conv1(x)))
    
    
# class BottleneckCSP(nn.Module):
#     def __init__(self, in_channels, out_channels, n=1, shortcut=True, depthwise=False, act=None):
#         super().__init__()
#         hidden_channels = out_channels // 2
#         self.conv1 = conv(in_channels, hidden_channels, 1, 1, 0, depthwise=depthwise, act=act)
#         self.conv2 = conv(in_channels, hidden_channels, 1, 1, 0, depthwise=depthwise, act=act)
#         self.conv3 = conv(hidden_channels, hidden_channels, 3, 1, 1, depthwise=depthwise, act=act)
#         self.conv4 = conv(2 * hidden_channels, out_channels, 1, 1, 0, depthwise=depthwise, act=act)
#         self.bn = nn.BatchNorm2d(2 * hidden_channels)
#         self.act = nn.SiLU() if act == "silu" else nn.ReLU()
#         self.n = n

#     def forward(self, x):
#         y1 = self.conv1(x)
#         y2 = self.conv2(x)
#         for _ in range(self.n):
#             y1 = self.conv3(y1)
#         return self.conv4(torch.cat((y1, y2), dim=1))


# # 定义SPPF模块
# class SPPF(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), depthwise=False, act=None):
#         super().__init__()
#         self.conv1 = conv(in_channels, out_channels, 1, 1, 0, depthwise=depthwise, act=act)
#         self.conv2 = conv(out_channels * (len(kernel_sizes) + 1), out_channels, 3, 1, 1, depthwise=depthwise, act=act)
#         self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes])

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = torch.cat([x1] + [pool(x1) for pool in self.maxpools], dim=1)
#         return self.conv2(x2)

# class v5(nn.Module):
#     def __init__(self, dep_mul=1.0, wid_mul=1.0, depthwise=False, act="silu"):
#         super().__init__()
#         # self.conv1 = conv(3, int(64 * wid_mul), 6, 2, 2, depthwise=depthwise, act=act)  # 0-P1/2
#         # self.conv2 = conv(int(64 * wid_mul), int(128 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)  # 1-P2/4
#         # self.c3_1 = C3(int(128 * wid_mul), int(128 * wid_mul), n=int(3 * dep_mul), depthwise=depthwise, act=act)  # 修改为128输出通道
#         # self.conv3 = conv(int(128 * wid_mul), int(256 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)  # 3-P3/8
#         # self.c3_2 = nn.Sequential(*[C3(int(256 * wid_mul), int(256 * wid_mul), n=int(6 * dep_mul), depthwise=depthwise, act=act) for _ in range(6)])  # 修改为256输出通道
#         # self.conv4 = conv(int(256 * wid_mul), int(512 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)  # 5-P4/16
#         # self.c3_3 = nn.Sequential(*[C3(int(512 * wid_mul), int(512 * wid_mul), n=int(3 * dep_mul), depthwise=depthwise, act=act) for _ in range(3)])
#         # self.conv5 = conv(int(512 * wid_mul), int(1024 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)  # 7-P5/32
#         # self.sppf = SPPF(int(1024 * wid_mul), int(1024 * wid_mul), kernel_sizes=(5, 9, 13), depthwise=depthwise, act=act)  # 9
#         self.conv1 = conv(3, int(64 * wid_mul), 6, 2, 2, depthwise=depthwise, act=act)  # 0-P1/2
#         self.conv2 = conv(int(64 * wid_mul), int(128 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)  # 1-P2/4
#         self.c3_1 = BottleneckCSP(int(128 * wid_mul), int(128 * wid_mul), n=int(3 * dep_mul), depthwise=depthwise, act=act)  # 修改为128输出通道
#         self.conv3 = conv(int(128 * wid_mul), int(256 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)  # 3-P3/8
#         self.c3_2 = nn.Sequential(*[BottleneckCSP(int(256 * wid_mul), int(256 * wid_mul), n=int(6 * dep_mul), depthwise=depthwise, act=act) for _ in range(6)])  # 修改为256输出通道
#         self.conv4 = conv(int(256 * wid_mul), int(512 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)  # 5-P4/16
#         self.c3_3 = nn.Sequential(*[BottleneckCSP(int(512 * wid_mul), int(512 * wid_mul), n=int(3 * dep_mul), depthwise=depthwise, act=act) for _ in range(3)])
#         self.conv5 = conv(int(512 * wid_mul), int(1024 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)  # 7-P5/32
#         self.sppf = SPPF(int(1024 * wid_mul), int(1024 * wid_mul), kernel_sizes=(5, 9, 13), depthwise=depthwise, act=act)  # 9

#     def forward(self, x):
#         outputs = {}
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.c3_1(x)
#         x = self.conv3(x)
#         outputs["dark3"] = x  # 添加dark3输出
#         x = self.c3_2(x)
#         x = self.conv4(x)
#         outputs["dark4"] = x  # 添加dark4输出
#         x = self.c3_3(x)
#         x = self.conv5(x)
#         x = self.sppf(x)
#         outputs["dark5"] = x  # 添加dark5输出
#         return {k: v for k, v in outputs.items() if k in ("dark3", "dark4", "dark5")}

# # # 创建模型实例
# # model = Backbone(dep_mul=1.0, wid_mul=1.0, depthwise=False, act="silu")

# # # 设置为评估模式
# # model.eval()

# # # 伪造一个输入张量
# # dummy_input = torch.randn(1, 3, 640, 640)  # 批大小为1,通道数为3,分辨率为640x640

# # # 前向传播
# # outputs = model(dummy_input)

# # # 打印输出特征图的形状
# # for name, output in outputs.items():
# #     print(f"{name} shape: {output.shape}")

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride, padding, depthwise=False, act=None):
    layers = []
    layers.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels if depthwise else 1, bias=False)
    )
    layers.append(nn.BatchNorm2d(out_channels))
    if depthwise:
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
    if act == "silu":
        layers.append(nn.SiLU())
    elif act == "relu":
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class BottleneckCSP(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, depthwise=False, act=None):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = conv(in_channels, hidden_channels, 1, 1, 0, depthwise=depthwise, act=act)
        self.conv2 = conv(in_channels, hidden_channels, 1, 1, 0, depthwise=depthwise, act=act)
        self.conv3 = conv(hidden_channels, hidden_channels, 3, 1, 1, depthwise=depthwise, act=act)
        self.conv4 = conv(2 * hidden_channels, out_channels, 1, 1, 0, depthwise=depthwise, act=act)
        self.bn = nn.BatchNorm2d(2 * hidden_channels)
        self.act = nn.SiLU() if act == "silu" else nn.ReLU()
        self.n = n

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        for _ in range(self.n):
            y1 = self.conv3(y1)
        return self.conv4(torch.cat((y1, y2), dim=1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), depthwise=False, act=None):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, 1, 1, 0, depthwise=depthwise, act=act)
        self.conv2 = conv(out_channels * (len(kernel_sizes) + 1), out_channels, 3, 1, 1, depthwise=depthwise, act=act)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat([x1] + [pool(x1) for pool in self.maxpools], dim=1)
        return self.conv2(x2)

class GameTheoreticAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(GameTheoreticAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values_payoff = nn.Linear(self.head_dim, 1, bias=False)
        self.keys_payoff = nn.Linear(self.head_dim, 1, bias=False)
        self.queries_payoff = nn.Linear(self.head_dim, 1, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values_payoff = self.values_payoff(values)
        keys_payoff = self.keys_payoff(keys)
        queries_payoff = self.queries_payoff(queries)

        values_prob = F.softmax(values_payoff, dim=1)
        keys_prob = F.softmax(keys_payoff, dim=1)
        queries_prob = F.softmax(queries_payoff, dim=1)

        values_weighted = values * values_prob
        keys_weighted = keys * keys_prob
        queries_weighted = queries * queries_prob

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries_weighted, keys_weighted])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values_weighted]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class v5(nn.Module):
    def __init__(self, dep_mul=1.0, wid_mul=1.0, depthwise=False, act="silu"):
        super().__init__()
        self.conv1 = conv(3, int(64 * wid_mul), 6, 2, 2, depthwise=depthwise, act=act)
        self.conv2 = conv(int(64 * wid_mul), int(128 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)
        self.c3_1 = BottleneckCSP(int(128 * wid_mul), int(128 * wid_mul), n=int(3 * dep_mul), depthwise=depthwise, act=act)
        self.conv3 = conv(int(128 * wid_mul), int(256 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)
        self.c3_2 = nn.Sequential(
            *[BottleneckCSP(int(256 * wid_mul), int(256 * wid_mul), n=int(6 * dep_mul), depthwise=depthwise, act=act)
              for _ in range(6)])
        self.conv4 = conv(int(256 * wid_mul), int(512 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)
        self.c3_3 = nn.Sequential(
            *[BottleneckCSP(int(512 * wid_mul), int(512 * wid_mul), n=int(3 * dep_mul), depthwise=depthwise, act=act)
              for _ in range(3)])
        self.conv5 = conv(int(512 * wid_mul), int(1024 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)
        self.sppf = SPPF(int(1024 * wid_mul), int(1024 * wid_mul), kernel_sizes=(5, 9, 13), depthwise=depthwise, act=act)

        self.gta3 = GameTheoreticAttention(int(256 * wid_mul), 8)
        self.gta4 = GameTheoreticAttention(int(512 * wid_mul), 8)
        # self.gta5 = GameTheoreticAttention(int(1024 * wid_mul), 8)  # 注释掉这行

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        x = self.conv3(x)
        outputs["dark3"] = x  # 添加dark3输出

        x = self.c3_2(x)
        N, C, H, W = x.shape
        x = x.reshape(N, C, H * W).permute(0, 2, 1)
        x = self.gta3(x, x, x).permute(0, 2, 1).reshape(N, C, H, W)
        x = self.conv4(x)
        outputs["dark4"] = x  # 添加dark4输出

        x = self.c3_3(x)
        N, C, H, W = x.shape
        x = x.reshape(N, C, H * W).permute(0, 2, 1)
        x = self.gta4(x, x, x).permute(0, 2, 1).reshape(N, C, H, W)
        x = self.conv5(x)
        x = self.sppf(x)
        outputs["dark5"] = x  # 添加dark5输出

        # 去掉dark5的GameTheoreticAttention
        # N, C, H, W = x.shape
        # x = x.reshape(N, C, H * W).permute(0, 2, 1)
        # x = self.gta5(x, x, x).permute(0, 2, 1).reshape(N, C, H, W)

        return {k: v for k, v in outputs.items() if k in ("dark3", "dark4", "dark5")}

# 示例用法
# model = Backbone()
# input_data = torch.randn(1, 3, 320, 320)
# outputs = model(input_data)
# for key, value in outputs.items():
#     print(f"{key} shape:", value.shape)

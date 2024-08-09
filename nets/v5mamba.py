import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, stride, padding, depthwise=True, act=None):
    layers = []
    if depthwise:
        layers.append(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        )
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
    else:
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

    if act == "silu":
        layers.append(nn.SiLU())
    elif act == "relu":
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BottleneckCSP(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, depthwise=True, act=None):
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
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), depthwise=True, act=None):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, 1, 1, 0, depthwise=depthwise, act=act)
        self.conv2 = conv(out_channels * (len(kernel_sizes) + 1), out_channels, 3, 1, 1, depthwise=False, act=act)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat([x1] + [pool(x1) for pool in self.maxpools], dim=1)
        return self.conv2(x2)


class C3(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, depthwise=True, act=None):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, 1, 1, 0, depthwise=depthwise, act=act)
        self.conv2 = conv(out_channels, out_channels * 2, 3, 1, 1, depthwise=depthwise, act=act)
        self.conv3 = conv(out_channels * 2, out_channels, 1, 1, 0, depthwise=False, act=act)  # 这里不使用深度可分离卷积
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv3(self.conv2(self.conv1(x))) if self.shortcut else self.conv3(self.conv2(self.conv1(x)))


class v5mamba(nn.Module):
    def __init__(self, dep_mul=1.0, wid_mul=1.0, depthwise=True, act="silu"):
        super().__init__()
        self.conv1 = conv(3, int(64 * wid_mul), 6, 2, 2, depthwise=False, act=act)  # 将depthwise设置为False
        self.conv2 = conv(int(64 * wid_mul), int(128 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)
        self.c3_1 = C3(int(128 * wid_mul), int(128 * wid_mul), n=int(3 * dep_mul), depthwise=depthwise, act=act)
        self.conv3 = conv(int(128 * wid_mul), int(256 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)
        self.c3_2 = nn.Sequential(
            *[C3(int(256 * wid_mul), int(256 * wid_mul), n=int(6 * dep_mul), depthwise=depthwise, act=act)
              for _ in range(6)])
        self.conv4 = conv(int(256 * wid_mul), int(512 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)
        self.c3_3 = nn.Sequential(
            *[C3(int(512 * wid_mul), int(512 * wid_mul), n=int(3 * dep_mul), depthwise=depthwise, act=act)
              for _ in range(3)])
        self.conv5 = conv(int(512 * wid_mul), int(1024 * wid_mul), 3, 2, 1, depthwise=depthwise, act=act)
        self.sppf = SPPF(int(1024 * wid_mul), int(1024 * wid_mul), kernel_sizes=(5, 9, 13), depthwise=depthwise,
                         act=act)

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        x = self.conv3(x)
        outputs["dark3"] = x  # 添加dark3输出

        x = self.c3_2(x)
        x = self.conv4(x)
        outputs["dark4"] = x  # 添加dark4输出

        x = self.c3_3(x)
        x = self.conv5(x)
        x = self.sppf(x)
        outputs["dark5"] = x  # 添加dark5输出

        return {k: v for k, v in outputs.items() if k in ("dark3", "dark4", "dark5")}


# # 测试代码
# if __name__ == "__main__":
#     model = v5()
#     x = torch.randn(1, 3, 640, 640)
#     outputs = model(x)
#     for k, v in outputs.items():
#         print(f"{k}: {v.shape}")

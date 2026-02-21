import torch
from torch import nn
from typing import Optional, Callable, Tuple

class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

class DDPM(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, kernel_size=3, down_factor=4):
        """DDPM，利用nn.Unfold实现的动态卷积模块

        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            kernel_size (int): 指定的生成的卷积核的大小
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DDPM, self).__init__()
        self.kernel_size = kernel_size
        self.mid_c = out_C // 4
        self.down_input = nn.Conv2d(in_xC, self.mid_c, 1)
        self.branch_1 = DepthDC3x3_1(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.branch_3 = DepthDC3x3_3(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.branch_5 = DepthDC3x3_5(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.fuse = BasicConv2d(4 * self.mid_c, out_C, 3, 1, 1)

    def forward(self, x, y):
        x = self.down_input(x)
        result_1 = self.branch_1(x, y)
        result_3 = self.branch_3(x, y)
        result_5 = self.branch_5(x, y)
        return self.fuse(torch.cat((x, result_1, result_3, result_5), dim=1))


class DepthDC3x3_1(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        """DepthDC3x3_1，利用nn.Unfold实现的动态卷积模块

        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DepthDC3x3_1, self).__init__()
        self.kernel_size = 3
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


class DepthDC3x3_3(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        """DepthDC3x3_3，利用nn.Unfold实现的动态卷积模块

        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DepthDC3x3_3, self).__init__()
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.kernel_size = 3
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=3, padding=3, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


class DepthDC3x3_5(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        """DepthDC3x3_5，利用nn.Unfold实现的动态卷积模块

        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DepthDC3x3_5, self).__init__()
        self.kernel_size = 3
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=5, padding=5, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


class DDPMLSTM2d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format.
    """
    def __init__(self,
                 in_xC: int,
                 in_yC: int,
                 out_C: int,
                 kernel_size=3, 
                 down_factor=4,
                 ):
        super().__init__()
        self.dim = in_xC

        xh_dim = out_C
        gates_dim = out_C * 4

        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        self.ddpm = DDPM(in_xC, in_yC, out_C, kernel_size, down_factor)

    def forward(self, x: torch.Tensor, h_and_c_previous: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W))
        :return: ((N C H W), (N C H W))
        """
        if h_and_c_previous is None:
            # generate zero states
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        xh = self.ddpm(h_tm1, x)
        mix = self.conv1x1(xh)

        gates, cell_input = torch.tensor_split(mix, [self.dim * 3], dim=1)
        assert gates.shape[1] == cell_input.shape[1] * 3

        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)
        assert forget_gate.shape == input_gate.shape == output_gate.shape

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * torch.tanh(c_t)

        return h_t, c_t


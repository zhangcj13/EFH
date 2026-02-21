import torch
import torch.nn as nn

from .transformer import LayerNorm2d

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


## Few Shot Medical Image Segmentation with Cross Attention Transformer
## https://github.com/hust-linyi/CAT-Net
class CrossAttention(nn.Module):
    def __init__(self, dim, single_out=False):
        super(CrossAttention, self).__init__()
        self.query = nn.Conv2d(dim, dim // 8, 1)
        self.key = nn.Conv2d(dim, dim // 8, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        # self.norm = nn.LayerNorm([[256, 32, 32]])
        self.norm = LayerNorm2d(dim)

        self.single_out = single_out

    def forward(self, x, y):
        B, C, H, W = x.shape
        scale = (C // 8) ** -0.5

        qx = self.query(x).view(B, -1, H * W).permute(0, 2, 1) * scale  # B, H*W, C'
        ky = self.key(y).view(B, -1, H * W)  # B, C', H*W
        vy = self.value(y).view(B, -1, H * W)  # B, C, H*W
        attn = self.softmax(torch.bmm(qx, ky))  # B, H*W, H*W
        outx = torch.bmm(vy, attn.permute(0, 2, 1)).view(B, C, H, W)  # B, C, H, W
        outx = self.mlp(outx.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # Apply MLP and permute back
        outx = self.norm(outx)  # Apply normalization

        if self.single_out:
            return outx

        qy = self.query(y).view(B, -1, H * W).permute(0, 2, 1) * scale  # B, H*W, C'
        kx = self.key(x).view(B, -1, H * W)  # B, C', H*W
        vx = self.value(x).view(B, -1, H * W)  # B, C, H*W
        attn = self.softmax(torch.bmm(qy, kx))  # B, H*W, H*W
        outy = torch.bmm(vx, attn.permute(0, 2, 1)).view(B, C, H, W)  # B, C, H, W
        outy = self.mlp(outy.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # Apply MLP and permute back
        outy = self.norm(outy)  # Apply normalization

        return outx, outy
    

# -*- encoding: utf-8 -*-
'''
@File    :   cmtf_head.py
@Time    :   2024/04/28 13:32:24
@Author  :   GauthierLi 
@Version :   1.0
@Contact :   lwklxh@163.com
@License :   Copyright (C) 2024 GauthierLi, All rights reserved.
'''

'''
Description here ...
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

# modules
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv0_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim //fc_ratio)
        self.bn0_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn0_3 = nn.BatchNorm2d(dim)

        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim // fc_ratio)
        self.bn1_2 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn1_3 = nn.BatchNorm2d(dim)

        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn2_2 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv2_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn2_3 = nn.BatchNorm2d(dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()

        attn0_1 = self.relu(self.bn0_1(self.conv0_1(x)))
        attn0_2 = self.relu(self.bn0_2(self.conv0_2(attn0_1)))
        attn0_3 = self.relu(self.bn0_3(self.conv0_3(attn0_2)))

        attn1_2 = self.relu(self.bn1_2(self.conv1_2(attn0_1)))
        attn1_3 = self.relu(self.bn1_3(self.conv1_3(attn1_2)))

        attn2_2 = self.relu(self.bn2_2(self.conv2_2(attn0_1)))
        attn2_3 = self.relu(self.bn2_3(self.conv2_3(attn2_2)))

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.relu(self.bn3(self.conv3(attn)))
        attn = attn * u

        pool = self.Avg(attn)

        return pool


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads, atten_drop = 0., proj_drop = 0., dilation = [3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim//fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn
    
class E_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.):
        super(E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=ksize,
                                groups=hidden_features)
        self.conv2 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                                groups=hidden_features)
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.fc2(x1 + x2)
        x = self.act(x)

        return x

class Block(nn.Module):
    def __init__(self, dim=512, num_heads=16,  mlp_ratio=4, pool_ratio=16, drop=0., dilation=[3, 5, 7],
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Mutilscal_MHSA(dim, num_heads=num_heads, atten_drop=drop, proj_drop=drop, dilation=dilation,
                                   pool_ratio=pool_ratio, fc_ratio=mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)

        self.mlp = E_FFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                         drop=drop)

    def forward(self, x):

        x = x + self.drop_path(self.norm1(self.attn(x)))
        x = x + self.drop_path(self.mlp(x))

        return x


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class MAF(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[3, 5, 7], dropout=0., num_classes=6):
        super(MAF, self).__init__()

        self.conv0 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0 = nn.BatchNorm2d(dim//fc_ratio)

        self.conv1_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim//fc_ratio)
        self.bn1_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn1_2 = nn.BatchNorm2d(dim)

        self.conv2_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim//fc_ratio)
        self.bn2_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn2_2 = nn.BatchNorm2d(dim)

        self.conv3_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn3_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv3_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn3_2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.bn4 = nn.BatchNorm2d(dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(256, num_classes, kernel_size=1))

    def forward(self, x):
        u = x.clone()

        attn1_0 = self.relu(self.bn0(self.conv0(x)))
        attn1_1 = self.relu(self.bn1_1(self.conv1_1(attn1_0)))
        attn1_1 = self.relu(self.bn1_2(self.conv1_2(attn1_1)))
        attn1_2 = self.relu(self.bn2_1(self.conv2_1(attn1_0)))
        attn1_2 = self.relu(self.bn2_2(self.conv2_2(attn1_2)))
        attn1_3 = self.relu(self.bn3_1(self.conv3_1(attn1_0)))
        attn1_3 = self.relu(self.bn3_2(self.conv3_2(attn1_3)))

        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn

        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn

        attn = attn1_1 + attn1_2 + attn1_3
        attn = self.relu(self.bn4(self.conv4(attn)))
        attn = u * attn

        out = self.head(attn + c_attn + s_attn)

        return out

@MODELS.register_module()
class CMTFHead(BaseDecodeHead):
    def __init__(self, 
                 in_channels=[256, 512, 1024, 2048], 
                 channels=512, 
                 dilation=[[1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]],
                 img_size=(512, 512),
                 fc_ratio=4,
                 dropout_ratio=0.1,
                 num_classes=6,
                 **kwargs):
        super().__init__(in_channels=in_channels, 
                         channels=channels,
                         num_classes=num_classes,
                         dropout_ratio=dropout_ratio,
                         **kwargs)
        self.Conv1 = ConvBNReLU(in_channels[-1], channels, 1)
        self.Conv2 = ConvBNReLU(in_channels[-2], channels, 1)
        self.b4 = Block(dim=channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[0])
        
        self.p3 = Fusion(channels)
        self.b3 = Block(dim=channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[1])

        self.p2 = Fusion(channels)
        self.b2 = Block(dim=channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[2])

        self.Conv3 = ConvBN(in_channels[-3], in_channels[-4], 1)

        self.p1 = Fusion(in_channels[-4])
        self.seg_head = MAF(in_channels[-4], fc_ratio=fc_ratio, dilation=dilation[3], 
                            dropout=dropout_ratio, num_classes=num_classes)

        self.pooling2 = nn.MaxPool2d(2)
        self.pooling4 = nn.MaxPool2d(4)
        self.img_size = img_size
        self.init_weight()
        
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, inputs):
        # import pdb; pdb.set_trace()
        res1, res2, res3, res4 = inputs
        res3 = self.pooling2(res3)
        res4 = self.pooling4(res4)
        
        h, w = self.img_size
        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)

        x = self.b4(res4)

        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.Conv3(x)
        x = self.p1(x, res1)

        x = self.seg_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

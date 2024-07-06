# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:33:22 2023

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, repeat, unpack

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.batchnorm(self.cnn(x)))

class dil_conv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(dil_conv, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, dilation=2, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.batchnorm(self.cnn(x))

## from bot50
class MHSA(nn.Module):
    def __init__(self, n_dims, width, height, head, num_register_tokens=1):
        super(MHSA, self).__init__()
        self.head = head

        self.Q = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.K = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.V = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        
        self.rel_h = nn.Parameter(torch.randn([1, head, n_dims // head, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, head, n_dims // head, width, 1]), requires_grad=True)
        
        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, width*height, width*height))
        self.register_tokens_v = nn.Parameter(torch.randn(num_register_tokens, n_dims // head, width*height))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        identity = x
        n_batch, C, width, height = x.size()
        
        q = self.Q(x).view(n_batch, self.head, C // self.head, -1)
        k = self.K(x).view(n_batch, self.head, C // self.head, -1)
        v = self.V(x).view(n_batch, self.head, C // self.head, -1)
        
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.head, C // self.head, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)
        energy = content_content + content_position
        
        # Repeat the register tokens across the batch, width, and height dimensions
        r_qk = repeat(
            self.register_tokens, 
            'n w h -> b n w h', 
            b=n_batch
        )
        
        r_v = repeat(
            self.register_tokens_v, 
            'n w h -> b n w h', 
            b=n_batch
        )

        energy, _ = pack([energy, r_qk], 'b * w h')
        v, ps = pack([v, r_v], 'b * d h')
        
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out, _ = unpack(out, ps, 'b * d h')
        out = out.view(n_batch, C, width, height)
        out += identity
        return out

# x = torch.rand(1, 196, 32, 32)
# model = MHSA(196)
# print(model(x).shape)

class iia_block(nn.Module):
    def __init__(self, in_channels, out_dil, out_mb, out_dims, H, W, head, ratio=1/8):
        expand = out_mb * 4
        super(iia_block, self).__init__()
        
        c_ratio = int(in_channels * ratio)
        self.c_ratio = c_ratio
        
        self.dil_conv = nn.Sequential(
            dil_conv(c_ratio, out_dil, kernel_size=3, padding=2))
        self.relu_dil = nn.ReLU()
        
        self.mb_conv = nn.Sequential(
            conv_block(c_ratio*6, expand, kernel_size=1),
            conv_block(expand, expand, kernel_size=3, groups=expand, padding=1),
            nn.Conv2d(expand, out_mb, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_mb))
        self.relu_mb = nn.ReLU()
        
        self.MHSA = nn.Sequential(
            MHSA(c_ratio, H, W, head),
            nn.Conv2d(c_ratio, out_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dims))
        self.relu_mhsa = nn.ReLU()

        self.shortcut_dil = nn.Sequential(
            nn.Conv2d(c_ratio, out_dil, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dil))
        
        self.shortcut_mb = nn.Sequential(
            nn.Conv2d(c_ratio*6, out_mb, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_mb))
        
        self.shortcut_MHSA = nn.Sequential(
            nn.Conv2d(c_ratio, out_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dims))
        
    def forward(self, x):
        x1, x2, x3 = torch.split(x, (self.c_ratio, self.c_ratio*6, self.c_ratio), dim=1)
        
        x1 = self.shortcut_dil(x1) + self.dil_conv(x1)
        x1 = self.relu_dil(x1)
        
        x2 = self.shortcut_mb(x2) + self.mb_conv(x2)
        x2 = self.relu_mb(x2)
        
        x3 = self.shortcut_MHSA(x3) + self.MHSA(x3)
        x3 = self.relu_mhsa(x3)
        
        return torch.cat([x1, x2, x3], 1)

class ECA(nn.Module):
    def __init__(self, in_channels):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # Compute channel attention weights
        avg_pool = self.avg_pool(x)
        avg_pool = self.conv(avg_pool)
        channel_attention = self.sigmoid(avg_pool)

        # Apply channel attention weights
        output = x * channel_attention

        # Perform downsampling using strided convolution
        output = F.max_pool2d(output, kernel_size=2, stride=2)

        return output
    
class iiANET(nn.Module):
    def __init__(self, in_channels=3, num_classes=37):
        super(iiANET, self).__init__()
        
        self.conv_a = conv_block(3, 32, kernel_size=3, stride=2)
        self.conv_b = conv_block(32, 64, kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.stage1 = nn.Sequential(
            iia_block(64, 20, 120, 20, 73, 73, 1),
            iia_block(160, 24, 144, 24, 73, 73, 1),
              ECA(192))
        
        self.stage2 = nn.Sequential(
            iia_block(192, 32, 192, 32, 36, 36, 2),
            iia_block(256, 36, 216, 36, 36, 36, 2),
            iia_block(288, 40, 240, 40, 36, 36, 2),
            ECA(320))
        
        self.stage3 = nn.Sequential(
            iia_block(320, 56, 336, 56, 18, 18, 4),
            iia_block(448, 80, 480, 80, 18, 18, 4),
            iia_block(640, 80, 480, 80, 18, 18, 4),
            iia_block(640, 88, 528, 88, 18, 18, 4),
            iia_block(704, 96, 576, 96, 18, 18, 4),
            ECA(768))
        
        self.stage4 = nn.Sequential(
            iia_block(768, 128, 768, 128, 9, 9, 8),
            iia_block(1024, 160, 960, 160, 9, 9, 8))
        
        self.avgpool = nn.AvgPool2d(kernel_size=9, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1280, num_classes)
    
    def forward(self,x):
        x = self.conv_a(x)
        x = self.conv_b(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def main():
    x = torch.randn(1, 3, 299, 299)
    model = iiANET()
    output = model(x)
    print(output.shape)

if __name__ == "__main__":
    main()
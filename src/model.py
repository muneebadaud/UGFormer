from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn * self.temperature.view(1, self.num_heads, 1, 1)
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.view(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, expansion_factor=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, expansion_factor=expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.conv(x))


class UncertaintyGuidedTransformerUIE(nn.Module):
    
    def __init__(self, dim=48, enc_blocks=(2, 4, 6), dec_blocks=(2, 2), heads=(1, 2, 4)):
        super().__init__()
        self.stem = nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=False)

        self.enc1 = nn.Sequential(*[TransformerBlock(dim, num_heads=heads[0]) for _ in range(enc_blocks[0])])
        self.down1 = Downsample(dim, dim * 2)

        self.enc2 = nn.Sequential(*[TransformerBlock(dim * 2, num_heads=heads[1]) for _ in range(enc_blocks[1])])
        self.down2 = Downsample(dim * 2, dim * 4)

        self.enc3 = nn.Sequential(*[TransformerBlock(dim * 4, num_heads=heads[2]) for _ in range(enc_blocks[2])])

        self.up2 = Upsample(dim * 4, dim * 2)
        self.fuse2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.dec2 = nn.Sequential(*[TransformerBlock(dim * 2, num_heads=heads[1]) for _ in range(dec_blocks[0])])

        self.up1 = Upsample(dim * 2, dim)
        self.fuse1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.dec1 = nn.Sequential(*[TransformerBlock(dim, num_heads=heads[0]) for _ in range(dec_blocks[1])])

        self.recon_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 3, kernel_size=3, padding=1, bias=True),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1, bias=True),
        )

    def _match_size(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        x = self.stem(x)
        f1 = self.enc1(x)
        x = self.down1(f1)

        f2 = self.enc2(x)
        x = self.down2(f2)

        f3 = self.enc3(x)

        x = self.up2(f3)
        x = self._match_size(x, f2)  
        x = self.fuse2(torch.cat([x, f2], dim=1))
        x = self.dec2(x)

        x = self.up1(x)
        x = self._match_size(x, f1)  
        x = self.fuse1(torch.cat([x, f1], dim=1))
        feat = self.dec1(x)

        pred = torch.sigmoid(self.recon_head(feat))
        
        log_var = self.uncertainty_head(feat)
        log_var = torch.clamp(log_var, -5.0, 5.0)
        return pred, log_var

    

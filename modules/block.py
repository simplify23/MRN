import torch
import torch.nn as nn
import einops
import torch
import torch.nn as nn


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()

        self.norm = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        if v.dim()==4:
            v = v.permute(0, 1, 3, 2).contiguous()
        elif v.dim()==3:
            v = v.permute(0, 2, 1)
        v = self.proj(v)
        if v.dim()==4:
            v = v.permute(0, 1, 3, 2).contiguous()
        elif v.dim()==3:
            v = v.permute(0, 2, 1)
        return u * v


class GatingMlpBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)
    def forward(self, x):
        # if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
        #     return x

        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut

class GatingMlpBlockv2(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len,taski):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len)
        self.spatial_gating_unit2 = SpatialGatingUnit(d_model, taski)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)
        self.proj_3 = nn.Linear(d_model // 2, d_model)
    def forward(self, x):
        # if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
        #     return x
        # B, H, W, C = x.shape
        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        x = self.spatial_gating_unit2(x.permute(0,2,1,3)).permute(0,2,1,3)
        x = self.proj_3(x)
        return x + shorcut

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if ratio > 0 :
            self.shared_MLP = nn.Sequential(
                nn.Conv2d(channel, channel // ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channel // ratio, channel, 1, bias=False)
            )
        else:
            self.shared_MLP = nn.Sequential(
                nn.Conv2d(channel, channel * -ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channel * -ratio, channel, 1, bias=False)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.fusion = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1).permute(0,2,1)
        out = self.sigmoid(self.fusion(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel,ratio=16):
        super(CBAM, self).__init__()
        self.ratio = ratio
        self.channel_attention = ChannelAttentionModule(channel,ratio)
        self.spatial_attention = SpatialAttentionModule()
        if ratio < 0:
            self.down = nn.Conv2d(channel, 1, 1, bias=False)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
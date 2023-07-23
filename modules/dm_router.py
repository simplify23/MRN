import torch.nn as nn
from einops import rearrange

class SpatialDomainGating(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()

        self.norm = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return u * v

class ChannelDomainGating(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()

        self.norm = nn.LayerNorm(d_ffn)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        # b w (i c)
        # x = u
        v = self.norm(x)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return x * v

# class GatingMlpBlock(nn.Module):
#     def __init__(self, d_model, d_ffn, seq_len):
#         super().__init__()
#
#         self.norm = nn.LayerNorm(d_model)
#         self.proj_1 = nn.Linear(d_model, d_ffn)
#         self.activation = nn.GELU()
#         self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len)
#         self.proj_2 = nn.Linear(d_ffn // 2, d_model)
#     def forward(self, x):
#         shorcut = x.clone()
#         x = self.norm(x)
#         x = self.proj_1(x)
#         x = self.activation(x)
#         x = self.spatial_gating_unit(x)
#         x = self.proj_2(x)
#         return x + shorcut

class DM_Router(nn.Module):
    def __init__(self, channel, d_ffn, patch,domain):
        super().__init__()
        self.patch = patch
        self.channel = channel
        self.norm = nn.LayerNorm(channel)
        self.proj_1 = nn.Linear(channel, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating = SpatialDomainGating(d_ffn, patch * domain)
        self.channel_gating = ChannelDomainGating(patch, domain * channel)
        self.proj_2 = nn.Linear(d_ffn//2, channel)
        self.proj_3 = nn.Linear(channel, channel)
        # self.route = nn.Linear(self.patch  , 1)
        # self.channel_route = nn.Linear(self.feature_dim, domain)

    def forward(self, x):
        # if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
        #     return x
        # B, H, W, C = x.shape
        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = rearrange(x,'b d p c -> b (d p) c')
        x = self.spatial_gating(x)
        x = self.proj_2(x)
        x = rearrange(x, 'b (d p) c -> b d p c',p=self.patch)
        x = x + shorcut
        x = rearrange(x,'b d p c -> b (d c) p',c=self.channel)
        x = self.channel_gating(x)
        x = rearrange(x, 'b (d c) p -> b d p c', c=self.channel)
        x = self.proj_3(x)
        return x + shorcut
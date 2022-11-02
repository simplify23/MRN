import torch
import torch.nn as nn
import einops
import torch
import torch.nn as nn
from einops import rearrange

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

class SpatialGatingUnitv2(nn.Module):
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
        self.seq_len = seq_len
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len * taski)
        self.spatial_gating_unit2 = SpatialGatingUnitv2(seq_len, taski * d_model)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)
        self.proj_3 = nn.Linear(d_model, d_model)
    def forward(self, x):
        # if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
        #     return x
        # B, H, W, C = x.shape
        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = rearrange(x,'b i w c -> b (i w) c')
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = rearrange(x, 'b (i w) c -> b i w c',w=self.seq_len)
        x = x + shorcut
        x = rearrange(x,'b i w c -> b (i c) w',c=self.d_model)
        x = self.spatial_gating_unit2(x)
        x = rearrange(x, 'b (i c) w -> b i w c', c=self.d_model)
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

class Autoencoder(nn.Module):
    def __init__(self, out_dim, patch, taski,hidden_dim=512):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
        nn.Linear(out_dim, 512),
        nn.ReLU())
        self.decoder = nn.Sequential(
        nn.Linear(hidden_dim, out_dim),
        nn.Sigmoid())


    def forward(self, x):
        B, H, W, C = x.shape
        # return x
        # x = rearrange(x, 'b i w c -> b i (w c)')
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        # reconstructed_x = rearrange(reconstructed_x, 'b i (w c) -> b i w c',w=W, c=C)
        return reconstructed_x

class Autoencoderv2(nn.Module):
    def __init__(self, input_dims = 4, code_dims = 128):
        super(Autoencoderv2, self).__init__()
        self.atten = nn.Sequential(
            nn.Linear(4*32, 4 * 32),
            # nn.ReLU(),
            # nn.Linear(512, 4 * 32 * 64),
            nn.Sigmoid()
        )
        # self.cbam = CBAM(64)
        self.encoder = nn.Sequential(
        nn.Conv2d(4, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )
        self.decoder = nn.Sequential(
        nn.Upsample(
                scale_factor=2,
                mode='nearest',
                align_corners=None),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Upsample(
                scale_factor=2,
                mode='nearest',
                align_corners=None),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Upsample(
                scale_factor=2,
                mode='nearest',
                align_corners=None),
        nn.Conv2d(64, 4, 3, 1, 1),
        nn.BatchNorm2d(4),
        # nn.ReLU(),
        )


    def forward(self, x):
        encoded_x = self.encoder(x)
        encoded_x = rearrange(encoded_x, 'b c h w -> b c (h w)')
        atten_x = self.atten(encoded_x)
        encoded_x = rearrange(encoded_x * atten_x, 'b c (h w) -> b c h w', h=4, w=32)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x

class Autoencoderv3(nn.Module):
    def __init__(self, input_dims = 4, code_dims = 128):
        super(Autoencoderv3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
        )
        # self.atten = CBAM(64)
        self.decoder = nn.Sequential(
        nn.Upsample(
                size = (8,64),
                # scale_factor=2,
                mode='nearest',
                align_corners=None),
        nn.Conv2d(256, 128, 3, 1, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Upsample(
                scale_factor=2,
                mode='nearest',
                align_corners=None),
        nn.Conv2d(128, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Upsample(
                scale_factor=2,
                mode='nearest',
                align_corners=None),
        nn.Conv2d(64, 4, 3, 1, 1),
        nn.BatchNorm2d(4),
        # nn.ReLU(),
        )

    def forward(self, x):
        # encoded_x = self.encoder(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.fc(x)
        x = rearrange(x, 'b c (h w) -> b c h w',h=4, w=16)
        # encoded_x = self.atten(encoded_x)
        # encoded_x = atten * encoded_x
        # encoded_x = rearrange(encoded_x, 'b c (h w) -> b c h w', h=4, w=32)
        reconstructed_x = self.decoder(x)
        return reconstructed_x


class Autoencoderv4(nn.Module):
    def __init__(self, input_dims = 4, out_dim = 128):
        super(Autoencoderv4, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, out_dim),
            # nn.ReLU(),
            # nn.Linear(512, 4 * 32 * 64),
            # nn.Sigmoid()
        )
        # self.fc =
        self.encoder = nn.Sequential(
        nn.Conv2d(4, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, (1,3), (1,2), 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )
        self.decoder = nn.Sequential(
        nn.Upsample(
                size=(8,64),
                # scale_factor=(2,4),
                mode='nearest',
                align_corners=None),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Upsample(
                scale_factor=2,
                mode='nearest',
                align_corners=None),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Upsample(
                scale_factor=2,
                mode='nearest',
                align_corners=None),
        nn.Conv2d(64, 4, 3, 1, 1),
        nn.BatchNorm2d(4),
        # nn.ReLU(),
        )


    def forward(self, x):
        encoded_x = self.encoder(x)
        logits = rearrange(encoded_x, 'b c h w -> b (h w) c')
        logits = self.fc(logits)
        # encoded_x = rearrange(logits, 'b c (h w) -> b c h w', h=4, w=32)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x,logits


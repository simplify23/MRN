import numpy as np
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from functools import partial
from einops import rearrange

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMixer(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 HW=[8, 25],
                 local_k=[3, 3],
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(dim, dim, local_k, 1, [local_k[0] // 2, local_k[1] // 2], groups=num_heads)

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose(1, 2).reshape(-1, self.dim, h, w)
        x = self.local_mixer(x)
        x = x.reshape(-1, self.dim, h * w).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=(8, 25),
                 local_k=[7,11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones([H * W, H + hk - 1, W + wk - 1], dtype=torch.float32).cuda()
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_torch = torch.flatten(mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //
                               2], 1)
            mask_inf = torch.full([H * W, H * W], -np.inf, dtype=torch.float32).cuda()
            mask = torch.where(mask_torch < 1, mask_torch, mask_inf)
            self.mask = mask.unsqueeze(0)
            self.mask = self.mask.unsqueeze(0)
            # print(self.mask.size())

        self.mixer = mixer

    def forward(self, x):
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape((-1, N, 3, self.num_heads, C // self.num_heads))
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q.matmul(k.permute(0, 1, 3, 2)))
        if self.mixer == 'Local':
            attn += self.mask.to(attn.device)
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute(0, 2, 1, 3).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=[8, 25],
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6):
        super().__init__()

        self.norm1 = norm_layer(dim)

        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        if mixer == 'Conv':
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        self.N, self.C = x.shape[1:]
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self):
        flops = self.attn.flops() + self.N * self.C * 2 + 2 * self.N * self.C * self.C * self.mlp_ratio
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=[32, 100], in_channels=3, embed_dim=768, sub_num=2):
        super().__init__()
        # img_size = to_2tuple(img_size)
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        # num_patches = (img_size[1] ) * \
        #               (img_size[0] )
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if sub_num == 2:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, 3, 2, 1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU())
        if sub_num == 3:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 4, 3, 2, 1),
                nn.BatchNorm2d(embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
                nn.BatchNorm2d(embed_dim)
            )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x1 = self.proj1(x)
        # x2 = self.proj2(x1).flatten(2).transpose(1, 2)
        # x1 = x1.flatten(2).transpose(1, 2)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
        # return x1,x2

    def flops(self):
        Ho, Wo = self.img_size
        flops = Ho // 2 * Wo // 2 * 3 * self.embed_dim // 2 * (3 * 3) \
                + Ho // 4 * Wo // 4 * self.embed_dim // 2 * self.embed_dim * (3 * 3) \
                + Ho * Wo * self.embed_dim * 2
        return flops


class SubSample(nn.Module):
    def __init__(self, in_channels, out_channels, types='Pool', stride=[2, 1], sub_norm='nn.LayerNorm', act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        elif types == 'Unet':
            # For mini-Unet
            attn_mode = 'nearest'
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None


    def forward(self, x):

        if self.types == 'Pool':
            # x = x.transpose((0, 2, 1))
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).transpose(1, 2))
        elif self.types == 'Unet':
        # Apply mini U-Net on k
            features = []
            for i in range(len(self.encoder)):
                x = self.encoder[i](x)
                features.append(x)
            x = self.block(x)
            for i in range(len(self.decoder) - 1):
                x = self.k_decoder[i](x)
                x = x + features[len(self.decoder) - 2 - i]
            out = self.decoder[-1](x)
        else:
            # self.H, self.W = x.shape[2:]
            x = self.conv(x)
            out = x.flatten(2).transpose(1, 2)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


class SVTR2(nn.Module):

    def __init__(self,
                 img_size=[32, 256],
                 in_channels=3,
                 embed_dim=[64, 128, 256],
                 depth=[3, 6, 3],
                 num_heads=[2, 4, 8],
                 mixer=['Local', 'Local', 'Local', 'Local', 'Local', 'Local', 'Global', 'Global', 'Global', 'Global',
                        'Global', 'Global'],  # Local atten, Global atten, Conv
                 local_mixer=[[7, 11], [7, 11], [7, 11]],
                 patch_merging='Conv',  # Conv, Pool, None
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 last_drop=0.1,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer='nn.LayerNorm',
                 sub_norm='nn.LayerNorm',
                 epsilon=1e-6,
                 out_channels=192,
                 out_char_num=25,
                 block_unit='Block',
                 act='nn.GELU',
                 last_stage=True,
                 sub_num=2,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim
        self.out_channels = out_channels
        norm_layer = partial(eval(norm_layer), eps=epsilon)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        # self.HW = [img_size[0], img_size[1]]
        self.HW = [img_size[0] // (2 ** sub_num), img_size[1] // (2 ** sub_num)]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        # self.add_parameter("pos_embed", self.pos_embed)
        # self.cls_token = self.create_parameter(
        #     shape=(1, 1, embed_dim), default_initializer=zeros_)
        # self.add_parameter("cls_token", self.cls_token)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # self.up_linear = nn.Linear(256, 512)
        Block_unit = eval(block_unit)
        if block_unit == 'CSWinBlock':
            split_size_h = [1, 2, 2]
            split_size_w = [5, 5, 25]
            ex_arg = [{'reso': [img_size[0] // 4, img_size[1] // 4],
                       'split_size_h': split_size_h[0],
                       'split_size_w': split_size_w[0]},
                      {'reso': [img_size[0] // 8, img_size[1] // 4],
                       'split_size_h': split_size_h[1],
                       'split_size_w': split_size_w[1]},
                      {'reso': [img_size[0] // 16, img_size[1] // 4],
                       'split_size_h': split_size_h[2],
                       'split_size_w': split_size_w[2]}
                      ]
        else:
            ex_arg = [{'epsilon': epsilon},
                      {'epsilon': epsilon},
                      {'epsilon': epsilon}]
        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mixer=mixer[0:depth[0]][i],
                HW=self.HW,
                local_mixer=local_mixer[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                norm_layer=norm_layer,
                **ex_arg[0],
            ) for i in range(depth[0])
        ])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(embed_dim[0], embed_dim[1], sub_norm=sub_norm, stride=[2, 1],
                                         types=patch_merging)  # ConvBNLayer(embed_dim[0], embed_dim[1], kernel_size=3, stride=[2, 1], sub_norm=sub_norm)
            HW = [self.HW[0] // 2, self.HW[1]]
            # self.sub_sample1_0 = SubSample(embed_dim[0], embed_dim[0], sub_norm=sub_norm, stride=[2, 2],
            #                              types=patch_merging)  # ConvBNLayer(embed_dim[0], embed_dim[1], kernel_size=3, stride=[2, 1], sub_norm=sub_norm)
            # HW = [self.HW[0] // 2, self.HW[1] // 2]
            # self.sub_sample1_1 = SubSample(embed_dim[0], embed_dim[1], sub_norm=sub_norm, stride=[1, 1],
            #                              types=patch_merging)  # ConvBNLayer(embed_dim[0], embed_dim[1], kernel_size=3, stride=[2, 1], sub_norm=sub_norm)

        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mixer=mixer[depth[0]:depth[0] + depth[1]][i],
                HW=HW,
                local_mixer=local_mixer[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                norm_layer=norm_layer,
                **ex_arg[1]) for i in range(depth[1])
        ])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(embed_dim[1], embed_dim[2], sub_norm=sub_norm, stride=[2, 1],
                                         types=patch_merging)  # ConvBNLayer(embed_dim[1], embed_dim[2], kernel_size=3, stride=[2, 1], sub_norm=sub_norm)
            HW = [self.HW[0] // 4, self.HW[1]]
            # self.sub_sample2 = SubSample(embed_dim[1], embed_dim[2], sub_norm=sub_norm, stride=[2, 2],
            #                              types=patch_merging)  # ConvBNLayer(embed_dim[1], embed_dim[2], kernel_size=3, stride=[2, 1], sub_norm=sub_norm)
            # HW = [self.HW[0] // 4, self.HW[1] // 4]
        else:
            HW = self.HW
        self.blocks3 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mixer=mixer[depth[0] + depth[1]:][i],
                HW=HW,
                local_mixer=local_mixer[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                norm_layer=norm_layer,
                **ex_arg[2]) for i in range(depth[2])
        ])
        if patch_merging is not None:
            # self.sub_sample1 = SubSample(embed_dim[0], embed_dim[1], sub_norm=sub_norm, stride=[2, 1],
            #                              types=patch_merging)  # ConvBNLayer(embed_dim[0], embed_dim[1], kernel_size=3, stride=[2, 1], sub_norm=sub_norm)
            # HW = [self.HW[0] // 2, self.HW[1]]
            self.sub_sample3 = SubSample(embed_dim[2], out_channels, sub_norm=sub_norm, stride=[2, 1],
                                         types=patch_merging)  # ConvBNLayer(embed_dim[0], embed_dim[1], kernel_size=3, stride=[2, 1], sub_norm=sub_norm)

        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            self.linear = nn.Linear(384, 512)

            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=False)

            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop)
        self.norm = norm_layer(embed_dim[-1])

        # Classifier head
        # self.head = nn.Linear(embed_dim,
        #                       class_num) if class_num > 0 else Identity()

        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.bias, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward_features(self, x, tpsnet=None):
        # B = x.shape[0]
        # out = []

        B = x.shape[0]
        x = self.patch_embed(x)
        # out.append(rearrange(x1, 'b (h w) c -> b c h w',h=32,w=128))
        # out.append(rearrange(x, 'b (h w) c -> b c h w',h=32,w=128))
        # cls_tokens = self.cls_token.expand((B, -1, -1))
        # x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(x.transpose(1, 2).reshape(B, self.embed_dim[0], self.HW[0], self.HW[1]))

        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(x.transpose(1, 2).reshape(B, self.embed_dim[1], self.HW[0] // 2, self.HW[1]))
            # x = self.sub_sample2(x.transpose(1, 2).reshape(B, self.embed_dim[1], self.HW[0] // 2, self.HW[1] // 2))
        for blk in self.blocks3:
            x = blk(x)
        if self.patch_merging is not None:
            # x = self.sub_sample2(x.transpose(1, 2).reshape(B, self.embed_dim[1], self.HW[0] // 2, self.HW[1]))
            x = self.sub_sample3(x.transpose(1, 2).reshape(B, self.embed_dim[2], self.HW[0] // 4, self.HW[1]))
        x = x.permute(0, 2, 1).reshape([-1, self.out_channels, self.HW[0] // 8, self.HW[1]])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x
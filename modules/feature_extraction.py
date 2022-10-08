import torch.nn as nn
import torch.nn.functional as F
from collections import Callable
import numpy as np
import torch
from timm.models.layers import trunc_normal_
from functools import partial
# from mmcv.runner import BaseModule
# from mmcv.cnn import ConvModule
from einops import rearrange

class VGG_FeatureExtractor(nn.Module):
    """FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(
                self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),  # 512x4x25
            nn.Conv2d(
                self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),
        )  # 512x1x24

    def forward(self, input):
        return self.ConvNet(input)


class RCNN_FeatureExtractor(nn.Module):
    """FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super(RCNN_FeatureExtractor, self).__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64 x 16 x 50
            GRCL(
                self.output_channel[0],
                self.output_channel[0],
                num_iteration=5,
                kernel_size=3,
                pad=1,
            ),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 25
            GRCL(
                self.output_channel[0],
                self.output_channel[1],
                num_iteration=5,
                kernel_size=3,
                pad=1,
            ),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 128 x 4 x 26
            GRCL(
                self.output_channel[1],
                self.output_channel[2],
                num_iteration=5,
                kernel_size=3,
                pad=1,
            ),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 256 x 2 x 27
            nn.Conv2d(
                self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
        )  # 512 x 1 x 26

    def forward(self, input):
        return self.ConvNet(input)


class ResNet_FeatureExtractor(nn.Module):
    """FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)


# For Gated RCNN
class GRCL(nn.Module):
    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()
        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(
            input_channel, output_channel, kernel_size, 1, pad, bias=False
        )
        self.wr_x = nn.Conv2d(
            output_channel, output_channel, kernel_size, 1, pad, bias=False
        )

        self.BN_x_init = nn.BatchNorm2d(output_channel)

        self.num_iteration = num_iteration
        self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, input):
        """The input of GRCL is consistant over time t, which is denoted by u(0)
        thus wgf_u / wf_u is also consistant over time t.
        """
        wgf_u = self.wgf_u(input)
        wf_u = self.wf_u(input)
        x = F.relu(self.BN_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

        return x


class GRCL_unit(nn.Module):
    def __init__(self, output_channel):
        super(GRCL_unit, self).__init__()
        self.BN_gfu = nn.BatchNorm2d(output_channel)
        self.BN_grx = nn.BatchNorm2d(output_channel)
        self.BN_fu = nn.BatchNorm2d(output_channel)
        self.BN_rx = nn.BatchNorm2d(output_channel)
        self.BN_Gx = nn.BatchNorm2d(output_channel)

    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        G_first_term = self.BN_gfu(wgf_u)
        G_second_term = self.BN_grx(wgr_x)
        G = F.sigmoid(G_first_term + G_second_term)

        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
        x = F.relu(x_first_term + x_second_term)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
            output_channel,
        ]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(
            input_channel,
            int(output_channel / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(
            int(output_channel / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(
            self.output_channel_block[0],
            self.output_channel_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(
            block, self.output_channel_block[1], layers[1], stride=1
        )
        self.conv2 = nn.Conv2d(
            self.output_channel_block[1],
            self.output_channel_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(
            block, self.output_channel_block[2], layers[2], stride=1
        )
        self.conv3 = nn.Conv2d(
            self.output_channel_block[2],
            self.output_channel_block[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(
            block, self.output_channel_block[3], layers[3], stride=1
        )
        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1),
            bias=False,
        )
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x
import numpy as np
import math


def truncated_normal_(tensor,mean=0,std=0.02):
    with torch.no_grad():
        size = tensor.size()
        tmp = tensor.new_empty(size+(4,)).normal_().cuda()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind.cuda()).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x

    keep_prob = torch.tensor(1-drop_prob).cuda()
    shape = (x.size()[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype).cuda()
    random_tensor = torch.floor(random_tensor)
    output = torch.div(x, keep_prob) * random_tensor
    return output


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.GELU):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias_attr
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
                 HW=(8, 25),
                 local_k=(3, 3)):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            local_k,
            1,
            (local_k[0] // 2, local_k[1] // 2),
            groups=num_heads
        )

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.permute(0, 2, 1).reshape([-1, self.dim, h, w])
        x = self.local_mixer(x)
        x = torch.flatten(x, 2).permute(0, 2, 1)
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
            attn += self.mask
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
                 drop=0,
                 attn_drop=0,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        else:
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
                proj_drop=drop
            )
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer
            )
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    def __init__(self,
                 img_size=[32, 100],
                 in_channels=3,
                 embed_dim=768,
                 sub_num=2):
        super().__init__()
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if sub_num == 2:
            self.proj = nn.Sequential(
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=embed_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False),
                ConvBNLayer(
                    in_channels=embed_dim // 2,
                    out_channels=embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False)
            )
        if sub_num == 3:
            self.proj = nn.Sequential(
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=embed_dim // 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False),
                ConvBNLayer(
                    in_channels=embed_dim // 4,
                    out_channels=embed_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False),
                ConvBNLayer(
                    in_channels=embed_dim // 2,
                    out_channels=embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False))

    def forward(self, x):
        B, C, H, W = x.size()
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).permute(0, 2, 1)
        return x


class SubSample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 types='Pool',
                 stride=(2, 1),
                 sub_norm='nn.LayerNorm',
                 act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(
                kernel_size=(3, 5), stride=stride, padding=(1, 2))
            self.maxpool = nn.MaxPool2d(
                kernel_size=(3, 5), stride=stride, padding=(1, 2))
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1
            )
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))
        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


class SVTRNet(nn.Module):
    def __init__(
            self,
            img_size=[32, 256],
            in_channels=3,
            embed_dim=[64, 128, 256],
            depth=[3, 6, 3],
            num_heads=[2, 4, 8],
            mixer=['Local'] * 6 + ['Global'] *
            6,  # Local atten, Global atten, Conv
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
            out_char_num=100,
            block_unit='Block',
            act='nn.GELU',
            last_stage=False,
            sub_num=2,
            prenorm=True,
            use_lenhead=False,
            **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]
        # self.pos_embed = self.create_parameter(
        #     shape=[1, num_patches, embed_dim[0]], default_initializer=zeros_)
        # self.add_parameter("pos_embed", self.pos_embed)
        self.pos_embed = nn.Parameter(torch.zeros([1, num_patches, embed_dim[0]], dtype=torch.float32), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)

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
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[0])
        ])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 2, self.HW[1]]
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
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[1])
        ])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 4, self.HW[1]]
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
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[2])
        ])
        if patch_merging is not None:
            self.sub_sample3 = SubSample(
                embed_dim[2],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 8, self.HW[1]]
        else:
            HW = self.HW
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, out_char_num))
            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop)
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], epsilon=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(p=last_drop)

        truncated_normal_(self.pos_embed)
        self.apply(self._init_weights)
        print('---------model weight inits-----------')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            truncated_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(
                x.permute(0, 2, 1).reshape(
                    [-1, self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(
                x.permute(0, 2, 1).reshape(
                    [-1, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks3:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample3(
                x.permute(0, 2, 1).reshape(
                    [-1, self.embed_dim[2], self.HW[0] // 4, self.HW[1]]))
        x = x.permute(0, 2, 1).reshape([-1, self.embed_dim[2], self.HW[0] // 8, self.HW[1]])
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = self.avg_pool(
                x.permute(0, 2, 1).reshape(
                    [-1, self.embed_dim[2], h, self.HW[1]]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        if self.use_lenhead:
            return x, len_x
        return x

class SVTR_FeatureExtractor(nn.Module):
    """FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super(SVTR_FeatureExtractor, self).__init__()
        self.ConvNet = SVTRNet(in_channels=input_channel, out_channels=output_channel)
        # self.neck = Im2Seq()

    def forward(self, input):
        return self.ConvNet(input)


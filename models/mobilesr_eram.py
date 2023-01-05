import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange
from torchinfo import summary
from torch.nn import init 

from DepthwiseSeparableConvolution import depthwise_separable_conv

class ERAM(nn.Module):
    def __init__(self, channel_begin, dimension):
        super().__init__()
        self.conv = nn.Conv2d(channel_begin, channel_begin, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(dimension)
        
        self.conv1 = nn.Conv2d(channel_begin, channel_begin//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_begin//2, channel_begin, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channel_begin, channel_begin, kernel_size=3, stride=1, padding=1)

        self.dconv = depthwise_separable_conv(channel_begin, channel_begin, kernel_size = 3, padding = 1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        si_ca = self.avgpool(x) + torch.var_mean(x, dim=(2,3))[0].unsqueeze(2).unsqueeze(2)
        mi_ca = self.conv2(self.relu(self.conv1(si_ca)))

        mi_sa = self.conv3(self.relu(self.dconv(x)))

        return self.sigmoid(mi_ca+mi_sa) * x



class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 2 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        # print("shape of x :",x.shape)
        x1 = self.lrelu(self.conv1(x))
        # print(f'RRDB SHAPE x1: {x1.shape}')
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # print(f'RRDB SHAPE x2: {x2.shape}')
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # print(f'RRDB SHAPE x3 : {x3.shape}')
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2), 1)))
        # print(f'RRDB SHAPE x4 : {x4.shape}')
        x5 = self.conv5(torch.cat((x, x1, x2), 1))
        # print(f'RRDB SHAPE x5 : {x5.shape}')
        return x5 * 0.2 + x

# class RRDBAttention(nn.Module):
#     '''Residual in Residual Dense Block with Efficient Attention'''

#     def __init__(self, nf, gc=32):
#         super(RRDBAttention, self).__init__()
#         self.RDB1 = ResidualDenseBlock_5C(nf, gc)
#         self.eat1 = EfficientAttention(nf, 64, 4, 64)
#         self.RDB2 = ResidualDenseBlock_5C(nf, gc)
#         self.eat2 = EfficientAttention(nf, 64, 4, 64)
#         self.RDB3 = ResidualDenseBlock_5C(nf, gc)
#         self.eat3 = EfficientAttention(nf, 64, 4, 64)

#     def forward(self, x):
#         out = self.eat1( self.RDB1(x) )
#         out = self.eat2( self.RDB2(out) )
#         out = self.eat3( self.RDB3(out) )
#         return out * 0.2 + x
    

class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(SelfAttn, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, N, c = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)
        # [b, N, c] -> [b, N, head, c//head] -> [b, head, N, c//head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # [b, head, N, c//head] * [b, head, N, c//head] -> [b, head, N, N]
        attn = torch.einsum('bijc, bikc -> bijk', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        # [b, head, N, N] * [b, head, N, c//head] -> [b, head, N, c//head] -> [b, N, head, c//head]
        x = torch.einsum('bijk, bikc -> bijc', attn, v)
        x = rearrange(x, 'b i j c -> b j (i c)')
        x = self.proj_out(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4):
        super(Mlp, self).__init__()
        hidden_features = in_features * mlp_ratio

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, x):
        return self.fc(x)


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size, window_size, c) [non-overlap]
    """
    return rearrange(x, 'b (h s1) (w s2) c -> (b h w) s1 s2 c', s1=window_size, s2=window_size)


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    return rearrange(windows, '(b h w) s1 s2 c -> b (h s1) (w s2) c', b=b, h=h // window_size, w=w // window_size)


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=4, qkv_bias=False):
        super(Transformer, self).__init__()
        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttn(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = Mlp(dim, mlp_ratio)
        # self.rrdBAtt = RRDBAttention(256,32)


    def forward(self, x):
        x = x + self.pos_embed(x)
        x = rearrange(x, 'b c h w -> b h w c')
        b, h, w, c = x.shape

        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, c
        x_windows = rearrange(x_windows, 'B s1 s2 c -> B (s1 s2) c', s1=self.window_size,
                              s2=self.window_size)  # nW*b, window_size*window_size, c

        # W-MSA/SW-MSA
        # print("shape before : ",x_windows.shape)
        attn_windows = self.attn(x_windows)  # nW*b, window_size*window_size, c

        # x_windows = x_windows.view(b,256,64,-1)

        # attn_windows = self.rrdBAtt(x_windows)

        attn_windows = attn_windows.reshape(b*256,64,-1)

        # print("shape after : ",x_windows.shape)
        # merge windows
        attn_windows = rearrange(attn_windows, 'B (s1 s2) c -> B s1 s2 c', s1=self.window_size, s2=self.window_size)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # b H' W' c

        # reverse cyclic shift
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return rearrange(x, 'b h w c -> b c h w')


class ResBlock(nn.Module):
    def __init__(self, in_features, ratio=4):
        super(ResBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_features, in_features * ratio, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_features * ratio, in_features * ratio, 3, 1, 1, groups=in_features * ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_features * ratio, in_features, 1, 1, 0),
        )

    def forward(self, x):
        print(x.shape)
        return self.net(x) + x


class BaseBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, ratios=[1, 2, 2, 4, 4], qkv_bias=False):
        super(BaseBlock, self).__init__()
        self.layers = nn.ModuleList([])
        # self.eram = ERAM(dim,128)
        print('dim : ',dim)
        for ratio in ratios:
            self.layers.append(nn.ModuleList([
                Transformer(dim, num_heads, window_size, ratio, qkv_bias),
                ResBlock(dim, ratio),
                ERAM(dim,128)
            ]))

    def forward(self, x):
        for tblock, rblock ,eram in self.layers:
            x = tblock(x)
            x = rblock(x)
            x = eram(x)
        return x


class MobileSR(nn.Module):
    def __init__(self, n_feats=40, n_heads=8, ratios=[4, 2, 2, 2, 4], upscaling_factor=2):
        super(MobileSR, self).__init__()
        self.scale = upscaling_factor
        self.head = nn.Conv2d(1, n_feats, 3, 1, 1)

        self.body = BaseBlock(n_feats, num_heads=n_heads, ratios=ratios)

        self.fuse = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)

        if self.scale == 4:
            self.upsapling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2)
            )
        else:
            self.upsapling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * self.scale * self.scale, 1, 1, 0),
                nn.PixelShuffle(self.scale)
            )

        self.tail = nn.Conv2d(n_feats, 1, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x0 = self.head(x)
        x0 = self.fuse(torch.cat([x0, self.body(x0)], dim=1))
        x0 = self.upsapling(x0)
        x0 = self.tail(self.act(x0))
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return (torch.tanh(x0 + x) +1.0)/2.0

model = MobileSR()
# model = RRDBAttention(1024,32)
print(summary(model,input_size=[1,1,128,128]))
# inp = torch.rand([1,1,128,128])
# out = model(inp)
# print(out.shape)
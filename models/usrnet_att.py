# Model Definition for USRNet 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

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
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDBAttention(nn.Module):
    '''Residual in Residual Dense Block with Efficient Attention'''

    def __init__(self, nf, gc=32):
        super(RRDBAttention, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.eat1 = EfficientAttention(nf, 64, 4, 64)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.eat2 = EfficientAttention(nf, 64, 4, 64)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        self.eat3 = EfficientAttention(nf, 64, 4, 64)

    def forward(self, x):
        out = self.eat1( self.RDB1(x) )
        out = self.eat2( self.RDB2(out) )
        out = self.eat3( self.RDB3(out) )
        return out * 0.2 + x

class EncoderWithAttention(nn.Module):
    def __init__(self, nf, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.upscaler = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')
        self.lrelu = nn.LeakyReLU()

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv_1 = nn.Conv2d(1, nf, kernel_size=3, stride=1, padding=1) 
        self.rrdb_1 = RRDBAttention(nf, 32)
        self.conv_res = nn.Conv2d(1, nf, kernel_size=3, stride=1, padding=1) 
        self.conv_2 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=1, padding=1) 
        self.rrdb_2 = RRDBAttention(nf*2, 32)
        self.conv_3 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=1, padding=1) 
        self.rrdb_3 = RRDBAttention(nf*4, 32)
        self.conv_4 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=1, padding=1)
        self.rrdb_4 = RRDBAttention(nf*8, 32)

    def forward(self, x):
        feats = []

        upscaled = self.upscaler(x) # 1 x 256 x 256
        res_in = self.conv_res(x)
        x = self.lrelu(self.conv_1(upscaled))
        x = self.rrdb_1(x)
        feats.append(x) # 64 x 256 x 256
        pooled = self.maxpool(x)
        x = self.lrelu(self.conv_2(pooled + res_in)) # residual connection from lr input
        x = self.rrdb_2(x)
        feats.append(x) # 128 x 64 x 64
        pooled = self.maxpool(x)
        x = self.lrelu(self.conv_3(pooled))
        x = self.rrdb_3(x)
        feats.append(x) # 256 x 64 x 64 
        pooled = self.maxpool(x)
        x = self.lrelu(self.conv_4(pooled))
        x = self.rrdb_4(x)
        feats.append(x) # 512 x 32 x 32

        return feats


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

class Decoder(nn.Module):
    def __init__(self, nf, dimension):
        super().__init__()

        self.lrelu = nn.LeakyReLU()

        self.eram_1 = ERAM(nf, dimension)
        self.conv_1 = nn.Conv2d(nf, nf//2, kernel_size=3, stride=1, padding=1)
        self.up_1 = nn.ConvTranspose2d(nf//2, nf//2, kernel_size=2, stride=2)
        
        self.rrdb_1 = RRDBAttention(nf, 32)

        self.eram_2 = ERAM(nf, dimension*2)
        self.conv_2 = nn.Conv2d(nf, nf//4, kernel_size=3, stride=1, padding=1)
        self.up_2 = nn.ConvTranspose2d(nf//4, nf//4, kernel_size=2, stride=2)

        self.rrdb_2 = RRDBAttention(nf//2, 32)

        self.eram_3 = ERAM(nf//2, dimension*4)
        self.conv_3 = nn.Conv2d(nf//2, nf//8, kernel_size=3, stride=1, padding=1)
        self.up_3 = nn.ConvTranspose2d(nf//8, nf//8, kernel_size=2, stride=2)

        self.rrdb_3 = RRDBAttention(nf//4, 32)

        self.eram_4 = ERAM(nf//4, dimension*8)
        self.conv_4 = nn.Conv2d(nf//4, 1, kernel_size=3, stride=1, padding=1)

    
    def forward(self, feats):
        x = self.lrelu(self.up_1(self.conv_1(self.eram_1(feats[-1]))))
        x = torch.cat((x, feats[-2]), dim=1)
        x = self.rrdb_1(x)
        x = self.lrelu(self.up_2(self.conv_2(self.eram_2(x)))) 
        x = torch.cat((x, feats[-3]), dim=1)
        x = self.rrdb_2(x)
        x = self.lrelu(self.up_3(self.conv_3(self.eram_3(x))))
        x = torch.cat((x, feats[0]), dim=1)
        x = self.rrdb_3(x)
        x = self.conv_4(self.eram_4(x))
        return x

class USRNet(nn.Module):
  def __init__(self, scale_factor, enc_filters, dec_filters, dec_dim):
    super().__init__()
    self.encoder = EncoderWithAttention(enc_filters, scale_factor)
    self.decoder = Decoder(dec_filters, dec_dim)
    self.relu = nn.ReLU()
  def forward(self, lr):
    out_feats = self.encoder(lr)
    sr = self.decoder(out_feats)
    return sr

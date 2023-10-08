import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(EncoderLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=2, padding=0)

    def forward(self, x):
        shortcut = self.layer(x)
        out = self.maxpool(shortcut)
        return out, shortcut


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias):
        super(DecoderLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        out = torch.cat((x, y), dim=1)
        out = self.layer(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias):
        super(BottleNeck, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.config = config
        bias = config['conv_bias']
        self.en_layer_1 = EncoderLayer(config['num_mode_input'], 30, bias)
        self.en_layer_2 = EncoderLayer(30, 60, bias)
        self.en_layer_3 = EncoderLayer(60, 120, bias)
        self.en_layer_4 = EncoderLayer(120, 240, bias)
        self.bottle_neck = BottleNeck(240, 480, 240, bias)
        self.de_layer_4 = DecoderLayer(480, 240, 120, bias)
        self.de_layer_3 = DecoderLayer(240, 120, 60, bias)
        self.de_layer_2 = DecoderLayer(120, 60, 30, bias)
        self.de_layer_1 = DecoderLayer(60, 30, 30, bias)
        self.end_layer = nn.Conv3d(30, 3, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x, branch=None):
        x_input = [x[i] for i in self.config['mode_input']]
        x_input = torch.cat(x_input, dim=1)
        en_1, sc_1 = self.en_layer_1(x_input)
        en_2, sc_2 = self.en_layer_2(en_1)
        en_3, sc_3 = self.en_layer_3(en_2)
        en_4, sc_4 = self.en_layer_4(en_3)
        bneck = self.bottle_neck(en_4)
        de_4 = self.de_layer_4(bneck, sc_4)
        de_3 = self.de_layer_3(de_4, sc_3)
        de_2 = self.de_layer_2(de_3, sc_2)
        de_1 = self.de_layer_1(de_2, sc_1)
        out = self.end_layer(de_1)
        if self.config['end_sigmoid']:
            out = torch.sigmoid(out)
        return [out]
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.super_layers import * 
     
class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias, img_dim = "3D" ):
        super(EncoderLayer, self).__init__()
        self.layer = nn.Sequential(
            SuperConv(in_channels, out_channels, img_dim =img_dim,kernel_size=3, stride=1, padding=1, bias=bias),         
            SuperInstanceNorm(out_channels,img_dim =img_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            SuperConv(out_channels, out_channels,img_dim =img_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            SuperInstanceNorm(out_channels,img_dim =img_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        
        self.maxpool = SuperMaxPool(kernel_size=2, padding=0,img_dim=img_dim)

    def forward(self, x):
        shortcut = self.layer(x)
        out = self.maxpool(shortcut)
        return out, shortcut


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias,img_dim = "3D"):
        super(DecoderLayer, self).__init__()
        self.img_dim = img_dim
        self.layer = nn.Sequential(
            SuperConv(in_channels, mid_channels, img_dim =img_dim,kernel_size=3, stride=1, padding=1, bias=bias),
            SuperInstanceNorm(mid_channels,img_dim =img_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            SuperConv(mid_channels, out_channels, img_dim =img_dim,kernel_size=3, stride=1, padding=1, bias=bias),
            SuperInstanceNorm(out_channels,img_dim =img_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x, y=None):

        if self.img_dim == "3D":
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        else: 
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)     
        if y is not None:
            out = torch.cat((x, y), dim=1)
        else:
            out = x
        out = self.layer(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, config):
        super(BottleNeck, self).__init__()
        bias = config['conv_bias']
        img_dim = config["img_dim"]
        self.layer = nn.Sequential(
            SuperConv(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=bias,img_dim =img_dim),
            SuperInstanceNorm(mid_channels,img_dim =img_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            SuperConv(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias,img_dim =img_dim),
            SuperInstanceNorm(out_channels,img_dim =img_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Encoder(nn.Module):
    def __init__(self, config, in_channels=1):
        super(Encoder, self).__init__()
        bias = config['conv_bias']
        img_dim = config["img_dim"]
        self.en_layer_1 = EncoderLayer(in_channels, 30, bias,img_dim = img_dim)
        self.en_layer_2 = EncoderLayer(30, 60, bias,img_dim = img_dim)
        self.en_layer_3 = EncoderLayer(60, 120, bias,img_dim = img_dim)
        #self.en_layer_4 = EncoderLayer(120, 240, bias,img_dim = img_dim)

    def forward(self, x):
        en_1, sc_1 = self.en_layer_1(x)
        en_2, sc_2 = self.en_layer_2(en_1)
        en_3, sc_3 = self.en_layer_3(en_2)
        #en_4, sc_4 = self.en_layer_4(en_3)
        return [en_1, en_2, en_3], [sc_1, sc_2, sc_3]

class Decoder(nn.Module):
    def __init__(self, config, out_channels=3):
        super(Decoder, self).__init__()
        self.config = config
        bias = self.config['conv_bias']
        img_dim = config["img_dim"]

        #self.de_layer_4 = DecoderLayer(480, 240, 120, bias,img_dim = img_dim) if config['skip_connection'] >= 1 else DecoderLayer(240, 240, 120, bias,img_dim = img_dim)
        self.de_layer_3 = DecoderLayer(240, 120, 60, bias,img_dim = img_dim) if config['skip_connection'] >= 2 else DecoderLayer(120, 120, 60, bias,img_dim = img_dim)
        self.de_layer_2 = DecoderLayer(120, 60, 30, bias,img_dim = img_dim) if config['skip_connection'] >= 3 else DecoderLayer(60, 60, 30, bias,img_dim = img_dim)
        self.de_layer_1 = DecoderLayer(60, 30, 30, bias,img_dim = img_dim) if config['skip_connection'] >= 4 else DecoderLayer(30, 30, 30, bias,img_dim = img_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if self.config['softmax']:
            self.end_layer = SuperConv(30, 4, kernel_size=1, stride=1, padding=0, bias=bias,img_dim = img_dim)
        else:
            self.end_layer = SuperConv(30, out_channels, kernel_size=1, stride=1, padding=0, bias=bias,img_dim = img_dim)

    def forward(self, bneck, sc, temperature=False):
        sc_1, sc_2, sc_3 = sc

        #de_4 = self.de_layer_4(bneck, sc_4) if self.config['skip_connection'] >= 1 else self.de_layer_4(bneck)
        de_3 = self.de_layer_3(bneck, sc_3) if self.config['skip_connection'] >= 2 else self.de_layer_3(de_4)
        de_2 = self.de_layer_2(de_3, sc_2) if self.config['skip_connection'] >= 3 else self.de_layer_2(de_3)
        de_1 = self.de_layer_1(de_2, sc_1) if self.config['skip_connection'] >= 4 else self.de_layer_1(de_2)

        logit = self.end_layer(de_1)

        if self.config['end_layer'] == 'sigmoid':
            return torch.sigmoid(out)
        elif self.config['end_layer'] == 'leaky_relu':
            return self.leaky_relu(out)
        elif self.config['end_layer'] == 'softmax':
            if temperature:
                return torch.softmax(logit, dim=1), torch.softmax(logit/self.config['temperature'], dim=1), de_1
            else:
                return torch.softmax(logit, dim=1)
        else:
            return out


class HUNetv4(nn.Module):
    def __init__(self, config):
        super(HUNetv4, self).__init__()
        self.config = config
        self.encoder_target = Encoder(config, in_channels=config['num_target_modality'])
        self.encoder_source = Encoder(config, in_channels=config['num_source_modality'])
        self.bottle_neck_target = BottleNeck(120, 240, 120, config)
        self.bottle_neck_source = BottleNeck(120, 240, 120, config)
        self.decoder_target = Decoder(config)
        self.decoder_source = Decoder(config)

    def forward(self, x, branch, return_feature="bneck", temperature=False):
        if branch == 'both':
            x_target = torch.cat([x[i] for i in self.config['target_modality']], dim=1)
            x_source = torch.cat([x[i] for i in self.config['source_modality']], dim=1)
            en_target, sc_target = self.encoder_target(x_target)
            en_source, sc_source = self.encoder_source(x_source)
            bneck_target = self.bottle_neck_target(en_target[-1])
            bneck_source = self.bottle_neck_source(en_source[-1])
            out_target, soft_target, de_target = self.decoder_target(bneck_target, sc_target, temperature)
            out_source, soft_source, de_source = self.decoder_source(bneck_source, sc_source, temperature)
            return [out_source, out_target], [soft_source, soft_target], [de_source, de_target], [bneck_source, bneck_target]

        elif branch == 'source':
            x_source = torch.cat([x[i] for i in self.config['source_modality']], dim=1)
            en_source, sc_source = self.encoder_source(x_source)
            bneck_source = self.bottle_neck_source(en_source[-1])
            out_source = self.decoder_source(bneck_source, sc_source)
            return [out_source]

        elif branch == 'target':
            x_target = torch.cat([x[i] for i in self.config['target_modality']], dim=1)
            en_target, sc_target = self.encoder_target(x_target)
            bneck_target = self.bottle_neck_target(en_target[-1])
            out_target = self.decoder_target(bneck_target, sc_target)
            return [out_target]
        else:
            assert "wrong mode for model forward."

class UNet(nn.Module):
    def __init__(self, config, in_channels=None, out_channels=3):
        super(UNet, self).__init__()
        self.config = config
        if in_channels:
            self.encoder_source = Encoder(config, in_channels)
        else:
            self.encoder_source = Encoder(config, in_channels=config['num_source_modality'])
        self.bottle_neck_source = BottleNeck(120, 240, 120, config)
        self.decoder = Decoder(config, out_channels)

    def forward(self, x, branch=None, temperature=False):
        x_source = torch.cat([x[i] for i in self.config['source_modality']], dim=1)
        en_source, sc_source = self.encoder_source(x_source)
        bneck_source = self.bottle_neck_source(en_source[-1])
        out_source = self.decoder(bneck_source, sc_source, temperature)
        return [out_source]


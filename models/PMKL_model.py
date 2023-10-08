import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.super_layers import *

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias, img_dim="3D", stride=1):
        super(EncoderLayer, self).__init__()
        self.layer1 = nn.Sequential(
            SuperConv(in_channels, out_channels, img_dim=img_dim, kernel_size=3, stride=stride, padding=1, bias=bias),
            SuperInstanceNorm(out_channels, img_dim=img_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            SuperConv(out_channels, out_channels, img_dim=img_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Dropout(0.3),
            SuperInstanceNorm(out_channels, img_dim=img_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            SuperConv(out_channels, out_channels, img_dim=img_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            SuperInstanceNorm(out_channels, img_dim=img_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        out =self.layer3(x2) + x1
        return out

class Encoder(nn.Module):
    def __init__(self, n_base_filters):
        super(Encoder, self).__init__()

        self.en1 = EncoderLayer(1, n_base_filters, True)
        self.en2 = EncoderLayer(n_base_filters, 2*n_base_filters, True, stride=2)
        self.en3 = EncoderLayer(2*n_base_filters, 4*n_base_filters, True, stride=2)
        self.en4 = EncoderLayer(4*n_base_filters, 8*n_base_filters, True, stride=2)

    def forward(self, x):
        e1_out = self.en1(x)
        e2_out = self.en2(e1_out)
        e3_out = self.en3(e2_out)
        e4_out = self.en4(e3_out)

        return [e1_out, e2_out, e3_out, e4_out]


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias, img_dim="3D"):
        super(DecoderLayer, self).__init__()
        self.img_dim = img_dim
        self.layer1 = nn.Sequential(
            SuperConv(in_channels, out_channels, img_dim=img_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            SuperInstanceNorm(out_channels, img_dim=img_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            SuperConv(in_channels, out_channels, img_dim=img_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            SuperInstanceNorm(out_channels, img_dim=img_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            SuperConv(out_channels, out_channels, img_dim=img_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            SuperInstanceNorm(out_channels, img_dim=img_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x, y):

        if self.img_dim == "3D":
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        c1 = self.layer1(x)
        c1_cat = torch.cat((c1, y), dim=1)
        c2 = self.layer2(c1_cat)
        out = self.layer3(c2)
        return out

class Decoder(nn.Module):
    def __init__(self, n_base_filters, config):
        super(Decoder, self).__init__()
        self.config = config
        self.d3 = DecoderLayer(8*n_base_filters, 4*n_base_filters, True)
        self.d2 = DecoderLayer(4 * n_base_filters, 2 * n_base_filters, True)
        self.d1 = DecoderLayer(2 * n_base_filters, 1 * n_base_filters, True)
        self.conv = nn.Conv3d(n_base_filters, 4, kernel_size=1)

    def forward(self, input, temperature=False):
        en1, en2, en3, en4 = input
        de3 = self.d3(en4, en3)
        de2 = self.d2(de3, en2)
        de1 = self.d1(de2, en1)
        logit = self.conv(de1)

        if temperature:
            return torch.softmax(logit, dim=1), torch.softmax(logit / self.config['temperature'], dim=1), de1
        else:
            return torch.softmax(logit, dim=1)

class FusionLayer(nn.Module):
    def __init__(self, in_channels, img_dim="3D"):
        super(FusionLayer, self).__init__()
        self.layer = nn.Sequential(
            SuperConv(4*in_channels, 4, img_dim=img_dim, kernel_size=3, stride=1, padding=1, bias=True),
            SuperInstanceNorm(in_channels, img_dim=img_dim)
        )

        self.layer2 = nn.Sequential(
            SuperConv(4*in_channels, in_channels, img_dim=img_dim, kernel_size=1, stride=1, padding=0, bias=True),
            SuperInstanceNorm(in_channels, img_dim=img_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input):
        f1, f2, f3, f4 = input
        feat_share_cat = torch.cat(input, dim=1)
        att_map = self.layer(feat_share_cat)
        att_map = torch.sigmoid(att_map)
        feat_share = torch.cat([f1 * att_map[:,0].unsqueeze(1).repeat((1,f1.shape[1],1,1,1)),
                                f2 * att_map[:,1].unsqueeze(1).repeat((1,f2.shape[1],1,1,1)),
                                f3 * att_map[:,2].unsqueeze(1).repeat((1,f3.shape[1],1,1,1)),
                                f4 * att_map[:,3].unsqueeze(1).repeat((1,f4.shape[1],1,1,1)) ], dim=1)
        out = self.layer2(feat_share)
        return out

class Teacher(nn.Module):
    def __init__(self, config, n_base_filters=16):
        super(Teacher, self).__init__()
        self.config = config
        self.encoder1 = Encoder(n_base_filters)
        self.encoder2 = Encoder(n_base_filters)
        self.encoder3 = Encoder(n_base_filters)
        self.encoder4 = Encoder(n_base_filters)

        self.fusion1 = FusionLayer(n_base_filters)
        self.fusion2 = FusionLayer(n_base_filters*2)
        self.fusion3 = FusionLayer(n_base_filters*4)
        self.fusion4 = FusionLayer(n_base_filters*8)

        self.decoder = Decoder(n_base_filters, config)

    def forward(self, input):
        t1, t1ce, t2, flair = input
        [en_1_s1, en_1_s2, en_1_s3, en_1_s4] = self.encoder1(t1)
        [en_2_s1, en_2_s2, en_2_s3, en_2_s4] = self.encoder2(t1ce)
        [en_3_s1, en_3_s2, en_3_s3, en_3_s4] = self.encoder3(t2)
        [en_4_s1, en_4_s2, en_4_s3, en_4_s4] = self.encoder4(flair)

        feat1 = self.fusion1([en_1_s1, en_2_s1, en_3_s1, en_4_s1])
        feat2 = self.fusion2([en_1_s2, en_2_s2, en_3_s2, en_4_s2])
        feat3 = self.fusion3([en_1_s3, en_2_s3, en_3_s3, en_4_s3])
        feat4 = self.fusion4([en_1_s4, en_2_s4, en_3_s4, en_4_s4])

        out = self.decoder([feat1, feat2, feat3, feat4])
        return out


class Baseline(nn.Module):
    def __init__(self, config, n_base_filters=16):
        super(Baseline, self).__init__()
        self.encoder1 = Encoder(n_base_filters)
        self.config = config
        self.decoder = Decoder(n_base_filters, config)

    def forward(self, input):
        x_source = torch.cat([input[i] for i in self.config['target_modality']], dim=1)
        en = self.encoder1(x_source)
        out = self.decoder(en)
        return out


class KDNet(nn.Module):
    def __init__(self, config, n_base_filters=16):
        super(KDNet, self).__init__()
        self.config = config
        self.encoder_student = Encoder(n_base_filters)
        self.encoder1 = Encoder(n_base_filters)
        self.encoder2 = Encoder(n_base_filters)
        self.encoder3 = Encoder(n_base_filters)
        self.encoder4 = Encoder(n_base_filters)

        self.fusion1 = FusionLayer(n_base_filters)
        self.fusion2 = FusionLayer(n_base_filters * 2)
        self.fusion3 = FusionLayer(n_base_filters * 4)
        self.fusion4 = FusionLayer(n_base_filters * 8)

        self.decoder = Decoder(n_base_filters, config)
        self.decoder_student = Decoder(n_base_filters, config)

    def forward(self, input, branch, temperature=False):
        t1, t1ce, t2, flair = input
        x_student = torch.cat([input[i] for i in self.config['target_modality']], dim=1)
        [en_1_s1, en_1_s2, en_1_s3, en_1_s4] = self.encoder1(t1)
        [en_2_s1, en_2_s2, en_2_s3, en_2_s4] = self.encoder2(t1ce)
        [en_3_s1, en_3_s2, en_3_s3, en_3_s4] = self.encoder3(t2)
        [en_4_s1, en_4_s2, en_4_s3, en_4_s4] = self.encoder4(flair)

        en = self.encoder_student(x_student)

        feat1 = self.fusion1([en_1_s1, en_2_s1, en_3_s1, en_4_s1])
        feat2 = self.fusion2([en_1_s2, en_2_s2, en_3_s2, en_4_s2])
        feat3 = self.fusion3([en_1_s3, en_2_s3, en_3_s3, en_4_s3])
        feat4 = self.fusion4([en_1_s4, en_2_s4, en_3_s4, en_4_s4])


        if temperature:
            out_teacher, soft_teacher, de_teacher = self.decoder([feat1, feat2, feat3, feat4], temperature=temperature)
            out_student, soft_student, de_student = self.decoder(en, temperature=temperature)
            return [out_teacher, out_student], [soft_teacher, soft_student], [de_teacher, de_student], [feat4, en[3]]
        else:
            out_student = self.decoder(en)
            return [out_student]

class KDNet_share(nn.Module):
    def __init__(self, config, n_base_filters=16):
        super(KDNet_share, self).__init__()
        self.config = config
        self.encoder1 = Encoder(n_base_filters)
        self.encoder2 = Encoder(n_base_filters)
        self.encoder3 = Encoder(n_base_filters)
        self.encoder4 = Encoder(n_base_filters)

        self.fusion1 = FusionLayer(n_base_filters)
        self.fusion2 = FusionLayer(n_base_filters * 2)
        self.fusion3 = FusionLayer(n_base_filters * 4)
        self.fusion4 = FusionLayer(n_base_filters * 8)

        self.decoder = Decoder(n_base_filters, config)
        self.decoder_student = Decoder(n_base_filters, config)

    def forward(self, input, branch, temperature=False):
        t1, t1ce, t2, flair = input
        x_student = torch.cat([input[i] for i in self.config['target_modality']], dim=1)
        [en_1_s1, en_1_s2, en_1_s3, en_1_s4] = self.encoder1(t1)
        [en_2_s1, en_2_s2, en_2_s3, en_2_s4] = self.encoder2(t1ce)
        [en_3_s1, en_3_s2, en_3_s3, en_3_s4] = self.encoder3(t2)
        [en_4_s1, en_4_s2, en_4_s3, en_4_s4] = self.encoder4(flair)

        en = self.encoder2(x_student)

        feat1 = self.fusion1([en_1_s1, en_2_s1, en_3_s1, en_4_s1])
        feat2 = self.fusion2([en_1_s2, en_2_s2, en_3_s2, en_4_s2])
        feat3 = self.fusion3([en_1_s3, en_2_s3, en_3_s3, en_4_s3])
        feat4 = self.fusion4([en_1_s4, en_2_s4, en_3_s4, en_4_s4])


        if temperature:
            out_teacher, soft_teacher, de_teacher = self.decoder([feat1, feat2, feat3, feat4], temperature=temperature)
            out_student, soft_student, de_student = self.decoder(en, temperature=temperature)
            return [out_teacher, out_student], [soft_teacher, soft_student], [de_teacher, de_student], [feat4, en[3]]
        else:
            out_student = self.decoder(en)
            return [out_student]



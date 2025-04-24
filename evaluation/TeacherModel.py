# -*- coding = utf-8 -*-
# @Time:2023/4/4 21:38
# @Author : Zhang Tong
# @File:TeacherModel.py
# @Software:PyCharm

import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import transforms

mytransforms = transforms.Compose([transforms.ToTensor()])
class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=0, batch_size=8, *args, **kwargs):
        if transform is None:
            transform = mytransforms
        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )


class Encoder_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Encoder_block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Encoder_Network(nn.Module):
    def __init__(self):
        super(Encoder_Network, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.block2 = Encoder_block(16, 32)
        self.block3 = Encoder_block(32, 64)
        self.block4 = Encoder_block(64, 128)
        self.block5 = Encoder_block(128, 256)
        self.block6 = Encoder_block(256, 128)
        self.block7 = Encoder_block(128, 64)
        self.block8 = Encoder_block(64, 32)
        self.block9 = Encoder_block(32, 16)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        y = self.conv1(x)
        out = self.norm1(y)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.conv2(out)
        return out


def conv3x3(in_channels, out_channels, stride, padding, dilation, groups):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                     groups=groups, bias=False)


def conv1x1(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Conv_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv_block, self).__init__()
        self.conv3_3 = nn.Conv2d(input_channels, output_channels, kernel_size=3)

    def forward(self, x):
        out = self.DB(x)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channles):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()
        self.conv = nn.Conv2d(in_channels, out_channles, kernel_size=1, stride=1, bias=False)

    def forward(self, input):
        out = self.conv(self.relu(self.bn(input)))
        return out


class decode_Network(nn.Module):
    def __init__(self):
        super(decode_Network, self).__init__()
        self.inter_channels = 64
        self.grow_rate = 32

        self.origin_conv = conv3x3(3, 32, stride=1, padding=1, dilation=1, groups=1)
        self.origin_bn = nn.BatchNorm2d(32)
        self.origin_act = nn.LeakyReLU()

        self.res1 = conv1x1(32, 192, stride=1)

        self.s1_branch1 = nn.Sequential(
            conv1x1(32, 64, stride=1),
            nn.ReLU(inplace=True))
        self.s1_branch2 = nn.Sequential(
            conv3x3(32, 64, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU(inplace=True))
        self.s1_branch3 = nn.Sequential(
            conv1x1(32, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=3, padding=3, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        self.s1_branch4 = nn.Sequential(
            conv1x1(32, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=6, padding=6, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        # self.td1 = Transition(192, 192//2)  #96
        self.res2 = conv1x1(192, 288, stride=1)

        self.s2_branch1 = nn.Sequential(
            conv1x1(192, 64, stride=1),
            nn.ReLU(inplace=True))
        self.s2_branch2 = nn.Sequential(
            conv1x1(192, 96, stride=1),
            nn.ReLU(inplace=True),
            conv3x3(96, 128, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU(inplace=True))
        self.s2_branch3 = nn.Sequential(
            conv1x1(192, 16, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True))
        self.s2_branch4 = nn.Sequential(
            conv1x1(192, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=6, padding=6, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        self.s2_branch5 = nn.Sequential(
            conv1x1(192, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=12, padding=12, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())

        # self.td2 = Transition(272,272//2)
        self.res3 = conv1x1(288, 512, stride=1)

        self.s3_branch1 = nn.Sequential(
            conv1x1(288, 128, stride=1),
            nn.ReLU(inplace=True))
        self.s3_branch2 = nn.Sequential(
            conv1x1(288, 128, stride=1),
            nn.ReLU(inplace=True),
            conv3x3(128, 256, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU(inplace=True))
        self.s3_branch3 = nn.Sequential(
            conv1x1(288, 32, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True))
        self.s3_branch4 = nn.Sequential(
            conv1x1(288, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=12, padding=12, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        self.s3_branch5 = nn.Sequential(
            conv1x1(288, self.inter_channels, stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            conv3x3(self.inter_channels, self.grow_rate, stride=1, dilation=18, padding=18, groups=1),
            nn.BatchNorm2d(self.grow_rate),
            nn.ReLU())
        # self.td3 = Transition(288, 288 // 2)

        self.conv_final = nn.Conv2d(512, 3, kernel_size=3, padding=1, stride=1)
        # self.sigmo = nn.Sigmoid()

    def forward(self, input):
        res = input
        out = self.origin_act(self.origin_bn(self.origin_conv(input)))
        res = self.res1(out)  # 32,192//2

        s1_branch1 = self.s1_branch1(out)
        s1_branch2 = self.s1_branch2(out)
        s1_branch3 = self.s1_branch3(out)
        s1_branch4 = self.s1_branch4(out)
        # B x 192 x 128 x128
        out = torch.cat((s1_branch1, s1_branch2, s1_branch3, s1_branch4), dim=1)
        out += res

        res = self.res2(out)
        s2_branch1 = self.s2_branch1(out)
        s2_branch2 = self.s2_branch2(out)
        s2_branch3 = self.s2_branch3(out)
        s2_branch4 = self.s2_branch4(out)
        s2_branch5 = self.s2_branch5(out)
        # B x 288 x 128 x 128
        out = torch.cat((s2_branch1, s2_branch2, s2_branch3, s2_branch4, s2_branch5), dim=1)
        out += res

        res = self.res3(out)
        s3_branch1 = self.s3_branch1(out)
        s3_branch2 = self.s3_branch2(out)
        s3_branch3 = self.s3_branch3(out)
        s3_branch4 = self.s3_branch4(out)
        s3_branch5 = self.s3_branch5(out)
        # B x 512 x 128 x 128
        out = torch.cat((s3_branch1, s3_branch2, s3_branch3, s3_branch4, s3_branch5), dim=1)
        out += res

        output = self.conv_final(out)
        return output


class Stegano_Network(nn.Module):
    def __init__(self):
        super(Stegano_Network, self).__init__()
        self.hidden = Encoder_Network()
        self.reveal = decode_Network()

    def forward(self, x):
        Stego = self.hidden(x)
        revealed_message = self.reveal(Stego)
        return Stego, revealed_message





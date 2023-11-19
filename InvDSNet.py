import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import module_util as mutil
from time import time
# from torchstat import stat
import cv2
import torchvision.transforms as transforms
from PIL import Image
from skimage.filters import threshold_otsu

to_pil = transforms.ToPILImage()
to_gray = transforms.Grayscale(num_output_channels=1)

from thop import profile


# snow loss attention (only in the middle) with multi unet


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature + channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init != 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'Resnet':
            return ResBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)

            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class attention_resnet(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(attention_resnet, self).__init__()

        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature, channel_out, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channel_out, 3, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.relu1(self.conv2(residual))
        out = self.relu1(self.conv3(residual))  # feature
        out_1 = self.relu1(self.conv4(out))  # out
        out_mask = self.conv5(out_1)  # 3
        out_mask_loss = self.conv6(out_mask)  # 1
        attention_map = self.sigmoid(out_mask_loss)  # 1
        out = torch.mul(out_1, attention_map) + out
        # snow_loss = self.conv7(snow)
        return out, out_mask_loss


class fuse_attention_coupling(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=1.):
        super(fuse_attention_coupling, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        # self.F = multi_resnet1(self.split_len2, self.split_len1)
        # self.G = multi_resnet2(self.split_len1, self.split_len2)
        # self.H = multi_resnet2(self.split_len1, self.split_len2)
        self.F = attention_resnet(self.split_len2, self.split_len1)
        self.G = attention_resnet(self.split_len1, self.split_len2)
        self.H = attention_resnet(self.split_len1, self.split_len2)

    def forward(self, x1, x2, rev=False):
        # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            f1, f2 = self.F(x2)
            y1 = x1 + f1
            h1, h2 = self.H(y1)
            self.s = self.clamp * (torch.sigmoid(h1) * 2 - 1)
            g1, g2 = self.G(y1)
            y2 = x2.mul(torch.exp(self.s)) + g1
        else:
            h1, h2 = self.H(x1)
            self.s = self.clamp * (torch.sigmoid(h1) * 2 - 1)
            g1, g2 = self.G(x1)
            y2 = (x2 - g1).div(torch.exp(self.s))
            f1, f2 = self.F(y2)
            y1 = x1 - f1

        return y1, y2, f2, h2, g2

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class fuse_coupling(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=1.):
        super(fuse_coupling, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        # self.F = multi_resnet1(self.split_len2, self.split_len1)
        # self.G = multi_resnet2(self.split_len1, self.split_len2)
        # self.H = multi_resnet2(self.split_len1, self.split_len2)
        self.F = ResBlock(self.split_len2, self.split_len1)
        self.G = ResBlock(self.split_len1, self.split_len2)
        self.H = ResBlock(self.split_len1, self.split_len2)

    def forward(self, x1, x2, rev=False):
        # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return y1, y2

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class InvNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(InvNet, self).__init__()

        operations = []

        current_channel = channel_in
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.tanh = nn.Tanh()
        self.haar1 = HaarDownsampling(3)
        self.haar2 = HaarDownsampling(12)
        self.fuse_coupling1 = fuse_coupling(24, 12)
        self.fuse_coupling2 = fuse_coupling(24, 12)

    def forward(self, x, x_in, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0
        count = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if count == 8:
                    x2 = self.haar1(x_in, rev)
                    x2_out, out = self.fuse_coupling1(x2, out, rev)
                    x2_out, out = self.fuse_coupling2(x2_out, out, rev)
                    x_out = self.haar2(x2_out, rev)
                count = count + 1
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if count == 8:
                    x2_in = self.haar2(x_in, rev)
                    x2_out, out = self.fuse_coupling2(x2_in, out, rev)
                    x2_out, out = self.fuse_coupling1(x2_out, out, rev)
                    x_out = self.haar1(x2_out, rev)
                count = count + 1
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return self.tanh(out), jacobian
        else:
            return out, x_out


class InvDDNet(nn.Module):
    def __init__(self):
        super(InvDDNet, self).__init__()
        self.down_num = 2
        self.netG = InvNet(3, 3, subnet('Resnet', 'xavier'), [8, 8], self.down_num).cuda()
        self.netSubG = InvNet(3, 3, subnet('Resnet', 'xavier'), [8, 8], self.down_num).cuda()
        self.fuse_coupling = fuse_attention_coupling(96, 48).cuda()
        self.fuse_coupling1 = fuse_coupling(96, 48).cuda()
        self.fuse_coupling2 = fuse_coupling(96, 48).cuda()

    def forward(self, x, x_in, rev=False):
        if rev:
            middle, middle_fuse = self.netSubG(x, x_in)
            middle_fuse, middle = self.fuse_coupling2(middle_fuse, middle, rev)
            middle_fuse, middle_out, f, h, g = self.fuse_coupling(middle_fuse, middle, rev)
            middle_fuse, middle_out = self.fuse_coupling1(middle_fuse, middle_out, rev)
            out, fuse_out = self.netG(middle_out, middle_fuse, rev=True)
            return out, fuse_out, f, h, g, time()
        else:
            middle, middle_fuse = self.netG(x, x_in)
            middle_fuse, middle = self.fuse_coupling1(middle_fuse, middle, rev)
            middle_fuse, middle_out, f, h, g = self.fuse_coupling(middle_fuse, middle, rev)
            middle_fuse, middle_out = self.fuse_coupling2(middle_fuse, middle_out, rev)
            out, fuse_out = self.netSubG(middle_out, middle_fuse, rev=True)
            return out, fuse_out, f, h, g, time()


if __name__ == '__main__':
    print('here')

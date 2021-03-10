import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import warping.homography as warp_homo
import utils.img_utils as img_utils
import random

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=0, padding=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=padding, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, bn_running_avg=False):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad,
                                   dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes, track_running_stats=bn_running_avg))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad, bn_running_avg=False):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes,
                                   kernel_size=kernel_size, padding=pad,
                                   stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes, track_running_stats=bn_running_avg))

def conv2d_leakyRelu(ch_in, ch_out, kernel_size, stride, pad, use_bias=True, dilation = 1):
    r'''
    Conv2d + leakyRelu
    '''
    return nn.Sequential(
            nn.Conv2d(
                ch_in, ch_out, kernel_size=kernel_size, stride = stride,
                padding = dilation if dilation >1 else pad, dilation = dilation, bias= use_bias),
            nn.LeakyReLU())

def conv2dTranspose_leakyRelu(ch_in, ch_out, kernel_size, stride, pad, use_bias = True, dilation=1 ):
    r'''
    ConvTrans2d + leakyRelu
    '''
    return nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size = kernel_size, stride =stride,
                padding= pad, bias = use_bias, dilation = dilation),
            nn.LeakyReLU())

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation, bn_running_avg=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation, bn_running_avg),
                                   nn.ReLU(inplace=True))
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation, bn_running_avg)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return out

class SweepDecoder(nn.Module):
    '''
    The refinement taking the DPV, using the D dimension as the feature dimension, plus the image features,
    then upsample the DPV (4 time the input dpv resolution)
    '''

    def __init__(self):
        super(SweepDecoder, self).__init__()

        self.conv0 = conv2d_leakyRelu(
            ch_in=34, ch_out=32, kernel_size=3, stride=1, pad=1, use_bias=True)

        self.conv0_1 = conv2d_leakyRelu(
            ch_in=32, ch_out=32, kernel_size=3, stride=1, pad=1, use_bias=True)

        self.trans_conv0 = conv2dTranspose_leakyRelu(
            ch_in=32, ch_out=16, kernel_size=4, stride=2, pad=1, use_bias=True)

        self.conv1 = conv2d_leakyRelu(
            ch_in=32, ch_out=32, kernel_size=3, stride=1, pad=1, use_bias=True)

        self.conv1_1 = conv2d_leakyRelu(
            ch_in=32, ch_out=32, kernel_size=3, stride=1, pad=1, use_bias=True)

        self.trans_conv1 = conv2dTranspose_leakyRelu(
            ch_in=32, ch_out=16, kernel_size=4, stride=2, pad=1, use_bias=True)

        self.conv2 = conv2d_leakyRelu(
            ch_in=19, ch_out=16, kernel_size=3, stride=1, pad=1, use_bias=True)

        self.conv2_1 = conv2d_leakyRelu(
            ch_in=16, ch_out=16, kernel_size=3, stride=1, pad=1, use_bias=True)

        self.conv2_2 = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1, bias=True)

        self.apply(self.weight_init)

    def forward(self, input_raw, img_features):
        '''
        dpv_raw - the low resolution (.25 image size) dpv (N D H W)
        img_features - list of image features [ .25 image size, .5 image size, 1 image size]

        NOTE:
        dpv_raw from 0, 1 (need to exp() if in log scale)

        output dpv in log-scale
        '''

        # print(input_raw.shape) # 2,64,80
        # print(img_features[0].shape) # 32, 64,80
        # print(img_features[1].shape) # 16, 128, 160
        # print(img_features[2].shape) # 3, 256, 320

        conv0_out = self.conv0(torch.cat([input_raw, img_features[0]], dim=1)) # 2 + 32
        conv0_1_out = self.conv0_1(conv0_out)
        trans_conv0_out = self.trans_conv0(conv0_1_out) # 1, 16, 128, 160


        conv1_out = self.conv1(torch.cat([trans_conv0_out, img_features[1]], dim=1)) # 16 + 16
        conv1_1_out = self.conv1_1(conv1_out)
        trans_conv1_out = self.trans_conv1(conv1_1_out) # 1, 16, 256, 320

        conv2_out = self.conv2(torch.cat([trans_conv1_out, img_features[2]], dim=1))
        conv2_1_out = self.conv2_1(conv2_out)
        conv2_2_out = self.conv2_2(conv2_1_out) # 1, 2, 256, 320

        return conv2_2_out

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[1]
            factor = (n + 1) // 2
            if n % 2 == 1:
                center = factor - 1
            else:
                center = factor - .5

            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            m.weight.data.copy_(torch.from_numpy(weights_np))

class SweepEncoder(nn.Module):
    def __init__(self, feature_dim=32, bn_running_avg=False, multi_scale=True):
        '''
        inputs:
        multi_scale - if output multi-sclae features:
        [1/4 scale of input image, 1/2 scale of input image]
        '''

        super(SweepEncoder, self).__init__()

        MUL = feature_dim / 64.
        S0 = int(16 * MUL)
        S1 = int(32 * MUL)
        S2 = int(64 * MUL)
        #S3 = int(128 * MUL)

        self.inplanes = S1
        self.multi_scale = multi_scale

        self.bn_ravg = bn_running_avg
        self.firstconv = nn.Sequential(convbn(3, S1, 3, 2, 1, 1, self.bn_ravg), nn.ReLU(inplace=True),
                                       convbn(S1, S1, 3, 1, 1, 1, self.bn_ravg), nn.ReLU(inplace=True),
                                       convbn(S1, S1, 3, 1, 1, 1, self.bn_ravg), nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, S1, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, S2, 3, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, S2, 3, 1, 1, 1)

        self.lastconv = nn.Sequential(convbn(64, 32, 1, 1, 0, self.bn_ravg),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(32,
                                                2, kernel_size=1, padding=0, stride=1, bias=False))

        self.apply(self.weight_init)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, track_running_stats=self.bn_ravg), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation, self.bn_ravg))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation, self.bn_ravg))

        return nn.Sequential(*layers)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def forward(self, x):
        output = self.firstconv(x) # 3 Convolutions [3, 256, 320] -> [16, 128, 160]
        output_layer1 = self.layer1(output) # 3 Convs in Layer 1 [16, 128, 160] -> [16, 128, 160]
        output_raw = self.layer2(output_layer1) # 3 Convs in Layer 1 [16, 128, 160] -> [32, 64, 80]
        output_skip = self.layer3(output_raw) # 3 Convs in Layer 1 [32, 64, 80] -> [64, 64, 80]

        output_feature = torch.cat(
            (output_raw, output_skip), 1)
        output_feature = self.lastconv(output_feature)

        if self.multi_scale:
            return output_layer1, output_raw, output_feature
        else:
            return output_feature

class SweepModel(nn.Module):
    def __init__(self, cfg, id):
        super(SweepModel, self).__init__()
        self.cfg = cfg
        self.sigma_soft_max = self.cfg.var.sigma_soft_max
        self.feature_dim = self.cfg.var.feature_dim
        self.nmode = self.cfg.var.nmode
        self.D = self.cfg.var.ndepth
        self.bn_avg = self.cfg.var.bn_avg
        self.id = id

        # Encoder
        self.sweep_encoder = SweepEncoder(feature_dim = self.feature_dim, multi_scale = True, bn_running_avg = self.bn_avg)
        self.sweep_decoder = SweepDecoder()

        # Apply Weights
        self.apply(self.weight_init)

    def set_viz(self, viz):
        self.viz = viz

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[1]
            factor = (n+1) // 2
            if n %2 ==1:
                center = factor - 1
            else:
                center = factor -.5

            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            m.weight.data.copy_(torch.from_numpy(weights_np))

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def constrain_output(self, input):
        ll = nn.LeakyReLU(0.1)
        pp = input[:, 0, :, :]
        ps = input[:, 1, :, :]

        # # Relu Constraint
        # pp = ll(pp)
        # ps = ll(ps)
        # pp = torch.clamp(pp, 0, 1)
        # ps = 0.1 + torch.clamp(ps, 0, 1) * 4

        # Sigmoid Constraint
        # pp = torch.sigmoid(pp)
        # ps = 0.1 + torch.sigmoid(ps) * 4

        # No Constraint?

        return torch.cat([pp.unsqueeze(1), ps.unsqueeze(1)], dim=1)

    def forward_int(self, model_input):
        # Remove other element from model input
        model_input["rgb"] = model_input["rgb"][:,-1,:,:,:]

        # Params
        bsize = model_input["rgb"].shape[0]
        d_candi = model_input["d_candi"]

        # Feature Extraction
        rgb = model_input["rgb"]
        f1, f2, small_output = self.sweep_encoder(rgb) # [8,32,128,192] [8,64,64,96]

        # Constrain small output
        small_output = self.constrain_output(small_output)

        # Decoder
        skip_connections = [f2, f1, rgb]
        big_output = self.sweep_decoder(small_output, skip_connections)

        # Constrain big output
        big_output = self.constrain_output(big_output)

        # Return 
        return {"output": [small_output], "output_refined": [big_output]}

    def forward(self, inputs):
        outputs = []
        for input in inputs:
            outputs.append(self.forward_int(input))
        return outputs
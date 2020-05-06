import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DefaultModel(nn.Module):
    def __init__(self, cfg):
        super(DefaultModel, self).__init__()
        self.cfg = cfg
        self.conv_1x1 = nn.Sequential(conv(3, 32, kernel_size=3, stride=1, dilation=0, padding=1),
                                      nn.MaxPool2d(2),
                                      conv(32, self.cfg.var.ndepth, kernel_size=3, stride=1, dilation=0, padding=1),
                                      nn.MaxPool2d(2),
                                      )

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

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

    def forward(self, input):
        images = input["rgb"][:, -1, :, :, :]
        output = self.conv_1x1(images)
        return {"output": output, "output_refined": None, "flow": None, "flow_refined": None}
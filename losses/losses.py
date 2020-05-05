import torch.nn as nn
import torch.nn.functional as F
import torch

class DefaultLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(DefaultLoss, self).__init__()
        self.cfg = cfg

    def forward(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        return torch.sum(torch.abs(output-target))
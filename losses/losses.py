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

        output_left, output_right = output
        target_left, target_right = target

        left_loss = 0.
        right_loss = 0.
        for b in range(0, len(target_left["soft_labels"])):
            label_left = target_left["soft_labels"][b].unsqueeze(0)
            label_right = target_right["soft_labels"][b].unsqueeze(0)

            left_loss += torch.sum(torch.abs(output_left["output"] - 0))
            right_loss += torch.sum(torch.abs(output_right["output"] - 0))

        loss = left_loss + right_loss

        return loss
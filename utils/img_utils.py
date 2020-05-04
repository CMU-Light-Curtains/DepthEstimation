import numpy as np
import math
import PIL.Image as image
import warping.homography as warp_homo

import torch
import torch.nn.functional as F
import torchvision
import cv2

def torchrgb_to_cv2(input, demean=True):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    input = input.detach().clone()
    if demean:
        input[0, :, :] = input[0, :, :] * __imagenet_stats["std"][0] + __imagenet_stats["mean"][0]
        input[1, :, :] = input[1, :, :] * __imagenet_stats["std"][1] + __imagenet_stats["mean"][1]
        input[2, :, :] = input[2, :, :] * __imagenet_stats["std"][2] + __imagenet_stats["mean"][2]
    return cv2.cvtColor(input[:, :, :].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)

def powerf(d_min, d_max, nDepth, power):
    f = lambda x: d_min + (d_max - d_min) * x
    x = np.linspace(start=0, stop=1, num=nDepth)
    x = np.power(x, power)
    candi = [f(v) for v in x]
    return np.array(candi)

def minpool(tensor, scale, default=0):
    if default:
        tensor_copy = tensor.clone()
        tensor_copy[tensor_copy == 0] = default
        tensor_small = -F.max_pool2d(-tensor_copy, scale)
        tensor_small[tensor_small == default] = 0
    else:
        tensor_small = -F.max_pool2d(-tensor, scale)
    return tensor_small

def intr_scale(intr, raw_img_size, img_size):
    uchange = float(img_size[0]) / float(raw_img_size[0])
    vchange = float(img_size[1]) / float(raw_img_size[1])
    intr_small = intr.copy()
    intr_small[0, :] *= uchange
    intr_small[1, :] *= vchange
    return intr_small

def intr_scale_unit(intr, scale=1.):
    intr_small = intr.copy()
    intr_small[0, :] *= scale
    intr_small[1, :] *= scale
    return intr_small
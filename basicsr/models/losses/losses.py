# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from math import exp, log10, sqrt

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

#L1 Sobel loss
class EdgeDifferenceLoss(nn.Module):
    def __init__(self, device=torch.device('cuda'), loss=nn.L1Loss):
        super(EdgeDifferenceLoss, self).__init__()

        self.device = device
        self.conv_op_x = nn.Conv2d(3, 1, 3, bias=False)
        self.conv_op_y = nn.Conv2d(3, 1, 3, bias=False)
        self.conv_op_xy = nn.Conv2d(3, 1, 3, bias=False)
        self.conv_op_yx = nn.Conv2d(3, 1, 3, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_xy = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]], dtype='float32')
        sobel_kernel_yx = np.array([[[2, 1, 0], [1, 0, -1], [0, -1, -2]], 
                                    [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
                                    [[2, 1, 0], [1, 0, -1], [0, -1, -2]]], dtype='float32')
        
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))
        sobel_kernel_xy = sobel_kernel_xy.reshape((1, 3, 3, 3))
        sobel_kernel_yx = sobel_kernel_yx.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_xy.weight.data = torch.from_numpy(sobel_kernel_xy).to(device)
        self.conv_op_yx.weight.data = torch.from_numpy(sobel_kernel_yx).to(device)

        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False
        self.conv_op_xy.weight.requires_grad = False
        self.conv_op_yx.weight.requires_grad = False

        self.loss = loss().to(device)

    def get_sobel_output(self, x):
        
        sobel_x = self.conv_op_x(x)
        sobel_y = self.conv_op_y(x)
        sobel_xy = self.conv_op_xy(x)
        sobel_yx = self.conv_op_yx(x)
        sobel_out = torch.abs(sobel_x) + torch.abs(sobel_y) + \
            torch.abs(sobel_xy) + torch.abs(sobel_yx)

        return sobel_out

    def forward(self, img1, img2):

        sobel_1 = self.get_sobel_output(img1)
        sobel_2 = self.get_sobel_output(img2)

        loss = self.loss(sobel_1, sobel_2)

        return loss

#Total variation
#Anisotropic version
class BackgroundBlurLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(BackgroundBlurLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        # h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        h_tv = torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]).sum()
        w_tv = torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class ForegroundEdgeLoss(nn.Module):
    def __init__(self, device=torch.device('cuda'), loss=nn.L1Loss):
        super(ForegroundEdgeLoss, self).__init__()

        self.device = device
        self.conv_op_x = nn.Conv2d(3, 1, 3, bias=False)
        self.conv_op_y = nn.Conv2d(3, 1, 3, bias=False)
        self.conv_op_xy = nn.Conv2d(3, 1, 3, bias=False)
        self.conv_op_yx = nn.Conv2d(3, 1, 3, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_xy = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]], dtype='float32')
        sobel_kernel_yx = np.array([[[2, 1, 0], [1, 0, -1], [0, -1, -2]], 
                                    [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
                                    [[2, 1, 0], [1, 0, -1], [0, -1, -2]]], dtype='float32')
        
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))
        sobel_kernel_xy = sobel_kernel_xy.reshape((1, 3, 3, 3))
        sobel_kernel_yx = sobel_kernel_yx.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_xy.weight.data = torch.from_numpy(sobel_kernel_xy).to(device)
        self.conv_op_yx.weight.data = torch.from_numpy(sobel_kernel_yx).to(device)

        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False
        self.conv_op_xy.weight.requires_grad = False
        self.conv_op_yx.weight.requires_grad = False

        self.loss = loss().to(device)

    def get_sobel_output(self, x):
        sobel_x = self.conv_op_x(x)
        sobel_y = self.conv_op_y(x)
        sobel_xy = self.conv_op_xy(x)
        sobel_yx = self.conv_op_yx(x)
        sobel_out = torch.abs(sobel_x) + torch.abs(sobel_y) + \
            torch.abs(sobel_xy) + torch.abs(sobel_yx)

        return sobel_out

    def forward(self, img1):
        b, c, h, w = img1.size()
        sobel_1 = self.get_sobel_output(img1) 
        sobel_out = torch.sum(sobel_1) / (h*w)
        return -sobel_out

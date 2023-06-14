# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# modified from https://github.com/mayorx/matlab_ssim_pytorch_implementation/blob/main/calc_ssim.py
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import numpy as np
from basicsr.metrics.metric_util import reorder_image, to_y_channel
import torch
import lpips

def calculate_lpips(img1, 
                    img2, 
                    crop_border, 
                    input_order='HWC', 
                    test_y_channel=False):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    
    '''if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
    
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    
    loss_fn_alex = lpips.LPIPS(net='alex')

    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:,:,:3], img1[:,:,3:]
        l2, r2 = img2[:,:,:3], img2[:,:,3:]
        return (loss_fn_alex(l1, l2) + loss_fn_alex(r1, r2))/2
    else:
        return loss_fn_alex(img1, img2)'''
    
    loss_fn_alex = lpips.LPIPS(net='alex')
    result = loss_fn_alex(img1, img2)
    result = result[0][0][0][0] #value buried under a bunch of brackets

    return result
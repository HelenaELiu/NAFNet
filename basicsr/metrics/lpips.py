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
                    img2):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    loss_fn_alex = lpips.LPIPS(net = 'alex', verbose = False)
    result = loss_fn_alex(img1, img2)
    
    result = result[0][0][0][0].item() #value buried under a bunch of brackets
    return result
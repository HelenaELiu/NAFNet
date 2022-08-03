import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from basicsr.models.archs.OCL_NAFNet_arch import NAFNetPlain, NAFNetResidualOCLI, NAFNetResidualOCLII, NAFNetResidualOCLConv
import argparse
from basicsr.models.archs.discriminator import get_discriminator
import yaml
from basicsr.models.archs.vgg import vgg_19
from basicsr.models.losses.losses import PSNRLoss, SSIM
import numpy as np
from basicsr.data.OCL_dataset import DataLoaderCenterShiftToShift, DataLoaderCenterViewDiffAndShiftToShift, DataLoaderCenterViewsAndShiftToShift, \
    DataLoaderCenterViewsToShift
import random
import os
from os import path as osp
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from typing import Any, Union
from tqdm import tqdm as _tqdm
from pytorch_lightning.trainer import Trainer
from collections import OrderedDict

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    with open(args.opt, 'r') as f:
        cfg = yaml.safe_load(f)
    
    return cfg

def get_dataloaders(opt):
    datasets = {'DataLoaderCenterShiftToShift': DataLoaderCenterShiftToShift, 
                'DataLoaderCenterViewDiffAndShiftToShift': DataLoaderCenterViewDiffAndShiftToShift,
                'DataLoaderCenterViewsAndShiftToShift': DataLoaderCenterViewsAndShiftToShift, 
                'DataLoaderCenterViewsToShift': DataLoaderCenterViewsToShift}
    train_dataset = datasets[opt['type']](opt = opt, test = False)
    test_dataset = datasets[opt['type']](opt = opt, test = True)

    num_gpu = cfg['num_gpu']

    #DataLoader settings
    use_shuffle = opt['use_shuffle']
    batch_size = opt['batch_size_per_gpu']
    num_worker_per_gpu = opt['num_worker_per_gpu']
    dataset_enlarge_ratio = opt['dataset_enlarge_ratio']

    train_dataloader = DataLoader(train_dataset, batch_size, use_shuffle, num_workers= num_worker_per_gpu)
    test_dataloader = DataLoader(test_dataset, 1, False)

    return train_dataloader, test_dataloader

def prepare_dirs(opt):
    os.makedirs(cfg['train']['default_root_dir'], exist_ok = True)

def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

class NAFGAN(pl.LightningModule):

    def __init__(self, cfg, train_opt):
        super(NAFGAN, self).__init__()

        g_name = cfg['network_g']['type']
        in_channels = cfg['network_g']['in_channels']
        out_channels = cfg['network_g']['out_channels']
        width = cfg['network_g']['width']
        enc_blk_nums = cfg['network_g']['enc_blk_nums']
        middle_blk_num = cfg['network_g']['middle_blk_num']
        dec_blk_nums = cfg['network_g']['dec_blk_nums']

        generators = {'NAFNetPlain': NAFNetPlain, 'NAFNetResidualOCLI': NAFNetResidualOCLI, 
        'NAFNetResidualOCLII': NAFNetResidualOCLII, 'NAFNetResidualOCLConv': NAFNetResidualOCLConv}
        
        self.generator = generators[g_name](in_channels  = in_channels, out_channels = out_channels, width = width, 
        enc_blk_nums = enc_blk_nums, middle_blk_num = middle_blk_num, dec_blk_nums = dec_blk_nums)

        #vgg 
        self.vgg = vgg_19()
        self.vgg.eval()

        #discriminators
        self.patchD = get_discriminator(cfg)['patch']
        self.fullD = get_discriminator(cfg)['full']

        #losses
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        self.SSIM_loss = SSIM()
        self.PSNR_loss = PSNRLoss()
        
        #train settings
        self.train_opt = cfg['train']

        #epoch values
        self.warmup_epochs = self.train_opt['warmup_epochs']
        self.full_epochs = self.train_opt['full_epochs']
        
    def forward(self, img):
        return self.generator(img)

    def compute_gradient_penalty_patch(self, netD, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.Tensor(np.random.random(real_samples.size())).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = netD(interpolates)
        fake = torch.ones(d_interpolates.size()).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, gt = batch['lq'], batch['gt']
        inputs, gt = inputs.float(), gt.float()

        lambda_gp = self.train_opt['lambda_gp']

        if optimizer_idx == 0:
            generated = self(inputs)

            if self.current_epoch < self.warmup_epochs:
                l1_loss = torch.mean(self.L1_loss(generated, gt))
                ssim_loss = torch.mean(1-self.SSIM_loss(generated, gt))
                g_loss = self.cfg['magic_nums']['l1'] * l1_loss + self.cfg['magic_nums']['ssim']*ssim_loss

            elif self.current_epoch >= self.cfg['training']['warmup_epochs']:
                patch_g_loss = -torch.mean(self.patchD(generated))
                full_g_loss = -torch.mean(self.fullD(generated))
                gan_loss = (patch_g_loss + full_g_loss) / 2

                adv_loss = self.cfg['magic_nums']['g_loss']*gan_loss
                enhanced_vgg = self.vgg(normalize_batch(generated))
                target_vgg = self.vgg(normalize_batch(gt))
                vgg_loss = self.MSE_loss(enhanced_vgg, target_vgg)
                
                l1_loss = torch.mean(self.L1_loss(generated, gt))
                
                ssim_loss = torch.mean(1-self.SSIM_loss(generated, gt))

                g_loss = self.cfg['magic_nums']['l1'] * l1_loss + self.cfg['magic_nums']['ssim'] * ssim_loss \
                    + self.cfg['magic_nums']['vgg'] * vgg_loss + self.cfg['magic_nums']['adv'] * adv_loss

        tqdm_dict = {'g_loss': g_loss}
        output = OrderedDict({
            'loss': g_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output
    
    def configure_optimizers(self):
        lr = self.train_opt['optim_g']['lr']
        betas = self.train_opt['optim_g']['betas']
        optimizer_name = self.train_opt['optim_g']['type']
        optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}

        opt_g = optimizers[optimizer_name](self.generator)(self.generator.parameters(), lr = lr, betas = betas)
        opt_d = optimizers[optimizer_name](list(self.patchD.parameters()) + list(self.fullD.parameters()), lr = lr, betas = betas)

        #add more schedulers later
        if self.train_opt['scheduler']['type'] == 'CosineAnnealingLR':
            T_max = self.train_opt['scheduler']['T_max']
            eta_min = self.train_opt['scheduler']['eta_min']
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max, eta_min) 
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max, eta_min)

        return [opt_g, opt_d], [scheduler_g, scheduler_d]

_PAD_SIZE = 5

class Tqdm(_tqdm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from
        flickering."""
        # this just to make the make docs happy, otherwise it pulls docs which has some issues...
        super().__init__(*args, **kwargs)

    @staticmethod
    def format_num(n: Union[int, float, str]) -> str:
        """Add additional padding to the formatted numbers."""
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
            assert isinstance(n, str)
        if should_be_padded and "e" not in n:
            if "." not in n and len(n) < _PAD_SIZE:
                try:
                    _ = float(n)
                except ValueError:
                    return n
                n += "."
            n += "0" * (_PAD_SIZE - len(n))
        return n

class TQDM_BAR(TQDMProgressBar):
    def init_train_tqdm(self):
        bar =  Tqdm(
            desc=self.train_description,
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            smoothing=0,
        )
        return bar

if __name__ == '__main__':
    cfg = parse_options()

    opt = cfg['datasets']['train']
   
    train_dataloader, test_dataloader = get_dataloaders(opt)

    #Model Training
    model = NAFGAN(cfg)
    bar = TQDM_BAR()
    trainer = Trainer(gpus = -1, default_root_dir= cfg['train']['default_root_dir'], max_epochs = cfg['train']['warmup_epochs'] + cfg['train']['full_epochs'], \
        callbacks = [bar], accumulate_grad_batches= cfg['train']['grad_batches'], accelerator= 'ddp')
    
    trainer.fit(model = model, train_dataloaders= train_dataloader)
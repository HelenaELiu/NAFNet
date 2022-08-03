from tokenize import Double
import torch
import numpy as np
import copy
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from collections import OrderedDict
import functools
import yaml
from torch.utils.data import DataLoader


###############################################################################
# Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, use_parallel=True, learn_residual=True, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output,min = -1,max = 1)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class DicsriminatorTail(nn.Module):
    def __init__(self, nf_mult, n_layers, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(DicsriminatorTail, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(MultiScaleDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.scale_one = nn.Sequential(*sequence)
        self.first_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=3)
        nf_mult_prev = 4
        nf_mult = 8

        self.scale_two = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        nf_mult_prev = nf_mult
        self.second_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=4)
        self.scale_three = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        self.third_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=5)

    def forward(self, input):
        x = self.scale_one(input)
        x_1 = self.first_tail(x)
        x = self.scale_two(x)
        x_2 = self.second_tail(x)
        x = self.scale_three(x)
        x = self.third_tail(x)
        return [x_1, x_2, x]


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def get_fullD(model_config):
    model_d = NLayerDiscriminator(n_layers=5,
                                  norm_layer=get_norm_layer(norm_type=model_config['training']['norm_layer']),
                                  use_sigmoid=False)
    return model_d


def get_discriminator(model_config):
    discriminator_name = model_config['discriminator']['d_name']

    assert discriminator_name in ['no_gan', 'patch_gan', 'double_gan']

    if discriminator_name == 'no_gan':
        model_d = None
    elif discriminator_name == 'patch_gan':
        model_d = NLayerDiscriminator(n_layers=model_config['discriminator']['d_layers'],
                                      norm_layer=get_norm_layer(norm_type=model_config['discriminator']['norm_layer']),
                                      use_sigmoid=False)
    elif discriminator_name == 'double_gan':
        patch_gan = NLayerDiscriminator(n_layers=model_config['discriminator']['d_layers'],
                                        norm_layer=get_norm_layer(norm_type=model_config['discriminator']['norm_layer']),
                                        use_sigmoid=False)
        full_gan = get_fullD(model_config)
        model_d = {'patch': patch_gan,
                   'full': full_gan}

    return model_d


# class DiscLossWGANGP(torch.nn.Module):
#     def __init__(self, LAMBDA = 10):
#         super(DiscLossWGANGP, self).__init__()
#         self.LAMBDA = LAMBDA
#         self.dummy_param = nn.Parameter(torch.empty(0))

#     def get_g_loss(self, netD, realB, fakeB):
#         '''
#         fakeB: fake Bokeh Image
#         '''
#         D_fake = netD(fakeB)
#         return -torch.mean(D_fake)

#     def calc_gradient_penalty(self, netD, real_data, fake_data):
#         '''
#         real_data: real Bokeh image
#         fake_data: fake Bokeh image
#         '''
#         BATCH_SIZE = real_data.shape[0]
#         alpha = torch.rand(real_data.size()).to(real_data.device)

#         interpolates = alpha * real_data + ((1 - alpha) * fake_data)

#         interpolates = autograd.Variable(interpolates, requires_grad=True)
#         disc_interpolates = netD(interpolates)
#         gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                                 grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
#                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
#         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
#         return gradient_penalty

#     def get_loss(self, netD, realB, fakeB):
        
#         #fake
#         D_fake = netD(fakeB)
#         D_fake = torch.mean(D_fake)

#         #real
#         D_real = netD(realB)
#         D_real = torch.mean(D_real)

#         #combined loss
#         loss_D = D_fake - D_real
#         gradient_penalty = self.calc_gradient_penalty(netD, realB, fakeB)
#         return loss_D + gradient_penalty

#     def __call__(self, netD, realB, fakeB):
#         return self.get_loss(netD, realB, fakeB)

class GANFactory:
    factories = {}

    def __init__(self):
        pass

    def add_factory(gan_id, model_factory):
        GANFactory.factories.put[gan_id] = model_factory

    add_factory = staticmethod(add_factory)

    def create_model(gan_id, net_d=None, criterion=None):
        if gan_id not in GANFactory.factories:
            GANFactory.factories[gan_id] = \
                eval(gan_id + '.Factory()')
        return GANFactory.factories[gan_id].create(net_d, criterion)

    create_model = staticmethod(create_model)

# class GANTrainer(nn.Module):
#     def __init__(self, net_d, criterion):
#         super(GANTrainer, self).__init__(net_d, criterion)
#         self.net_d = net_d
#         self.criterion = criterion

#     def loss_d(self, pred, gt):
#         pass

#     def loss_g(self, pred, gt):
#         pass

#     def get_params(self):
#         pass

# class DoubleGAN(nn.Module):
#     def __init__(self, net_d, criterion):
#         super(DoubleGAN, self).__init__()
#         self.net_d = net_d
#         self.criterion = criterion
#         self.patch_d = net_d['patch']
#         self.full_d = net_d['full']
#         self.full_criterion = copy.deepcopy(criterion)

#     def loss_d(self, pred, gt):
#         return (self.criterion(self.patch_d, pred, gt) + self.full_criterion(self.full_d, pred, gt)) / 2

#     def loss_g(self, pred, gt):
#         return (self.criterion.get_g_loss(self.patch_d, pred, gt) + self.full_criterion.get_g_loss(self.full_d, pred,
#                                                                                                   gt)) / 2

#     def get_params(self):
#         return list(self.patch_d.parameters()) + list(self.full_d.parameters())


if __name__ == '__main__':
    # train_dataset = EBB_Dataset('train', './data')
    # train_dataloader = DataLoader(train_dataset, 1, True, num_workers=1, drop_last=True)

    # with open('cfg.yml', 'r') as f:
    #     cfg = yaml.safe_load(f)
    # discriminator = get_discriminator(cfg)
    # criterion_d = DiscLossWGANGP()
    # gan = DoubleGAN(discriminator, criterion_d)
    # gan.to('cuda')
    # adv_loss = 0.004*gan.loss_g(generator_image, gt)
    pass
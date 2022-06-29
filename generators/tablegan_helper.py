#!/usr/bin/env python
# coding: utf-8

from typing import Any, List
import numpy as np
import torch
from torch.nn import (
    BatchNorm2d, Conv2d, ConvTranspose2d, LeakyReLU, Module, Sequential, Sigmoid, Tanh, init, BCELoss, Dropout, Flatten)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sdgym.constants import CATEGORICAL
from sdgym.synthesizers.base import LegacySingleTableBaseline
from sdgym.synthesizers.utils import TableganTransformer, select_device
from functools import partial
import math

def compute_sigma(epsilon, delta, mM):
    return (2*(mM)*math.sqrt(math.log(1/delta)))/epsilon

class Discriminator(Module):
    def __init__(self, meta, side, layers):
        super(Discriminator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, input):
        return self.seq(input)


class Generator(Module):
    def __init__(self, meta, side, layers):
        super(Generator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)


class Classifier(Module):
    def __init__(self, meta, side, layers, device):
        super(Classifier, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        self.valid = True
        # We expect the last one to have a boolean classifier that is categorical
        if meta[-1]['type'] != CATEGORICAL or meta[-1]['size'] != 2:
            self.valid = False

        masking = np.ones((1, 1, side, side), dtype='float32')
        index = len(self.meta) - 1
        self.r = index // side
        self.c = index % side
        masking[0, 0, self.r, self.c] = 0
        self.masking = torch.from_numpy(masking).to(device)

    def forward(self, input):
        label = (input[:, :, self.r, self.c].view(-1) + 1) / 2
        input = input * self.masking.expand(input.size())
        return self.seq(input).view(-1), label


def determine_layers(side, random_dim, num_channels):
    assert side >= 4 and side <= 32
    layer_dims = [(1, side), (num_channels, side // 2)]
    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))
    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True),
            Dropout(0.3)
        ]
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        Flatten(),
        Sigmoid()
    ]
    layers_G: List[Any] = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        print(prev, curr)
        padding = 0
        if (prev[1] * 2 == curr[1] - 1):
            padding = 1
        layers_G += [
            BatchNorm2d(prev[0]),
            LeakyReLU(),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=padding, bias=True)
        ]
    layers_G += [Tanh()]
    print(layers_G)

    layers_C = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_C += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]

    layers_C += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0)]

    return layers_D, layers_G, layers_C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class TableGAN(LegacySingleTableBaseline):
    """docstring for TableganSynthesizer??"""

    def __init__(self,
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=128,
                 epochs=200,
                 opacus=False,
                 epsilon=0.1,
                 delta=1e-5,
                 max_grad = 1,
                 sample_freq = 0.0001):
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale

        self.batch_size = batch_size
        self.epochs = epochs

        self.device = select_device()
        self.opacus = opacus
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad = max_grad
        self.sample_freq = sample_freq

        def process_grad(grad, sigma, C):
            bound = torch.norm(grad, 2).cpu()/C
            bound = bound.numpy()
            clip_value = np.maximum(max_grad, bound)
            grad = grad / clip_value
            grad += (1/batch_size) * torch.normal(mean=0.0, std=(sigma**2)*(C**2), size = grad.shape).to(self.device)
            return grad

        self.hook = process_grad

    def update_column_info(self, columns):
        self.columns = columns

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        sides = list(range(4,64,2))
        # sides = [2,4,6,8,16,32,64]
        for i in sides:
            if i * i >= data.shape[1]:
                self.side = i
                break
        
        self.transformer = TableganTransformer(self.side)
        print(data)
        print(len(data[0]))
        print(self.side)
        self.transformer.fit(data, categorical_columns, ordinal_columns)
        data = self.transformer.transform(data)

        data = torch.from_numpy(data.astype('float32')).to(self.device)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        layers_D, layers_G, layers_C = determine_layers(
            self.side, self.random_dim, self.num_channels)

        self.generator = Generator(self.transformer.meta, self.side, layers_G).to(self.device)
        discriminator = Discriminator(self.transformer.meta, self.side, layers_D).to(self.device)
        classifier = Classifier(self.transformer.meta, self.side, layers_C, self.device).to(self.device)

        optimizer_params = dict(lr=1e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)
        optimizerC = Adam(classifier.parameters(), **optimizer_params)
        laplace = None
        if (self.opacus):
            # TODO: Calculate sigma
            sigma = compute_sigma(self.epsilon, self.delta, self.sample_freq)
            for parameter in discriminator.parameters():
                parameter.register_hook(partial(self.hook, sigma = sigma, C=1))

            for parameter in classifier.parameters():
                parameter.register_hook(partial(self.hook, sigma = sigma, C=1))

            sensitivity = (1)/self.batch_size
            scale = sensitivity/self.epsilon
            laplace = torch.distributions.laplace.Laplace(0, scale)
        criterion = BCELoss()

        print("BEGIN")
        for i in range(self.epochs):
            total_loss, loss_g, loss_c = 0, 0, 0
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
            
                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                loss_d = (
                    -(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                total_loss = loss_d
                total_loss.backward()
                optimizerD.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                optimizerG.zero_grad()
                y_fake = discriminator(fake)
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True)
                if (self.opacus):
                    loss_mean = torch.norm(torch.mean(fake, dim=0) - torch.mean(real, dim=0) - laplace.sample(), 1)
                    loss_std = torch.norm(torch.std(fake, dim=0) - torch.std(real, dim=0) - laplace.sample(), 1)
                else:
                    loss_mean = torch.norm(torch.mean(fake, dim=0) - torch.mean(real, dim=0), 1)
                    loss_std = torch.norm(torch.std(fake, dim=0) - torch.std(real, dim=0), 1)
                
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                if classifier.valid:
                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fake)

                    loss_cc = binary_cross_entropy_with_logits(real_pre, real_label)
                    loss_cg = binary_cross_entropy_with_logits(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()
                    loss_c = (loss_cc, loss_cg)
                else:
                    loss_c = None

                if((id_ + 1) % 50 == 0):
                    print("epoch", i + 1, "step", id_ + 1, total_loss.data, loss_g.data, loss_c if loss_c else None)

            print("epoch done", i + 1, total_loss, loss_g, loss_c)

    def sample(self, n):
        self.generator.eval()

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
            fake = self.generator(noise)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        return self.transformer.inverse_transform(data[:n])




def import_model():
    # 256 filters -> 1
    # Discriminator with Wassertein loss
    # UniformGAN 12
    import torch
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self, kgenerator, noise_dim=100, batch_size=16, linear_size=15, size=4):
            super(Generator, self).__init__()
            total = 2 ** (len(kgenerator) + 7) # Close with 256 to 1
            self.total = total
            self.size = size
            self.batch_size = batch_size
            self.noise_dim = noise_dim
            self.fc1 = nn.Sequential(nn.Linear(noise_dim, 1*1*total, False))
            seq = []
            kgenerator = kgenerator.copy()
            while len(kgenerator) > 1:
                k, s, p, o = kgenerator.pop(0)
                print(k,s,p,o, total)
                seq.extend([nn.ConvTranspose2d(total, total//2, kernel_size=k, stride=s, padding=p, output_padding=o, bias=False),
                    nn.BatchNorm2d(total//2),
                    nn.LeakyReLU()])
                total //= 2
            self.seq = nn.Sequential(*seq)
            k,s,p,o = kgenerator.pop(0)
            self.convfinal = nn.ConvTranspose2d(total, 1, kernel_size=k, stride=s, padding=p, output_padding=o, bias=False)
            assert len(kgenerator) == 0
            self.linear = nn.Linear(size*size, linear_size, False)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.reshape(x, (self.batch_size, self.total, 1, 1))
            x = self.seq(x)
            x = self.convfinal(x)
            assert list(x.shape) == [self.batch_size, 1, self.size, self.size]
            x = torch.reshape(x, (self.batch_size, self.size * self.size))
            assert list(x.shape) == [self.batch_size, self.size * self.size]
            x = self.linear(x)
            # Already converted!
            return torch.tanh(x)

    
    class Discriminator(nn.Module):
        def __init__(self, kdiscrim, dropout=0.3, size=4, linear_size=15):
            super(Discriminator, self).__init__()
            seq = []
            assert len(kdiscrim) > 1
            kdiscrim = kdiscrim.copy()
            k, s, p = kdiscrim.pop(0)
            total = pow(2, 6)
            self.size = size
            self.linear_size = linear_size
            self.linear = nn.Sequential(
                nn.Linear(self.linear_size, self.size*self.size, False)
            )
            seq.extend([nn.Conv2d(1, total, kernel_size=k, stride=s, padding=p, bias=False),
                    nn.SELU(),
                    nn.Dropout(dropout)]) 
            while len(kdiscrim) > 0:
                k, s, p = kdiscrim.pop(0)
                seq.extend([nn.Conv2d(total, total*2, kernel_size=k, stride=s, padding=p, bias=False),
                        nn.BatchNorm2d(total*2),
                        nn.SELU(),
                        nn.Dropout(dropout)]) 
                total *= 2
                print(total, k, s, p)
            self.seq = nn.Sequential(*seq)
            self.finalconv = nn.Linear(total, 1)
            self.flatten = nn.Flatten()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.linear(x)
            x = torch.reshape(x, (len(x), 1, self.size, self.size))
            x = self.seq(x)
            x = self.flatten(x)
            x = self.finalconv(x)
            return x
    return Generator, Discriminator

def train(js, dataset, epochs=200):
    Generator, Discriminator = import_model()
    import math
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .uniformgan_helper import get_convolutions, get_masks_from_sequence, apply_masks_from_sequence, compute_privacy_sigma
    from .uniformgan_helper import transform_dataset
    import numpy as np
    import json

    columns = list(dataset)
    noise_dim = js['noise_dim'] if "noise_dim" in js else 90
    device = js['device']
    batch_size = js['batch_size']
    additionals = js['additionals']
    clamp_value = additionals("clamp_value") or 0.01
    privacy_epsilon = additionals("privacy_epsilon") or 2
    privacy_delta = additionals("privacy_delta") or 1e-5
    privacy_mM = additionals("privacy_mM") or 0.001
    normalizer, dataset = transform_dataset(js, dataset)
    js["extra_pickle"] = normalizer

    size = math.ceil(math.sqrt(dataset.shape[1]))
    linear_size = dataset.shape[1]
    tensordataset = TensorDataset(torch.from_numpy(dataset.to_numpy()))
    loader = DataLoader(tensordataset, batch_size=js['batch_size'], shuffle=True, drop_last=True)
    kgen, kdis = get_convolutions(size)
    print("SIZE", size, "from", linear_size)
    print("GEN", kgen)
    print("DIS", kdis)
    generator = Generator(kgen, noise_dim=noise_dim, batch_size=js['batch_size'], size=size, linear_size=len(columns)).to(device)
    discriminator = Discriminator(kdis, size=size, linear_size=len(columns)).to(device)
    optimizerG = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
    optimizerD = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    
    if additionals("use_dp"):
        sigma = compute_privacy_sigma(privacy_epsilon, privacy_delta, privacy_mM)
        print("SIGMA", sigma)
        for parameter in discriminator.parameters():
            parameter.register_hook(
                lambda grad: grad + (1 / batch_size) * sigma
                * torch.randn(parameter.shape).to(device)
            )
    
    masks = None # mask is none for DENSE, otherwise Sparse
    for i in range(epochs):
        print("EPOCH", i)
        if i % 100 < 30 or i % 100 > 60:
            masks = None
        else:
            if not masks:
                masks = get_masks_from_sequence(discriminator.seq)
        id_ = 0
        for id_, data in enumerate(loader):
            real = data[0].to(device, dtype=torch.float)
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake = generator(noise)
            if masks is not None:
                apply_masks_from_sequence(discriminator.seq, masks)
            optimizerD.zero_grad()
            y_real = discriminator(real)
            y_fake = discriminator(fake)
            loss_d = -(y_real.mean() - y_fake.mean())
            total_loss = loss_d
            total_loss.backward()
            optimizerD.step()
            for p in discriminator.parameters():
                p.data.clamp_(-clamp_value, clamp_value)
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake = generator(noise)
            optimizerG.zero_grad()
            y_fake = discriminator(fake)
            loss_g = -(y_fake.mean())
            loss_g.backward(retain_graph=True)
            rand = np.random.choice(list(range(10, 30)))
            pure = torch.tensor([i/rand for i in range(rand+1)], dtype=torch.float).to(device)
            actual = pure * 2 - 1
            quant = torch.quantile(fake, pure, dim=0)
            half_check = torch.square(torch.transpose(quant, 0, 1) - actual).mean()
            half_check.backward()
            optimizerG.step()
            if((id_ + 1) % 100 == 0):
                print("epoch", i + 1, "step", id_ + 1, total_loss.item(), loss_g.item(), half_check.item())
        if (id_ > 0):
            print("epoch", i + 1, "step", id_ + 1, total_loss.item(), loss_g.item(), half_check.item()) # type:ignore
    return generator

def generate(js, model, sample_total=40000):
    import torch
    import pandas as pd
    import numpy as np
    from .uniformgan_helper import reverse_transform_column

    batch_size = js['batch_size']
    noise_dim = js['noise_dim'] if "noise_dim" in js else 90
    columns = js['columns']
    device = js['device']
    normalizer = js["extra_pickle"]
    
    sampled = []
    for i in range(sample_total//batch_size):
        noise = torch.randn(batch_size, noise_dim, device=device)
        ss = model(noise).detach().cpu().numpy()
        sampled.append(ss)
    sampled = np.concatenate(sampled, axis=0)
    sampled_transform = pd.DataFrame(sampled, columns=columns)
    sample_dataset = sampled_transform.copy()
    for i in range(len(columns)):
        normalize = normalizer[i]
        if (normalize['compute'] is not None):
            compute = normalize['compute']
            sample_dataset[columns[i]] = reverse_transform_column((sample_dataset[columns[i]] + 1)/2, compute)
        try:
            print(columns[i], f"({normalize['min']}, {normalize['max']}) => ({sample_dataset[columns[i]].min()}, {sample_dataset[columns[i]].max()})")
        except:
            print(columns[i], f"({normalize['min']}, {normalize['max']}) => ({sample_dataset[columns[i]].unique()})")
    return sample_dataset
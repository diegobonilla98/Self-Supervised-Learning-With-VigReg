import torch.nn
import numpy as np
import tqdm
from Models import Encoder, Projector
from LARS_Opt import LARS
from DogCatDataLoader import DogCats
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import math


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def adjust_learning_rate(optimizer, loader, step):
    max_steps = 100 * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = LEARNING_RATE * BATCH_SIZE / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def exclude_bias_and_norm(p):
    return p.ndim == 1


BATCH_SIZE = 700
LEARNING_RATE = 0.2
USE_CUDA = torch.cuda.is_available()
N_EPOCHS = 100
IMAGE_SIZE = (128, 128)

encoder = Encoder()
projector = Projector()
data_loader = DataLoader(DogCats(IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

optimizer = LARS(
    list(encoder.parameters()) + list(projector.parameters()),
    lr=0,
    weight_decay=1e-6,
    weight_decay_filter=exclude_bias_and_norm,
    lars_adaptation_filter=exclude_bias_and_norm,
)
# optimizer = AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=LEARNING_RATE)

if USE_CUDA:
    encoder = encoder.cuda()
    projector = projector.cuda()

for param in encoder.parameters():
    param.requires_grad = True
for param in projector.parameters():
    param.requires_grad = True

scaler = torch.cuda.amp.GradScaler()

for epoch in range(N_EPOCHS + 1):
    data_iter = iter(data_loader)
    i = 0
    epoch_losses = []
    with tqdm.tqdm(total=len(data_loader)) as pbar:
        while i < len(data_loader):
            adjust_learning_rate(optimizer, data_loader, i)
            s_image, s_augmented = next(data_iter)

            optimizer.zero_grad()

            if USE_CUDA:
                s_image = s_image.cuda()
                s_augmented = s_augmented.cuda()
            s_image_v = Variable(s_image)
            s_augmented_v = Variable(s_augmented)

            with torch.cuda.amp.autocast():
                z_a = projector(encoder(s_image_v))
                z_b = projector(encoder(s_augmented_v))

                sim_loss = F.mse_loss(z_a, z_b)

                std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
                std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
                std_loss = torch.mean(torch.relu(1 - std_z_a)) + torch.mean(torch.relu(1. - std_z_b))

                z_a = z_a - z_a.mean(dim=0)
                z_b = z_b - z_b.mean(dim=0)
                cov_z_a = (z_a.T @ z_a) / (BATCH_SIZE - 1.)
                cov_z_b = (z_b.T @ z_b) / (BATCH_SIZE - 1.)
                cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / 2048. + off_diagonal(cov_z_b).pow_(2).sum() / 2048.

                err = 25. * sim_loss + 25. * std_loss + 1. * cov_loss

            scaler.scale(err).backward()
            scaler.step(optimizer)
            scaler.update()

            # err.backward()
            # optimizer.step()

            i += 1

            epoch_losses.append(err.cpu().data.numpy())

            pbar.set_description(f"Iter: {i}/{len(data_loader)}, [Loss: {np.mean(epoch_losses)}]")
            pbar.update()

    print(f'[Epoch: {epoch}/{N_EPOCHS}, [Loss: {np.mean(epoch_losses)}]')
    torch.save(encoder, f'./checkpoints/epoch_{epoch}_encoder_loss_{np.mean(epoch_losses)}.pth')

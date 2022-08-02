import numpy as np
import os, pdb
import argparse
from dataset import Dataset
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from augmentations import *
import sys
import time
import imp
import os
import cv2
import random
import numpy as np
from functools import partial
from matplotlib import pyplot
from models import *
import functools
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torchvision.utils import make_grid
import wandb
from ignite.metrics import PSNR
from ignite.engine import Engine

#config
####################################
####################################
device = "cuda"
batch_size_train=10
####################################
####################################

#helper function
####################################
####################################
def eval_step(engine, batch): 
    return batch 
    
def calculate_error(img1,img2):
    img1 = (img1 * 10500.0) - 500.0 
    img2 = (img2 * 10500.0) - 500.0  
    default_evaluator = Engine(eval_step)
    psnr = PSNR(data_range=10500)
    psnr.attach(default_evaluator,'psnr')
    state1 = default_evaluator.run([[img1,img2]])
    return state1.metrics['psnr']
####################################
####################################

#dataset
####################################
####################################
train_lr_dataset = Dataset("")
train_hr_dataset = Dataset("")

def preprocess(train_files):
    npy_files = load_numpy_files(train_files)
    npy_files = torch.from_numpy(npy_files).permute(0, 3, 1, 2)
    return npy_files

train_preprocess_function = lambda images: preprocess(images)

train_lr_dataset.start_batch_queue(
        batch_size_train, batch_format='random_samples', proc_func=train_preprocess_function
)

train_hr_dataset.start_batch_queue(
        batch_size_train, batch_format='random_samples', proc_func=train_preprocess_function
)

generator = RRDBNet(1, 1, 64, 2)
generator = generator.to(device)
discriminator = Discriminator_VGG_128(1,64)
discriminator = discriminator.to(device)

optim_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)


num_iterations = len(train_lr_dataset.images) // batch_size_train

for epoch in range(1, 100):
    print("Training for epoch",epoch)
    generator.train()
    for batchIndex in range(num_iterations):
        for p in discriminator.parameters():
            p.requires_grad = False
        #training generator
        optim_G.zero_grad()
        batch_lr = train_lr_dataset.pop_batch_queue()
        batch_hr = train_hr_dataset.pop_batch_queue()

        lr_images = batch_lr['images'].to(device)
        hr_images = batch_hr['images'].to(device)
        
        predicted_hr_images = generator(lr_images)
        predicted_hr_labels = discriminator(predicted_hr_images)

        gf_loss = F.binary_cross_entropy_with_logits(predicted_hr_labels, torch.ones_like(predicted_hr_labels))
        gr_loss = F.l1_loss(predicted_hr_images, hr_images)
        g_loss = gf_loss + 100.0 * gr_loss
        wandb.log({"G Adversarial Loss":gf_loss.item()})
        wandb.log({"G Reconstruction Loss":gr_loss.item()})
        wandb.log({"G Loss Total":g_loss.item()})
        g_loss.backward()
        optim_G.step()
        #training discriminator
        for p in discriminator.parameters():
            p.requires_grad = True
        optim_D.zero_grad()
        predicted_hr_images = generator(lr_images).detach()
        adv_hr_real = discriminator(hr_images)
        adv_hr_fake = discriminator(predicted_hr_images)
        df_loss = F.binary_cross_entropy_with_logits(adv_hr_real, torch.ones_like(adv_hr_real)) + F.binary_cross_entropy_with_logits(adv_hr_fake, torch.zeros_like(adv_hr_fake))
        wandb.log({"D Adversarial Loss":df_loss.item()})
        df_loss.backward()
        optim_D.step()
        if batchIndex%5==0:
            grid1 = make_grid(lr_images[0:2])
            grid2 = make_grid(hr_images[0:2])
            grid3 = make_grid(predicted_hr_images[0:2])
            grid1 = wandb.Image(grid1, caption="Low Resolution Image")
            grid2 = wandb.Image(grid2, caption="High Resolution Image")
            grid3 = wandb.Image(grid3, caption="Reconstructed High Resolution Image")
            wandb.log({"Original": grid1})
            wandb.log({"Reconstruced": grid2})
            wandb.log({"Reconstruced": grid3})
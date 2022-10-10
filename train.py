import os
import os.path
from PIL import Image

import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math

import torchvision
from torchvision import transforms

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# model
from models.vgg import VGG


# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

torch.manual_seed(17)


# Dataset
train_mean_std = {'mean': [0.418, 0.435, 0.448], 'std': [0.24, 0.237, 0.234]}
test_mean_std = {'mean': [0.417, 0.433, 0.446], 'std': [0.237, 0.234, 0.231]}

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean_std['mean'], std=train_mean_std['std'])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=test_mean_std['mean'], std=test_mean_std['std'])
])

train_set = torchvision.datasets.ImageFolder(root="../data/refined_fer2013/train",
                                            transform=train_transform)
test_set = torchvision.datasets.ImageFolder(root="../data/refined_fer2013/test",
                                           transform=test_transform)

print("==> Success to load train set and test set")

# Dataloader
# batch size
batch_size = 32

train_loader = DataLoader(train_set,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(test_set,
                        batch_size=batch_size,
                        shuffle=True)


print("==> Success to load train loader and test loader")

# load model
model = VGG(vgg_name='VGG16', landmark_num=124).to(device)

print("==> Success to load model")


# set loss function and optimizer
num_epoch = 40
learning_rate = 0.1
criterion = nn.CrossEntropyLoss().to(device)
optimizer = "adam"

if optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),weight_decay = 1e-4)
elif optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum=0.9,
                                weight_decay = 1e-4)
else:
    raise ValueError("Optimizer not supported.")


# remove gpu cache 
import torch, gc
gc.collect()
torch.cuda.empty_cache()


### train
print("-----------------------------")
print("Start Training!")
print("-----------------------------")

loss_list = []

for epoch in tqdm(range(num_epoch), desc="epoch"):
    cost = 0.0
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)
        
        optimizer.zero_grad()
        attention_weights, weighted_prob, land_2d = model.forward(x)
        loss = criterion(weighted_prob, y_)
        
        loss.backward()
        optimizer.step()
        
        cost += loss.item()
        
        
    batch_loss = cost / len(train_loader)
    loss_list.append(batch_loss)
    print(f'[{epoch+1}] loss: {batch_loss:.3f}')


# visualize loss list
plt.plot(loss_list)
plt.show()


# save model
torch.save(model.state_dict(), './model/refined_fer_vggnet.pt')

print("-----------------------------")
print("End Training!")
print("-----------------------------")
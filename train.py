import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import optimizer
from torch.optim import Adam
from mnist_svd import MnistSVD
from unet import *

# dataloader
mnist_train = MnistSVD("data", train_preprocess = True)
train_dataloader = DataLoader(mnist_train, batch_size = 64, shuffle = True, num_workers=12,pin_memory=True)

lr = 2e-5
net = UNet_MLP(input_dim = 28*28, time_emb_dim = 512, ranks = 28, scale = 12, block_layer = 2).cuda()

optimizer = Adam(net.parameters(), lr = lr)

ema_decay = 0.995
ema_net = UNet_MLP(input_dim = 28*28, time_emb_dim = 512, ranks = 28, scale = 12, block_layer = 2).cuda()
ema_net.load_state_dict(net.state_dict())

epoch = 1000
total_loss_frac = 0
total_loss = 0
steps = 0
for ep in range(epoch):
    for inp, out, rank_list, _ in train_dataloader:
        steps += 1
        optimizer.zero_grad()
        est_out = net(inp.cuda(),rank_list.cuda())
        loss = torch.sum(torch.pow(out.cuda() - est_out,2)) / len(rank_list) 
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
        loss_frac = loss.item() * len(rank_list) / torch.sum(torch.pow(out.cuda(),2)).item()
        total_loss_frac += loss_frac

        if steps % 5 == 0:
            for param, ema_param in zip(net.parameters(), ema_net.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)


        if steps % 100 == 0:
            print(f"epoch:{ep},steps:{steps},avg_loss:{total_loss/steps},loss:{loss.item()},avg_loss_frac:{total_loss_frac/steps},loss_frac:{loss_frac}")

        if steps % 1000 == 0:
            torch.save(net.state_dict(), "model.pth")
            torch.save(ema_net.state_dict(), "ema_model.pth")

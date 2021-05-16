# -*- coding: UTF-8 -*-
"""
@Project ：srgan-esrgan-pytorch
@File    ：train_esrgan.py
@IDE     ：PyCharm
@Author  ：AmazingPeterZhu
@Date    ：2021/5/11 下午4:00
"""
import os
import time
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import yaml
import matplotlib.pyplot as plt
from tqdm import trange
import wandb

from models import RRDBNet
from utils import init_weights_mini
from datasets import DIV2KDataset


# 读取配置文件
fs = open("options/rrdbnet_psnr/train.yaml", encoding="UTF-8")
opt = yaml.load(fs, Loader=yaml.FullLoader)

# 创建输出文件夹
# 用于保存模型和断点
# 文件夹名字为配置文件中设置的name
try:
    os.makedirs(osp.join('experiments', opt['name'], 'models'))
    os.makedirs(osp.join('experiments', opt['name'], 'checkpoints'))
except OSError:
    pass

# 读取dataset
rootGT = opt['datasets']['train']['dataroot_gt']
rootLQ = opt['datasets']['train']['dataroot_lq']
crop_size = opt['datasets']['train']['crop_size']
dataset = DIV2KDataset(rootGT=rootGT,
                       rootLQ=rootLQ,
                       RandomCrop=True,
                       ToTensor=True,
                       crop_size=128)

# 创建dataloader
dataloader = torch.utils.data.DataLoader(dataset,
                                         opt['batch_size'],
                                         shuffle=True,
                                         num_workers=opt['workers'])

# 设置运行设备
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# 定义损失函数
criterion = nn.L1Loss().to(device)

# 网络实例化
net = RRDBNet(opt['in_channels'], opt['num_features'],
              opt['num_blocks'], opt['num_grow']).to(device)

# 网络初始化权重参数
net.apply(init_weights_mini)

# 设定优化器
optimizer = optim.Adam(net.parameters(), lr=opt['lr'])

# 设定学习率变化调度

# wandb
# 1. Start a new run
wandb.init(project=opt['name'])

# 2. Save model inputs and hyperparameters
config = wandb.config
config.batch_size = opt['batch_size']
config.epochs = opt['total_iter']
config.lr = opt['lr']
config.log_interval = 10
config.RandomCrop = True

# 3. Log gradients and model parameters
wandb.watch(net)

# 开始训练
# 将模型设置为训练模式
net.train()

# 检查是否断点续训
if opt['checkpoint']['resume']:
    start_epoch = opt['checkpoint']['start_epoch']
    checkpoint_path = osp.join(opt['outroot'],
                               opt['name'],
                               'checkpoints',
                               'checkpoint_e{:03d}'.format(start_epoch-1))
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses = checkpoint['losses']
else:
    start_epoch = 0
    checkpoint = {}
    losses = []

# 计时
T1 = time.perf_counter()

print("Starting Training Loop...")
for epoch in trange(start_epoch, opt['total_iter']):
    for i, img_dic in enumerate(dataloader):
        lq = img_dic['lq'].to(device)
        gt = img_dic['gt'].to(device)
        optimizer.zero_grad()
        fake = net(lq)
        loss = criterion(fake, gt)
        loss.backward()
        optimizer.step()

        # Output training stats
        if (i+1) % 100 == 0:
            T2 = time.perf_counter()
            print('Epoch: [%d/%d]\tmini-Batch: [%d/%d]\tLoss: %.4f\t已运行时间:%.2f分钟'
                  % (epoch, opt['total_iter'], i, len(dataloader),
                     loss.item(), (T2 - T1)/60))

        # Save Losses for plotting later
        losses.append(loss.item())
        # wandb
        # 4. Log metrics to visualize performance
        if (i+1) % 10 == 0:
            wandb.log({"loss": loss})

    # 保存断点
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }
    checkpoint_path = osp.join(opt['outroot'], opt['name'],
                               'checkpoints', 'checkpoint_e{:03d}'.format(epoch))
    torch.save(checkpoint, checkpoint_path)
    wandb.save('net_e{:03d}'.format(epoch))
    net.save(os.path.join(wandb.run.dir, 'net_e{:03d}'.format(epoch)))


# 展示损失图
plt.figure(figsize=(10, 5))
plt.title("Loss During Training")
plt.plot(losses, label="losses")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 保存模型
torch.save(net.state_dict(), osp.join(opt['outroot'], opt['name'], 'models', 'RRDBNet_final.pth'))


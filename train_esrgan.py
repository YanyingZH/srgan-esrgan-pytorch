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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import yaml
import matplotlib.pyplot as plt
from tqdm import trange
import wandb

from models import RRDBNet, RaDiscriminator
from utils import FeatureExtractor, init_weights, init_weights_mini
from datasets import DIV2KDataset


# 读取配置文件
fs = open("options/esrgan/train.yaml", encoding="UTF-8")
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
                       RandomHorizontalFlip=True,
                       RandomRotation90Degree=True,
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
percep_criterion = nn.L1Loss().to(device)
adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
content_criterion = nn.L1Loss().to(device)

# 网络实例化
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True), feature_layer=35).to(device)
netG = RRDBNet(opt['in_channels'], opt['ngf'], opt['num_blocks']).to(device)
netD = RaDiscriminator(opt['in_channels'], opt['ndf']).to(device)

# 网络初始化权重参数
# netG.load_state_dict(osp.join('experiments', 'pretrained_models', 'RRDBNet_final.pth'))
netD.apply(init_weights)
netG.apply(init_weights_mini)

# 设定优化器
optimizerG = optim.Adam(netG.parameters(), lr=opt['lr'])
optimizerD = optim.Adam(netD.parameters(), lr=opt['lr'])

# 设定学习率变化调度
schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG,
                                            milestones=opt['scheduler']['milestones'],
                                            gamma=opt['scheduler']['gamma'])

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
wandb.watch(netG)
wandb.watch(netD)

# 开始训练
# 将模型设置为训练/测试模式
netG.train()
netD.train()
feature_extractor.eval()

# 检查是否断点续训
if opt['checkpoint']['resume']:
    start_epoch = opt['checkpoint']['start_epoch']
    checkpoint_path = osp.join('experiments',
                               opt['name'],
                               'checkpoints',
                               'checkpoint_e{:03d}'.format(start_epoch-1))
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    G_losses = checkpoint['G_losses']
    schedulerG = checkpoint['schedulerG_state_dict']
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    D_losses = checkpoint['D_losses']
else:
    start_epoch = 0
    checkpoint = {}
    G_losses = []
    D_losses = []

# 计时
T1 = time.perf_counter()

print("Starting Training Loop...")
for epoch in trange(opt['total_iter']):
    for i, img_dic in enumerate(dataloader):

        # 配置数据
        gt = img_dic['gt'].to(device)
        lq = img_dic['lq'].to(device)

        real_labels = torch.ones((gt.size(0), 1), requires_grad=False).to(device)
        fake_labels = torch.zeros((gt.size(0), 1), requires_grad=False).to(device)

        ############################
        # (1) Update G network
        ###########################
        optimizerG.zero_grad()

        # 生成fake高清图片
        fake = netG(lq)

        # 鉴别器评分
        score_real = netD(gt).detach()
        score_fake = netD(fake)

        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()


        # 损失
        content_loss = content_criterion(fake, gt)

        percep_loss = percep_criterion(feature_extractor(fake), feature_extractor(gt))

        adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
        adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
        adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

        errG = percep_loss + 0.005*adversarial_loss + 0.01*content_loss
        # 损失反向传播
        errG.backward()
        # 更新G
        optimizerG.step()

        ############################
        # (2) Update D network
        ###########################
        optimizerD.zero_grad()

        # 鉴别器评分
        score_real = netD(gt)
        score_fake = netD(fake).detach()

        discriminator_rf = score_real - score_fake.mean()
        discriminator_fr = score_fake - score_real.mean()

        # 损失
        adversarial_loss_rf = adversarial_criterion(discriminator_rf, real_labels)
        adversarial_loss_fr = adversarial_criterion(discriminator_fr, fake_labels)
        errD = (adversarial_loss_fr + adversarial_loss_rf) / 2

        # 损失反向传播
        errD.backward()
        # 更新G
        optimizerD.step()

        # Output training stats
        if (i+1) % 1000 == 0:
            T2 = time.perf_counter()
            print('epoch:[%d/%d]step:[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f\t已运行时间:%.2f分钟'
                  % (epoch, opt['total_iter'], i, len(dataloader),
                     errD.item(), errG.item(),
                     torch.sigmoid(score_real).mean().item(),
                     torch.sigmoid(score_fake).mean().item(),
                     (T2 - T1)/60))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        # wandb
        # 4. Log metrics to visualize performance
        if (i+1) % 10 == 0:
            wandb.log({"G_loss": errG})
            wandb.log({"D_loss": errD})

    # 学习率变化
    schedulerG.step()

    # 保存断点
    checkpoint = {
        'netG_state_dict': netG.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'G_losses': G_losses,
        'schedulerG_state_dict': schedulerG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'D_losses': D_losses
    }
    checkpoint_path = osp.join('experiments', opt['name'],
                               'checkpoints', 'checkpoint_e{:03d}'.format(epoch))
    torch.save(checkpoint, checkpoint_path)
    netG_name = 'netG_e{:03d}'.format(epoch)
    netD_name = 'netD_e{:03d}'.format(epoch)
    wandb.save(netG_name)
    torch.save(netG.state_dict(), osp.join(wandb.run.dir, netG_name))
    wandb.save(netD_name)
    torch.save(netD.state_dict(), osp.join(wandb.run.dir, netD_name))

# 展示损失图
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 保存模型
torch.save(netG.state_dict(), osp.join('experiments', opt['name'], 'models', 'netG_final.pth'))
torch.save(netD.state_dict(), osp.join('experiments', opt['name'], 'models', 'netD_final.pth'))

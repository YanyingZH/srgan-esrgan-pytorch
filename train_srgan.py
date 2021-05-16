# -*- coding: UTF-8 -*-
"""
@Project ：srgan-esrgan-pytorch
@File    ：train_srgan.py
@IDE     ：PyCharm 
@Author  ：AmazingPeterZhu
@Date    ：2021/4/23 下午3:18 
"""
import os
import time
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import yaml
import matplotlib.pyplot as plt
from tqdm import trange
import wandb

from models import SRResNet, Discriminator
from utils import FeatureExtractor, init_weights
from datasets import DIV2KDataset


# 读取配置文件
fs = open("options/srgan/train.yaml", encoding="UTF-8")
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
dataset = DIV2KDataset(rootGT=rootGT,
                       rootLQ=rootLQ,
                       transformGT=transforms.Compose([
                           transforms.ToTensor()
                       ]),
                       transformLQ=transforms.Compose([
                           transforms.ToTensor()
                       ]))

# 创建dataloader
dataloader = torch.utils.data.DataLoader(dataset,
                                         opt['batch_size'],
                                         shuffle=True,
                                         num_workers=opt['workers'])

# 设置运行设备
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# 定义损失函数
content_criterion = nn.MSELoss().to(device)
adversarial_criterion = nn.BCELoss().to(device)

# 网络实例化
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True), feature_layer=12).to(device)
netG = SRResNet(opt['in_channels'], opt['ngf'], opt['num_blocks']).to(device)
netD = Discriminator(opt['in_channels'], opt['ndf']).to(device)

# 网络初始化权重参数
netG.apply(init_weights)
netD.apply(init_weights)

# 设定优化器
optimizerG = optim.Adam(netG.parameters(), lr=opt['lr'])
optimizerD = optim.Adam(netD.parameters(), lr=opt['lr'])

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
    checkpoint_path = osp.join(opt['outroot'],
                               opt['name'],
                               'checkpoints',
                               'checkpoint_e{:03d}'.format(start_epoch-1))
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    G_losses = checkpoint['G_losses']
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    D_losses = checkpoint['D_losses']
else:
    start_epoch = 0
    checkpoint = {}
    G_losses = []
    D_losses = []


real_label = 1.
fake_label = 0.

# 计时
T1 = time.perf_counter()

print("Starting Training Loop...")
for epoch in trange(opt['total_iter']):
    for i, img_dic in enumerate(dataloader):

        ############################
        # (1) Update D network
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_hr = img_dic['gt'].to(device)
        b_size = real_hr.size(0) # size(0)=size()[0]=128
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_hr).view(-1)
        # Calculate loss on all-real batch
        errD_real = adversarial_criterion(output, label) #-log(D(x))
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item() # 对一个tensor求均值并将这个tensor转换成python中的float数值

        ## Train with all-fake batch
        # Generate batch of low quality image, not latent vectors
        lq = img_dic['lq'].to(device)
        # Generate fake image batch with G
        fake = netG(lq).to(device)
        label.fill_(fake_label)
        # Classify all fake batch with D
        # tensor.detach()的功能是将一个张量从graph中剥离出来，不用计算梯度
        # 单独的训练D，G网络，所以在D，G网络之间有数据传递的时间要用.detach()
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = adversarial_criterion(output, label) #-log(1 - D(G(z)))
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_lq1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Feature map extracted from VGG19 net
        fake_features = feature_extractor(fake)
        real_features = feature_extractor(real_hr)
        # Calculate G's loss based on this output
        content_loss = content_criterion(fake, real_hr) + 0.006 * content_criterion(fake_features, real_features)
        adversarial_loss = adversarial_criterion(output, label)
        errG = content_loss + 1e-3 * adversarial_loss
        # Calculate gradients for G
        errG.backward()
        D_G_lq2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if (i+1) % 100 == 0:
            T2 = time.perf_counter()
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f 程序运行时间:%.2f分钟'
                  % (epoch, opt['total_iter'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_lq1, D_G_lq2, (T2 - T1)/60))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        # wandb
        # 4. Log metrics to visualize performance
        if (i+1) % 10 == 0:
            wandb.log({"G_loss": errG})
            wandb.log({"D_loss": errD})

    # 保存断点
    checkpoint = {
        'netG_state_dict': netG.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'G_losses': G_losses,
        'netD_state_dict': netD.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'D_losses': D_losses
    }
    checkpoint_path = osp.join(opt['outroot'], opt['name'],
                               'checkpoints', 'checkpoint_e{:03d}'.format(epoch))
    torch.save(checkpoint, checkpoint_path)
    wandb.save('netG_e{:03d}'.format(epoch))
    wandb.save('netD_e{:03d}'.format(epoch))
    netG.save(os.path.join(wandb.run.dir, 'netG_e{:03d}'.format(epoch)))
    netD.save(os.path.join(wandb.run.dir, 'netD_e{:03d}'.format(epoch)))

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
torch.save(netG.state_dict(), osp.join(opt['outroot'], opt['name'], 'models', 'netG_final.pth'))
torch.save(netD.state_dict(), osp.join(opt['outroot'], opt['name'], 'models', 'netD_final.pth'))

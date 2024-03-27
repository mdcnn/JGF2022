from __future__ import print_function
import argparse
from math import log10
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import pdb
import socket
import time
import scipy.io as scio


from JGF_x8 import Net as JGF_x8_Net


os.environ["CUDA_VISIBLE_DEVICES"]='0'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='LR. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=10, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=float, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/data1/Color-Depth-DATA/wangke-train2021-0505/')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='ptrain1000nyu-depthHR/')
parser.add_argument('--rgb_train_dataset', type=str, default='ptrain1000nyu-color/')
parser.add_argument('--train_dataset', type=str, default='ptrain1000nyu-depthLR8/')
parser.add_argument('--model_type', type=str, default='JGF_x8_Net')
parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped HR image')
parser.add_argument('--save_folder', default='./weights/x8/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='dbpn', help='Location to save checkpoint models')
opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
print(opt)

def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):

        input_rgb, input, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input_rgb = input_rgb.cuda()
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        t0 = time.time()
        pre=model(input_rgb, input)
        loss = criterion(pre, target)
        t1 = time.time()
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        avg_loss = epoch_loss / len(training_data_loader)
        loss_list.append(avg_loss)

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.train_dataset+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)


train_set = get_training_set(opt.data_dir, opt.train_dataset, opt.hr_train_dataset, opt.rgb_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
model = JGF_x8_Net(num_channels=1, base_filter=64,  feat = 256)
criterion = nn.L1Loss()

###############
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)


loss_list = []
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)

    if (epoch+1) % 10 == 0:
        plt.plot(loss_list, linewidth=5)
        plt.title('Loss Table', fontsize=14)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.savefig('JGFX8.png')
        scio.savemat('loss.mat', {'loss_list': loss_list,'epoch':epoch})

    if (epoch+1) == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)



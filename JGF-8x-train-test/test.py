from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_eval_set
from functools import reduce
import scipy.io as sio
import time
import cv2

from JGF_x8 import Net as PMBAX8

os.environ["CUDA_VISIBLE_DEVICES"]='0'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=float, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='/data1/Color-Depth-DATA/wangke-test2021-0505/')
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='TESTING449nyu-depthLR8/')
parser.add_argument('--test_rgb_dataset', type=str, default='TESTING449nyu-color')
parser.add_argument('--model_type', type=str, default='PMBAX8')
parser.add_argument('--model', default="./weights/x8/ptrain1000nyu-depthLR8/JGF_x8_Net.pth", help='sr pretrained base model')

opt = parser.parse_args()
gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset),os.path.join(opt.input_dir,opt.test_rgb_dataset))
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.model_type == 'PMBAX8':
    model = PMBAX8(num_channels=1, base_filter=64,  feat = 256)
else:
    model = PMBAX8(num_channels=1, base_filter=64,  feat = 256)

if os.path.exists(opt.model):
    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.<---------------------------->')

if cuda:
    model = model.cuda()

def eval():
    model.eval()
    torch.set_grad_enabled(False)
    for batch in testing_data_loader:
        input,input_rgb, name = Variable(batch[0],volatile=True),Variable(batch[1],volatile=True), batch[2]
        if cuda:
            input = input.cuda()
            input_rgb = input_rgb.cuda()

        input_rgb1=torch.zeros(1, 3, 640, 640).cuda()
        input_rgb1[:, :, 0:480, :]=input_rgb
        input1=torch.zeros(1, 1, 80, 80).cuda()
        input1[:, :, 0:60, :] = input
        prediction = model(input_rgb1, input1)
        prediction=prediction[:,:,0:480,:]

        save_img(prediction.cpu().data, name[0])

def save_img(img, img_name):

    save_img = img.squeeze().clamp(0, 1).numpy()
    import numpy as np
    print(np.min(save_img*255.0))
    print(np.max(save_img*255.0))
    save_dir=os.path.join(opt.output,opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    cv2.imwrite(save_fn,save_img*255)

###Start TO Eval !!!!
eval()

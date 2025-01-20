from PIL import Image
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from glob import glob
import math
import cv2
import os
from six.moves import xrange
from PIL import Image
import ssim_matlab

# dataset_pre = './x4_result_depth_pre0805'0.270.36
# #dataset_pre = './duan_f22'
# dataset_gt = './x4_gt_depth_0805'
# #dataset_gt = './duan_gt2'
dataset_pre = './Results/testx16/'
dataset_gt = './data/Dtest/'

data1 = sorted(glob(os.path.join(
    dataset_pre, "*.png")))
data2 = sorted(glob(os.path.join(
    dataset_gt, "*.png")))
mad_list = []
rmse_list = []
psnr_list = []
ssim_list = []
rmse2_list = []
ssim_list_sk = []
psnr_list_sk = []
mse_list_sk = []
rmse_list_sk = []

# psnr2_list = []
for filename in data1:
    for filename2 in data2:

        i = filename.split('/')[-1].split('.')[0]
        j = filename2.split('/')[-1].split('.')[0]

        if i == j:
            print(i)
            img_pre = Image.open(filename)
            img_gt = Image.open(filename2)
            img_pre = cv2.imread(filename, 0)
            img_gt = cv2.imread(filename2, 0)

            ssim_list_sk.append(compare_ssim(img_pre, img_gt))
            psnr_list_sk.append(compare_psnr(img_pre, img_gt))
            mse_list_sk.append(compare_mse(img_pre, img_gt))
            rmse_list_sk.append(math.sqrt(compare_mse(img_pre, img_gt)))
            ###############ssim

            img_pre = img_pre.astype(np.double)
            img_gt = img_gt.astype(np.double)
            diff = img_gt - img_pre

            h, w = diff.shape
            mad = np.sum(abs(diff)) / (h * w)
            rmse = np.sqrt(np.sum(np.power(diff, 2) / (h * w)))

            psnr = 20 * math.log10(255.0 / rmse)
            # mse = np.mean((img_gt/1. - img_pre/1.) ** 2 )
            # psnr = 10 * math.log10(255.0*255.0/mse)
            # rmse2 = np.sqrt(mse)

            # img_gt = img_gt * 255.0
            # img_pre = img_pre * 255.0
            # img_gt_u = np.mean(img_gt)
            # img_pre_u = np.mean(img_pre)
            # img_gt_var = np.var(img_gt)
            # img_pre_var = np.var(img_pre)
            # img_gt_std = np.sqrt(img_gt_var)
            # img_pre_std = np.sqrt(img_pre_var)
            # c1 = np.square(0.01*7)
            # c2 = np.square(0.03*7)
            # ssim_0 = (2 * img_gt_u * img_pre_u + c1) * (2 * img_gt_std * img_pre_std + c2)
            # denom = (img_gt_u ** 2 + img_pre_u **2 +c1) * (img_gt_var + img_pre_var +c2)
            # ssim = ssim_0 / denom
            ssim_Matlab = ssim_matlab.ssim(img_gt, img_pre)

            #mad_list.append(mad)
            # rmse_list.append(rmse)
            # psnr_list.append(psnr)
            # ssim_list.append(ssim_Matlab)
            print('mad: ', mad)
            # rmse2_list.append(rmse2)
            # psnr2_list.append(psnr)

# print('mad: ', mad_list)
# print('mse_sk: ', mse_list_sk)
# print('rmse: ', rmse_list)
# print('rmse_sk: ', rmse_list_sk)
# print('psnr: ', psnr_list)
# print('psnr_sk: ', psnr_list_sk)
# print('ssim: ', ssim_list)
# print('ssim_sk: ', ssim_list_sk)
# print('rmse2: ', rmse2_list)
# print('psnr2: ', psnr2_list)



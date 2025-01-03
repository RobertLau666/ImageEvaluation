import cv2
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
import numpy as np
import config
from metrics.norm import *


def calculate_saturation_score(img_numpy):
    hsv_image = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_image[:, :, 1]
    saturation_score = np.mean(saturation_channel)
    saturation_score_normed = saturation_norm(saturation_score)
    return saturation_score, saturation_score_normed

PSNR_reference_image = cv2.resize(cv2.imread(config.metric_params["PSNR"]["PSNR_reference_image_path"]), (299, 299))
def calculate_PSNR_score(img_numpy):
    img_numpy = cv2.resize(img_numpy, (PSNR_reference_image.shape[1], PSNR_reference_image.shape[0]))
    PSNR_score = cv2.PSNR(PSNR_reference_image, img_numpy)
    PSNR_score_normed = PSNR_norm(PSNR_score)
    return PSNR_score, PSNR_score_normed

SSIM_reference_image = cv2.resize(cv2.imread(config.metric_params["SSIM"]["SSIM_reference_image_path"]), (299, 299))
def calculate_SSIM_score(img_numpy):
    img_numpy = cv2.resize(img_numpy, (SSIM_reference_image.shape[1], SSIM_reference_image.shape[0]))
    SSIM_score = ssim(SSIM_reference_image, img_numpy, win_size=3, data_range=255, multichannel=True)
    SSIM_score_normed = SSIM_norm(SSIM_score)
    return SSIM_score, SSIM_score_normed

# def calculate_FID_score(real_images_folder, generated_images_folder):
#     '''
#     FID（Frechet Inception Distance）分数是一种用于衡量生成模型与真实数据集之间相似性的指标，它是通过计算生成的样本与真实样本在Inception网络中特征表示上的差异程度来计算得出的。FID分数越低，表示生成的样本与真实样本之间的差异越小，生成模型的性能越好。
#     '''
#     # inception_model = torchvision.models.inception_v3(pretrained=True)
#     FID_ = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], # (0~INF)
#                                                     batch_size=1,
#                                                     device='cpu',
#                                                     dims=2048, num_workers=0,
#                                                     )
#     # FID_score = 1 - FID_ / 1000.0 # (0~1)
#     FID_score = FID_
#     # print(f'FID_score: {FID_score}')

#     return FID_score

def calculate_variance_score(img_numpy):
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
    variance = cv2.meanStdDev(img_numpy)[1]
    variance_score = variance[0][0]
    variance_score_normed = variance_norm(variance_score)
    return variance_score, variance_score_normed
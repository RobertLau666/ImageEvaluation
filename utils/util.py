import os
import cv2
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from tqdm import tqdm
import numpy as np


def calculate_average_saturation(image_folder):
    """
    计算给定文件夹中所有图像的平均色彩饱和度
    :param image_folder: 图像文件夹路径
    :return: 平均饱和度值
    """
    total_saturation = 0
    image_count = 0
    
    # 遍历文件夹中的所有图像文件
    for filename in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, filename)
        
        # 只处理图像文件
        if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 读取图像
            image = cv2.imread(image_path)
            
            if image is not None:
                # 将图像从BGR转换为HSV
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # 提取饱和度通道
                saturation_channel = hsv_image[:, :, 1]
                
                # 计算该图像的平均饱和度
                mean_saturation = np.mean(saturation_channel)
                total_saturation += mean_saturation
                image_count += 1
    
    # 计算所有图像的平均饱和度
    if image_count > 0:
        average_saturation = total_saturation / image_count
        return average_saturation
    else:
        return 0  # 如果没有图像，返回0


def calculate_average_PSNR_SSIM_score(generated_images_folder, reference_image_path):
    PSNR_score_list = []
    SSIM_score_list = []

    image_list = []
    for root, directories, files in os.walk(generated_images_folder):
        for file in files:
            file_path = os.path.join(root, file)
            image_list.append(file_path)

    # 标准PSNR评分
    psnr_standard = {
        "excellent": 1.0,
        "good": 0.8,
        "bad": 0.5,
        "unacceptable": 0.2
    }

    # real_images_folder = "/maindata/data/shared/public/chenyu.liu/Trash/ref_images"
    # 读取参考图像
    reference_image = cv2.imread(reference_image_path)
    reference_image = cv2.resize(reference_image, (299, 299))
    # 如果参考图像读取失败，退出
    if reference_image is None:
        print("Error: Failed to load reference image.")

    # 遍历所有测试图像
    for image_path in tqdm(image_list[1:]):
        test_image = cv2.imread(image_path)

        # 如果图像读取失败，跳过
        if test_image is None:
            print(f"Warning: Failed to load image {image_path}. Skipping...")
            continue

        # 检查图像尺寸是否一致
        if reference_image.shape != test_image.shape:
            print(f"Warning: Image {image_path} has a different size. Resizing it to match the reference image.")
            test_image = cv2.resize(test_image, (reference_image.shape[1], reference_image.shape[0]))

        # 计算PSNR
        psnr_ = cv2.PSNR(reference_image, test_image)
        psnr_score = psnr_
        # if psnr_ > 40:
        #     psnr_score = psnr_standard["excellent"]
        # elif 40 >= psnr_ > 30:
        #     psnr_score = psnr_standard["good"]
        # elif 30 >= psnr_ > 20:
        #     psnr_score = psnr_standard["bad"]
        # elif 20 >= psnr_:
        #     psnr_score = psnr_standard["unacceptable"]
        PSNR_score_list.append(psnr_score)

        # 计算SSIM
        ssim_ = ssim(reference_image, test_image, win_size=3, data_range=255, multichannel=True)
        # ssim_score = (ssim_ + 1) / 2  # SSIM范围是[-1, 1]，转换为[0, 1]
        ssim_score = ssim_
        SSIM_score_list.append(ssim_score)

    average_PSNR_score = sum(PSNR_score_list) / len(PSNR_score_list)
    average_SSIM_score = sum(SSIM_score_list) / len(SSIM_score_list)
    # print(f'PSNR_score: {PSNR_score}\nSSIM_score: {SSIM_score}')

    return average_PSNR_score, average_SSIM_score


def calculate_FID_score(real_images_folder, generated_images_folder):
    '''
    FID（Frechet Inception Distance）分数是一种用于衡量生成模型与真实数据集之间相似性的指标，它是通过计算生成的样本与真实样本在Inception网络中特征表示上的差异程度来计算得出的。FID分数越低，表示生成的样本与真实样本之间的差异越小，生成模型的性能越好。
    '''
    # inception_model = torchvision.models.inception_v3(pretrained=True)
    FID_ = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], # (0~INF)
                                                    batch_size=1,
                                                    device='cpu',
                                                    dims=2048, num_workers=0,
                                                    )
    # FID_score = 1 - FID_ / 1000.0 # (0~1)
    FID_score = FID_
    # print(f'FID_score: {FID_score}')

    return FID_score

def calculate_variance_score(generated_images_folder):
    variance_score_list = []

    image_list = []
    for root, directories, files in os.walk(generated_images_folder):
        for file in files:
            file_path = os.path.join(root, file)
            image_list.append(file_path)

    for image in image_list:
        test_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        variance = cv2.meanStdDev(test_image)[1] # (0～INF)
        variance = variance[0][0]
        variance_score = 1 - variance / 1000.0
        variance_score_list.append(variance_score)
    average_variance_score = sum(variance_score_list) / len(variance_score_list) # (0~1)
    # print(f'average_variance_score: {average_variance_score}')

    return average_variance_score

def resize_images_in_folder(generated_images_folder, resized_folder, target_size=(299, 299)):
    # 创建新的文件夹，后缀加上_resized
    
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)

    # 遍历原文件夹中的图片
    for filename in tqdm(os.listdir(generated_images_folder)):
        file_path = os.path.join(generated_images_folder, filename)
        
        # 判断文件是否为图片
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            # 读取图片
            img = cv2.imread(file_path)
            
            # Resize图片
            img_resized = cv2.resize(img, target_size)

            # 生成保存图片的新路径
            resized_file_path = os.path.join(resized_folder, filename)

            # 保存resize后的图片到新文件夹
            cv2.imwrite(resized_file_path, img_resized)

    print(f"Resized images are saved to: {resized_folder}")
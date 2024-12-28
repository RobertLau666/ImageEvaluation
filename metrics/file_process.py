import numpy as np
import csv
import requests
import pandas as pd
import cv2
import imageio
import torchvision
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import re
import os
from datetime import datetime
from PIL import Image
from io import BytesIO


# def resize_images_in_folder(generated_images_folder, resized_folder, target_size=(299, 299)):
#     # 创建新的文件夹，后缀加上_resized
#         resized_images_folder = generated_images_folder + "_resized"
#     if (not os.path.exists(resized_images_folder)) or len(os.listdir(resized_images_folder)) <= len(os.listdir(generated_images_folder)):
    
#     if not os.path.exists(resized_folder):
#         os.makedirs(resized_folder)

#     # 遍历原文件夹中的图片
#     for filename in tqdm(os.listdir(generated_images_folder)):
#         file_path = os.path.join(generated_images_folder, filename)
        
#         # 判断文件是否为图片
#         if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
#             # 读取图片
#             img = cv2.imread(file_path)
            
#             # Resize图片
#             img_resized = cv2.resize(img, target_size)

#             # 生成保存图片的新路径
#             resized_file_path = os.path.join(resized_folder, filename)

#             # 保存resize后的图片到新文件夹
#             cv2.imwrite(resized_file_path, img_resized)

#     print(f"Resized images are saved to: {resized_folder}")


video_suffix = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
img_suffix = ['.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tif', '.webp']

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)  # 流式下载
        response.raise_for_status()  # 检查请求是否成功
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):  # 分块写入
                file.write(chunk)
        print(f"文件已成功下载到: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")

def log_csv(img_results, save_csv_path):
    headers = ["url", "predict", "pro0", "pro1", "pro2"]

    with open(save_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # 写入列名（标题行）
        writer.writerow(headers)

        for data in img_results:
            writer.writerow(data)

    print(f"数据已写入 CSV 文件: {save_csv_path}")


def read_excel(excel_path, begin_r, end_r, url_c):
    if excel_path.endswith('.csv'):
        data = pd.read_csv(excel_path)
    elif excel_path.endswith('.xlsx'):
        data = pd.read_excel(excel_path)
    num_rows = len(data)

    first_column = data.iloc[:, url_c]
    first_url_list = first_column.tolist()[begin_r:(num_rows if end_r == -1 else end_r)]

    return first_url_list

def get_img_urls(all_files):
    img_urls = []
    for index, all_file in enumerate(tqdm(all_files)):
        all_file_dict = json.loads(all_file)
        push_data = all_file_dict["push_data"]
        img_url = ''
        if "img_url" in push_data:
            img_url = push_data["img_url"]
        elif "images" in push_data:
            img_url = push_data["images"][0]
        img_urls.append(img_url)
    return img_urls

def download_img(url, timeout=5, retry_count=3):
    img = None
    for _ in range(retry_count):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # 检查HTTP请求是否成功
            image_array = np.asarray(bytearray(response.content), dtype="uint8")
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            break
        except Exception as e:
            pass
    
    if img is None:
        print(f'[ERROR] url: {url}')
    # return url, img
    return img

def get_image_array_from_img_url(img_url, timeout=5, retry_count=3):
    image_array = None
    for _ in range(retry_count):
        try:
            response = requests.get(img_url, timeout=timeout)
            image_pil = Image.open(BytesIO(response.content))
            image_array = np.array(image_pil)
            break
        except Exception as e:
            pass
    return image_array


def is_url(string):
    url_pattern = re.compile(r'^(https?://|ftp://|file://)?[a-zA-Z0-9.-]+(\.[a-zA-Z]{2,})+(:\d+)?(/.*)?$')
    return bool(url_pattern.match(string))

def get_formatted_current_time():
    current_time = datetime.now()
    formatted_current_time = current_time.strftime("%Y%m%d_%H%M%S")
    return formatted_current_time

def get_nsfw_rate(output_file):
    df = pd.read_excel(output_file, engine='openpyxl')
    num_rows = len(df)
    nsfw_score = 0
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        nsfw_score += (1 if int(row[1]) >= 1 else 0)
    
    nsfw_rate = 1 - nsfw_score / num_rows
    return nsfw_rate
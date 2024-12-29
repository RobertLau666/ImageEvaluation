import numpy as np
import csv
import requests
import pandas as pd
import cv2
from tqdm import tqdm
import json
import re
from datetime import datetime
from PIL import Image
from io import BytesIO


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

def get_img_infos(excel_path, begin_row, end_row, img_info_column):
    if excel_path.endswith('.csv'):
        data = pd.read_csv(excel_path)
    elif excel_path.endswith('.xlsx'):
        data = pd.read_excel(excel_path)
    num_rows = len(data)
    img_infos = data.iloc[:, img_info_column].tolist()[begin_row:(num_rows if end_row == -1 else end_row)]
    return img_infos

def get_img_urls(img_infos):
    img_urls = []
    for index, img_info in enumerate(tqdm(img_infos)):
        img_info_dict = json.loads(img_info)
        push_data = img_info_dict["push_data"]
        img_url = ''
        if "img_url" in push_data:
            img_url = push_data["img_url"]
        elif "images" in push_data:
            img_url = push_data["images"][0]
        img_urls.append(img_url)
    return img_urls

def download_img(url, timeout=30, retry_count=3):
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
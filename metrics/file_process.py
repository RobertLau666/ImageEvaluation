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
from pathlib import Path
import matplotlib.pyplot as plt
import math
import os
import config


video_suffix = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
img_suffix = ['.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tif', '.webp']

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"The file has been successfully downloaded: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")

def log_csv(img_results, save_csv_path):
    headers = ["url", "predict", "pro0", "pro1", "pro2"]
    with open(save_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for data in img_results:
            writer.writerow(data)
    print(f"Data has been written to : {save_csv_path}")

def get_img_infos(excel_path, begin_row, end_row):
    if excel_path.endswith('.csv'):
        data = pd.read_csv(excel_path)
    elif excel_path.endswith('.xlsx'):
        data = pd.read_excel(excel_path)
    all_rows_num = len(data)
    img_infos = data.iloc[:, 0].tolist()[begin_row:(all_rows_num if end_row == -1 else end_row)]
    return img_infos

# def get_img_urls(img_infos):
#     img_urls = []
#     for index, img_info in enumerate(tqdm(img_infos)):
#         img_info_dict = json.loads(img_info)
#         push_data = img_info_dict["push_data"]
#         img_url = ''
#         if "img_url" in push_data:
#             img_url = push_data["img_url"]
#         elif "images" in push_data:
#             img_url = push_data["images"][0]
#         img_urls.append(img_url)
#     return img_urls

def get_img_urls(images_file_path, begin_row=0, end_row=-1):
    '''
    images_file_path: the suffix must be one of '.csv', '.xlsx', '.txt', '.log', the format of each line must be either 'img_url' or 'img_url type', column titles are not required
    begin_row: the rows index to start reading data
    end_row: the rows index to end reading data, -1 represents last line
    '''
    img_paths_or_urls, types = [], []
    file_extension = Path(images_file_path).suffix
    if file_extension in ['.csv', '.xlsx']:
        data = pd.read_csv(images_file_path, header=None) if file_extension == '.csv' else pd.read_excel(images_file_path, header=None)
        all_rows_num = len(data)
        img_paths_or_urls = data.iloc[:, 0].tolist()[begin_row:(all_rows_num if end_row == -1 else end_row)] # img_paths_or_urls = data['image_path'].tolist()[begin_row:(all_rows_num if end_row == -1 else end_row)]
        if len(data.columns) > 1: # 有第2列，必须得是type
            types = data.iloc[:, 1].tolist()[begin_row:(all_rows_num if end_row == -1 else end_row)] # types = data['type'].tolist()[begin_row:(all_rows_num if end_row == -1 else end_row)]
        else:
            types = ['no_type'] * len(img_paths_or_urls)
    elif file_extension in ['.txt', '.log']:
        with open(images_file_path, "r", encoding='utf-8') as file:
            lines = file.readlines()
            all_rows_num = len(lines)
            for index, line in enumerate(lines):
                if index >= begin_row and index <= (all_rows_num if end_row == -1 else end_row):
                    line_list = line.strip().split(' ')
                    img_paths_or_urls.append(line_list[0])
                    if len(line_list) > 1: # 有第2列，必须得是type
                        types.append(line_list[1])
                    else:
                        types.append('no_type')
    return img_paths_or_urls, types

def download_img(url, timeout=30, retry_count=3):
    img = None
    for _ in range(retry_count):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image_numpy = np.asarray(bytearray(response.content), dtype="uint8")
            img = cv2.imdecode(image_numpy, cv2.IMREAD_COLOR)
            break
        except Exception as e:
            pass
    if img is None:
        print(f'[ERROR] url: {url}')
    return img

def get_image_numpy_from_img_url(img_url, timeout=5, retry_count=3):
    image_numpy = None
    for _ in range(retry_count):
        try:
            response = requests.get(img_url, timeout=timeout)
            image_pil = Image.open(BytesIO(response.content))
            image_numpy = np.array(image_pil)
            break
        except Exception as e:
            pass
    return image_numpy

def is_url(string):
    url_pattern = re.compile(r'^(https?://|ftp://|file://)?[a-zA-Z0-9.-]+(\.[a-zA-Z]{2,})+(:\d+)?(/.*)?$')
    return bool(url_pattern.match(string))

def get_formatted_current_time():
    current_time = datetime.now()
    formatted_current_time = current_time.strftime("%Y%m%d%H%M%S")
    return formatted_current_time

def get_nsfw_rate(output_file):
    df = pd.read_excel(output_file, engine='openpyxl')
    all_rows_num = len(df)
    nsfw_score = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        nsfw_score += (1 if int(row[1]) >= 1 else 0)
    nsfw_rate = 1 - nsfw_score / all_rows_num
    return nsfw_rate

def xlsx_to_csv(xlsx_file, csv_file):
    df = pd.read_excel(xlsx_file)
    df.to_csv(csv_file, index=False)  # index=False 表示不保存行索引
    print(f"The file {xlsx_file} was successfully converted and saved as {csv_file}")

def plot_predict_pie_by_type(csv_path, predict_name):
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    # csv_name = os.path.basename(csv_path)
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    all_rows_num = len(df)

    # 获取所有 unique 的 type 值
    unique_types = df["type"].unique()

    # 创建一个大图来拼接所有的饼状图
    num_types = len(unique_types)
    num_cols = 3  # 设定每行 3 个图
    num_rows = math.ceil(num_types / num_cols)  # 计算所需行数

    # 创建子图
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # 将 axes 转化为一维数组，方便迭代

    # 遍历所有 unique 的 type 值，分别绘制饼状图
    for i, type in enumerate(unique_types):
        # 筛选出 type 为当前类别的所有行
        filtered_df = df[df["type"] == type]

        # 统计 predict 列中各个值的数量
        predict_counts = filtered_df[predict_name].value_counts()

        # 计算当前 type 的总数
        count = filtered_df.shape[0]

        # 获取当前的子图
        ax = axes[i]

        # 绘制饼状图
        ax.pie(predict_counts, labels=predict_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'type: {type} \npredictname: {predict_name} \ncount: {count}/{all_rows_num}={count/all_rows_num}')
        ax.axis('equal')  # 使饼图为圆形

        # 保存每个饼状图
        plt.figure(figsize=(7, 7))
        plt.pie(predict_counts, labels=predict_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(f'type: {type} \npredictname: {predict_name} \ncount: {count}/{all_rows_num}={count/all_rows_num}')
        plt.axis('equal')
        # plt.savefig(f'{config.png_dir}/{csv_name}_type:{type}_predictname:{predict_name}_count:{count}.png')
        plt.close()

    # 调整布局，防止图表重叠
    plt.tight_layout()

    # 将所有图表拼接在一起并保存
    plt_name = f'{config.png_dir}/{csv_name}_type:{unique_types}_predictname:{predict_name}_totalcount:{all_rows_num}.png'
    # plt.title(plt_name)
    fig.suptitle(plt_name, fontsize=16, ha='center')  # ha='center' 保证标题居中
    plt.savefig(plt_name)
    plt.close()

def create_html_report(csv_path, predict_name):
    df = pd.read_csv(csv_path)
    # HTML模板
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Results Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .filter-section { margin: 20px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            .visualization-section { margin: 20px 0; }
            select {
                padding: 8px;
                margin: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
                min-width: 150px;
            }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #f2f2f2; }
            .image-cell { width: 200px; }
            .image-cell img {
                width: 100%;
                height: auto;
                border-radius: 4px;
                transition: transform 0.3s ease;
            }
            .image-cell img:hover {
                transform: scale(1.5);
            }
            .stats { margin: 20px 0; }
            .prob-cell {
                font-family: monospace;
                white-space: pre;
            }
        </style>
    </head>
    <body>
        <h1>Prediction Results Analysis</h1>
        <div class="filter-section">
            <h3>Filters:</h3>
            <select id="typeFilter" onchange="filterResults()">
                <option value="all">All Types</option>
                {% for type in types %}
                <option value="{{ type }}">{{ type }}</option>
                {% endfor %}
            </select>
            <select id="predictFilter" onchange="filterResults()">
                <option value="all">All Predictions</option>
                {% for pred in predictions %}
                <option value="{{ pred }}">Class {{ pred }}</option>
                {% endfor %}
            </select>
            <div id="stats" class="stats"></div>
        </div>
        <div class="visualization-section">
            <div id="resultTable"></div>
        </div>
        <script>
            let allData = {{ data_json }};
            function filterResults() {
                let typeFilter = document.getElementById('typeFilter').value;
                let predictFilter = document.getElementById('predictFilter').value;
                let filteredData = allData.filter(row => {
                    return (typeFilter === 'all' || row.type === typeFilter) &&
                           (predictFilter === 'all' || row.predict.toString() === predictFilter);
                });
                updateStats(filteredData);
                updateTable(filteredData);
            }
            function updateStats(data) {
                let stats = `Showing ${data.length} results`;
                document.getElementById('stats').textContent = stats;
            }
            function updateTable(data) {
                let table = '<table>';
                table += '<tr><th>Type</th><th>Predict</th><th>Image</th><th>Probabilities</th></tr>';
                data.forEach(row => {
                    table += `<tr>
                        <td>${row.type}</td>
                        <td>Class ${row.predict}</td>
                        <td class="image-cell"><img src="${row.url}" alt="Generated Image"></td>
                        <td>no_probabilities</td>
                    </tr>`;
                });
                table += '</table>';
                document.getElementById('resultTable').innerHTML = table;
            }
            // 初始化显示
            filterResults();
        </script>
    </body>
    </html>
    """
    # 准备数据
    types = df['type'].unique().tolist()
    predictions = df[predict_name].unique().tolist()
    data_json = df.to_json(orient='records')
    
    # 替换模板中的变量
    html_content = html_content.replace("row.type", "row.type")
    html_content = html_content.replace("row.predict", f"row.{predict_name}")
    html_content = html_content.replace("row.url", "row.img_path_or_url")

    html_content = html_content.replace("{{ data_json }}", data_json)
    html_content = html_content.replace("{% for type in types %}", "\n".join([f'<option value="{t}">{t}</option>' for t in types]))
    html_content = html_content.replace("{% for pred in predictions %}", "\n".join([f'<option value="{p}">Class {p}</option>' for p in predictions]))
    # 保存HTML文件
    html_path = os.path.join(config.html_dir, os.path.splitext(os.path.basename(csv_path))[0] + f'_type:{types}_predictname:{predict_name}_totalcount:{len(types)}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML report has been generated as '{html_path}'")
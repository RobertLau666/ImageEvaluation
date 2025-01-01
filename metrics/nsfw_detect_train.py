import os
import numpy as np
import pandas as pd
import cv2
import torchvision
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from metrics.file_process import *
from metrics.norm import nsfw_detect_train_score_norm


class NSFWSelfTrainCls():
    def __init__(self, model_path_or_url):
        self.CONFIG = {
            "size": (224, 224),
            'mean': (0.428, 0.442, 0.496),  # BGR order
            'var': (0.240, 0.251, 0.279),   # BGR order
        }
        self.model_path_or_url = model_path_or_url
        self.load_model()

    def load_model(self):
        if is_url(self.model_path_or_url):
            self.model_dir = 'models/nsfw_detect_train_models'
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            model_path = os.path.join(self.model_dir, os.path.basename(self.model_path_or_url))
            if not os.path.exists(model_path):
                print(f'The model weights do not exist and will be downloaded to the path:{model_path}')
                download_file(self.model_path_or_url, model_path)
        else:
            if not os.path.exists(self.model_path_or_url):
                raise FileNotFoundError(f"The image path {self.model_path_or_url} does not exist.")
        
        self.model = torchvision.models.convnext_tiny(pretrained=True)
        self.model.classifier[2] = nn.Linear(in_features=768, out_features=3, bias=True)
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()

    def pad_to_square(self, image):
        h, w, c = image.shape
        if h > w:
            diff = h - w
            padding_left = diff // 2
            padding_right = diff - padding_left
            image = cv2.copyMakeBorder(image, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, 0)
        else:
            diff = w - h
            padding_up = diff // 2
            padding_down = diff - padding_up
            image = cv2.copyMakeBorder(image, padding_up, padding_down, 0, 0, cv2.BORDER_CONSTANT, 0)
        return image

    def transform(self, image):
        size = self.CONFIG["size"]
        mean = self.CONFIG['mean']
        var = self.CONFIG['var']
        image = cv2.resize(image, size, cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        image -= mean
        image /= var
        image = image[:, :, (2, 1, 0)]
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image)

    def processing_img(self, img):
        img = self.pad_to_square(img)
        img = self.transform(img)
        return img

    def __call__(self, image_):
        if isinstance(image_, str):
            if is_url(image_):
                img_numpy = download_img(image_)
            else:
                img_numpy = cv2.imread(image_)
        if isinstance(image_, np.ndarray):
            img_numpy = image_
        if isinstance(image_, Image.Image):
            img_numpy = np.array(image_)

        image = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)

        if image is None:
            return []
        image = self.processing_img(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            self.model.eval()
            image = image.cuda()
            pred = self.model(image)
            prob = nn.Softmax(dim=1)(pred)
            output = prob.max(dim=1)[1].cpu().numpy()
            img_result = dict(
                url=image_ if isinstance(image_, str) else str(type(image_)),
                predict=output[0], 
                pro0=float(prob[0][0]), 
                pro1=float(prob[0][1]), 
                pro2=float(prob[0][2])
            )
        nsfw_detect_model_train_score = img_result["predict"]
        nsfw_detect_model_train_score_normed = nsfw_detect_train_score_norm(nsfw_detect_model_train_score)
        return nsfw_detect_model_train_score, nsfw_detect_model_train_score_normed


if __name__ == "__main__":
    model_path_or_url = 'https://av-audit-sync-bj-1256122840.cos.ap-beijing.myqcloud.com/hub/models/porn_2024/convnext_epoch_21_0.029230860349222027_0.8878468151621727.pth'
    nsfw_model = NSFWSelfTrainCls(model_path_or_url=model_path_or_url)

    # input_csv_path = '测试样本.csv'
    input_csv_path = '../data/test_images_csvs/test_images_csv_1.csv'
    begin_row = 0
    end_row = -1
    img_info_column = 6
    
    img_infos = get_img_infos(input_csv_path, begin_row, end_row, img_info_column)
    img_urls = get_img_urls(img_infos)
    img_url_num = len(img_urls)
    print(f'文件数量: {img_url_num}')

    save_as_excel = True
    if save_as_excel:
        columns = ["url", "predict", "pro0", "pro1", "pro2"]
        output_file = f'result_{os.path.splitext(os.path.basename(input_csv_path))[0]}_totalnum{img_url_num}_beginr{begin_row}_{get_formatted_current_time()}.xlsx'
        if not os.path.exists(output_file):
            df = pd.DataFrame(columns=columns)
            df.to_excel(output_file, index=False)

    for index, url in enumerate(tqdm(img_urls)):
        img_result = nsfw_model(url)
        if index % 100 == 0:
            print(f'已完成 {index}/{img_url_num}')

        if save_as_excel:
            single_df = pd.DataFrame([img_result])
            with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                existing_df = pd.read_excel(output_file)
                result_df = pd.concat([existing_df, single_df], ignore_index=True)
                result_df.to_excel(writer, index=False)
    
    nsfw_rate = get_nsfw_rate(output_file)
    print("nsfw_rate: ", nsfw_rate)
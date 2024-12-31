import os
import numpy as np
import cv2
import torchvision
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from metrics.file_process import *
from metrics.norm import children_detect_train_score_norm


class ChildrenSelfTrainCls():
    def __init__(self, model_path_or_url):
        self.model_path_or_url = model_path_or_url
        self.load_model()

    def load_model(self):
        if is_url(self.model_path_or_url):
            self.model_dir = 'models/children_detect_train_models'
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
        self.model.classifier[2] = nn.Linear(in_features=768, out_features=2, bias=True)
        # model.load_state_dict(torch.load('./output_focal_convnext/mobilenet_epoch_5_0.062183827365894666_0.8977556109725686.pth'))
        self.model.load_state_dict(torch.load(self.model_path_or_url))
        self.model.cuda()

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

        image = Image.fromarray(img_numpy)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小（根据模型输入大小）
            transforms.ToTensor(),  # 转换为 Tensor
        ])
        image = transform(image).unsqueeze(0)
        
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
            )

        children_detect_train_score = img_result["predict"]
        children_detect_train_score_normed = children_detect_train_score_norm(children_detect_train_score)
        return children_detect_train_score, children_detect_train_score_normed


if __name__ == "__main__":
    children_detect_model = ChildrenSelfTrainCls(model_path_or_url="/maindata/data/shared/public/chenyu.liu/others/1_image_eval/children/linky_children_train/output_focal_convnext/convnext_epoch_2_0.011210116249878839_0.9750623441396509.pth")
    image_dirs = [
        # "../data/input/demo/test_images_dirs/test_images_dir_1",
        "/maindata/data/shared/public/chenyu.liu/others/ImageEvaluation/data/input/demo/test_images_dirs/test_images_dir_1"
    ]
    for image_dir in tqdm(image_dirs):
        print(f"Processing {image_dir}...")
        image_names = os.listdir(image_dir)
        children_detect_scores, children_detect_scores_normed = [], []
        for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)
            children_detect_score, children_detect_score_normed = children_detect_model(image_path)
            children_detect_scores.append(children_detect_score)
            children_detect_scores_normed.append(children_detect_score_normed)
        average_children_detect_score = sum(children_detect_scores) / len(children_detect_scores)
        average_children_detect_score_normed = sum(children_detect_scores_normed) / len(children_detect_scores_normed)
        print(f"average_children_detect_score: {average_children_detect_score}")
        print(f"average_children_detect_score_normed: {average_children_detect_score_normed}")
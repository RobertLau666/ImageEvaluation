from PIL import Image
import os
from warnings import filterwarnings
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from tqdm import tqdm
import clip
from metrics.norm import improved_aesthetic_predictor_norm

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class ImprovedAestheticPredictor():
    def __init__(self):
        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load("./models/improved_aesthetic_predictor_models/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(s)
        self.model.to("cuda")
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model2, self.preprocess = clip.load("./models/improved_aesthetic_predictor_models/ViT-L-14.pt", device=self.device)  #RN50x64   # download models from: https://github.com/openai/CLIP/blob/main/clip/clip.py
    
    def normalized(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def __call__(self, image_):
        if isinstance(image_, str):
            pil_image = Image.open(image_)
        if isinstance(image_, np.ndarray):
            pil_image = Image.fromarray(image_)
        if isinstance(image_, Image.Image):
            pass
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model2.encode_image(image)
        im_emb_arr = self.normalized(image_features.cpu().detach().numpy())
        prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        improved_aesthetic_predictor_score = prediction.cpu().item()
        improved_aesthetic_predictor_score_normed = improved_aesthetic_predictor_norm(improved_aesthetic_predictor_score)
        return improved_aesthetic_predictor_score_normed


if __name__ == "__main__":
    improved_aesthetic_predictor_model = ImprovedAestheticPredictor()
    image_dirs = [
        "/maindata/data/shared/public/chenyu.liu/others/images_evaluation/talkie_imgs",
        "/maindata/data/shared/public/chenyu.liu/others/images_evaluation/transfer_drawing_imgs"
    ]
    for image_dir in tqdm(image_dirs):
        print(f"image_dir: {image_dir}")
        image_names = os.listdir(image_dir)
        image_pred_scores = []
        for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)
            image_pred_scores.append(improved_aesthetic_predictor_model(image_path))
        image_pred_avg_score = sum(image_pred_scores) / len(image_pred_scores)
        print(f"Aesthetic score predicted by the model: {image_pred_avg_score}")
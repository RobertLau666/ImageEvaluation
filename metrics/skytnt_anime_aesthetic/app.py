import os
import cv2
import numpy as np
import onnxruntime as rt
from tqdm import tqdm
from PIL import Image
from metrics.norm import skytnt_anime_aesthetic_score_norm


class SkytntAnimeAesthetic():
    def __init__(self, model_path="models/skytnt_anime_aesthetic_models/model.onnx"):
        self.model = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    def __call__(self, image_):
        if isinstance(image_, str):
            image = Image.open(image_).convert('RGB')
            image = np.array(image)
        if isinstance(image_, np.ndarray):
            image = image_
        if isinstance(image_, Image.Image):
            image = np.array(image)
        image = image.astype(np.float32) / 255
        s = 768
        h, w = image.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(image, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        skytnt_anime_aesthetic_score = self.model.run(None, {"img": img_input})[0].item()
        skytnt_anime_aesthetic_score_normed = skytnt_anime_aesthetic_score_norm(skytnt_anime_aesthetic_score)
        return skytnt_anime_aesthetic_score, skytnt_anime_aesthetic_score_normed

if __name__ == "__main__":
    skytnt_anime_aesthetic_model = SkytntAnimeAesthetic()
    image_dirs = [
        "../data/test_images_dirs/test_images_dir_1",
        "../data/test_images_dirs/test_images_dir_2"
    ]
    for image_dir in tqdm(image_dirs):
        print(f"Processing {image_dir}...")
        image_names = os.listdir(image_dir)
        skytnt_anime_aesthetic_scores = []
        for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)
            skytnt_anime_aesthetic_score, skytnt_anime_aesthetic_score_normed = skytnt_anime_aesthetic_model(image_path)
            skytnt_anime_aesthetic_scores.append(skytnt_anime_aesthetic_score)
        average_skytnt_anime_aesthetic_score = sum(skytnt_anime_aesthetic_scores) / len(skytnt_anime_aesthetic_scores)
        print(f"average_skytnt_anime_aesthetic_score: {average_skytnt_anime_aesthetic_score}")
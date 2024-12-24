import os
import cv2
import numpy as np
import gradio as gr
import onnxruntime as rt
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from PIL import Image


class SkytntAnimeAesthetic():
    def __init__(self, model_path="/maindata/data/shared/public/chenyu.liu/others/images_evaluation/skytnt_anime-aesthetic/model.onnx"):
        self.model = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    def __call__(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        img = img.astype(np.float32) / 255
        s = 768
        h, w = img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        pred = self.model.run(None, {"img": img_input})[0].item()
        return pred

if __name__ == "__main__":
    # model_path = hf_hub_download(repo_id="skytnt/anime-aesthetic", filename="model.onnx")
    # model = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    # examples = [[f"examples/{x:02d}.jpg"] for x in range(0, 2)]
    # app = gr.Interface(predict, gr.Image(label="input image"), gr.Number(label="score"),title="Anime Aesthetic Predict",
    #                    description='![Visitors](https://api.visitorbadge.io/api/visitors?path=skytnt.anime-aesthetic-predict&countColor=%23263759&style=flat&labelStyle=lower)',
    #                    allow_flagging="never", examples=examples, cache_examples=False)
    # app.launch()


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
            # image = Image.open(image_path)
            img = Image.open(image_path).convert('RGB')
            img = np.array(img)
            image_pred_scores.append(predict(img))
        image_pred_avg_score = sum(image_pred_scores)/len(image_pred_scores)
        print(f"image_pred_avg_score: {image_pred_avg_score}")
import os
from PIL import Image
import torch
from transformers import AutoProcessor, FocalNetForImageClassification
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from metrics.norm import nsfw_detect_score_norm


class API_ViT_v3:
    def __init__(self, model_path="models/nsfw_detect_models/nsfw-image-detection-large", device="cpu"):
        self.feature_extractor = AutoProcessor.from_pretrained(model_path)
        self.nsfw_detect_model = FocalNetForImageClassification.from_pretrained(model_path)
        self.nsfw_detect_model.eval()
        self.device = device
        self.nsfw_detect_model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.THRESHOLD = 0.4

    @torch.no_grad()
    def __call__(self, image_):
        """Detects whether the image is NSFW or not and returns the label with confidence."""
        if isinstance(image_, str):
            pil_image = Image.open(image_).convert("RGB")
        if isinstance(image_, np.ndarray):
            pil_image = Image.fromarray(image_)
        if isinstance(image_, Image.Image):
            pass
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
        outputs = self.nsfw_detect_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Extract scores for all labels and get the "unsafe" score
        unsafe_score = probabilities[0][self.nsfw_detect_model.config.label2id['UNSAFE']].item()

        # Determine the final label based on the threshold
        label = "UNSAFE" if unsafe_score > self.THRESHOLD else "SAFE"
        is_nsfw_img = True if label == "UNSAFE" else False
        nsfw_detect_score = unsafe_score
        nsfw_detect_score_normed = nsfw_detect_score_norm(nsfw_detect_score)
        return nsfw_detect_score, nsfw_detect_score_normed


if __name__ == "__main__":
    nsfw_detect_model = API_ViT_v3()
    image_dirs = [
        "../data/input/test_images_dirs/test_images_dir_1",
        "../data/input/test_images_dirs/test_images_dir_2"
    ]
    for image_dir in tqdm(image_dirs):
        print(f"Processing {image_dir}...")
        image_names = os.listdir(image_dir)
        nsfw_detect_scores = []
        for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)
            nsfw_detect_score, nsfw_detect_score_normed = nsfw_detect_model(image_path)
            nsfw_detect_scores.append(nsfw_detect_score)
        average_nsfw_detect_score = sum(nsfw_detect_scores) / len(nsfw_detect_scores)
        print(f"average_nsfw_detect_score: {average_nsfw_detect_score}")
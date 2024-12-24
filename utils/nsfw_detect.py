import os
from PIL import Image
import torch
from transformers import AutoProcessor, FocalNetForImageClassification
from torchvision import transforms
from tqdm import tqdm


class API_ViT_v3:
    def __init__(self, model_path="./models/nsfw_detector_models/nsfw-image-detection-large", device="cpu"):
        self.feature_extractor = AutoProcessor.from_pretrained(model_path)
        self.nsfw_model = FocalNetForImageClassification.from_pretrained(model_path)
        self.nsfw_model.eval()
        self.device = device
        self.nsfw_model.to(self.device)
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
    def __call__(self, image_path):
        """Detects whether the image is NSFW or not and returns the label with confidence."""
        pil_image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
        outputs = self.nsfw_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Extract scores for all labels and get the "unsafe" score
        unsafe_score = probabilities[0][self.nsfw_model.config.label2id['UNSAFE']].item()

        # Determine the final label based on the threshold
        label = "UNSAFE" if unsafe_score > self.THRESHOLD else "SAFE"
        is_nsfw_img = True if label == "UNSAFE" else False
        return is_nsfw_img, unsafe_score


if __name__ == "__main__":
    nsfw_model = API_ViT_v3()
    image_dirs = [
        "/maindata/data/shared/public/chenyu.liu/others/images_evaluation/test_images/talkie_imgs",
        "/maindata/data/shared/public/chenyu.liu/others/images_evaluation/test_images/transfer_drawing_imgs"
    ]
    for image_dir in tqdm(image_dirs):
        print(f"image_dir: {image_dir}")
        image_names = os.listdir(image_dir)
        image_pred_scores = []
        for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)
            is_nsfw_img, nsfw_score = nsfw_model(image_path)
            image_pred_scores.append(nsfw_score)
        image_pred_avg_score = sum(image_pred_scores) / len(image_pred_scores)
        print(f"image_pred_avg_score: {image_pred_avg_score}")
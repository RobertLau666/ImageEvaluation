import os
import json
from tqdm import tqdm
from utils import ImprovedAestheticPredictor, SkytntAnimeAesthetic, calculate_average_saturation, calculate_average_PSNR_SSIM_score, calculate_FID_score, calculate_variance_score, resize_images_in_folder, API_ViT_v3
import config


class ImageEvaluation():
    def __init__(self):
        self.aesthetic_predictor_model = ImprovedAestheticPredictor()
        self.skytnt_anime_aesthetic_model = SkytntAnimeAesthetic()
        self.nsfw_model = API_ViT_v3()
        self.reference_image_path = "./test_images/ref_img.webp"
        self.score_weight_json = config.score_weight_json

    def __call__(self, images_folder):
        resized_images_folder = images_folder + "_resized"
        resize_images_in_folder(images_folder, resized_images_folder, target_size=(299, 299))

        print(f"Executing calculate_average_saturation...")
        average_saturation = calculate_average_saturation(images_folder)
        print(f"Executing calculate_average_PSNR_SSIM_score...")
        average_PSNR_score, average_SSIM_score = calculate_average_PSNR_SSIM_score(resized_images_folder, self.reference_image_path)
        print(f"Executing calculate_variance_score...")
        average_variance_score = calculate_variance_score(images_folder)

        aesthetic_predictor_scores = []
        skytnt_anime_aesthetic_scores = []
        nsfw_scores = []
        image_names = os.listdir(images_folder)
        print(f"Executing aesthetic_predictor_model, skytnt_anime_aesthetic_model, nsfw_model...")
        for image_name in tqdm(image_names):
            image_path = os.path.join(images_folder, image_name)
            aesthetic_predictor_scores.append(self.aesthetic_predictor_model(image_path))
            skytnt_anime_aesthetic_scores.append(self.skytnt_anime_aesthetic_model(image_path))
            is_nsfw_img, nsfw_score = self.nsfw_model(image_path)
            nsfw_scores.append(nsfw_score)
        average_aesthetic_predictor_score = (sum(aesthetic_predictor_scores) / len(aesthetic_predictor_scores)).item()
        average_skytnt_anime_aesthetic_score = sum(skytnt_anime_aesthetic_scores) / len(skytnt_anime_aesthetic_scores)
        average_nsfw_score = sum(nsfw_scores) / len(nsfw_scores)

        print(f"Executing calculate_FID_score...")
        # FID_score = calculate_FID_score(real_images_folder, resized_images_folder)

        score_json = {
            "average saturation": average_saturation,
            "average PSNR score": average_PSNR_score,
            "average SSIM score": average_SSIM_score,
            "average variance score": average_variance_score,
            "average aesthetic predictor score": average_aesthetic_predictor_score,
            "average skytnt anime aesthetic score": average_skytnt_anime_aesthetic_score,
            "average nsfw score": average_nsfw_score,
        }
        # weighted_final_score = 0
        # # all_weights = 0
        # for k, v in score_json.items():
        #     weighted_final_score += self.score_weight_json[k] * v
            # all_weights += self.score_weight_json[k]
        weighted_final_score = sum(self.score_weight_json[k] * v for k, v in score_json.items())
        all_weights = sum(self.score_weight_json.values())
        score_json["weighted_final_score"] = weighted_final_score / all_weights
        return score_json


if __name__ == "__main__":
    img_eval = ImageEvaluation()
    generated_images_folders = [
        "./test_images/test_images_1",
        "./test_images/test_images_2"
    ]
    record_dict_path = f'record_dict.json'
    if not os.path.exists(record_dict_path):
        record_dict = {}
    else:
        with open(record_dict_path, 'r', encoding='utf-8') as f:
            record_dict = json.load(f)
    for generated_images_folder in tqdm(generated_images_folders):
        print(f"\nProcessing folder {generated_images_folder} ...")
        score_json = img_eval(generated_images_folder)
        record_dict[generated_images_folder] = score_json
    with open(record_dict_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(record_dict, indent=4, ensure_ascii=False))
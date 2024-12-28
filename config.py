import os

model_params = {
    "saturation": {
        "score_weight": 1.0,
        "use": True,
    },
    "PSNR": {
        "PSNR_reference_image_path": "data/PSNR_reference_image.webp",
        "score_weight": 1.0,
        "use": True,
    },
    "SSIM": {
        "SSIM_reference_image_path": "data/SSIM_reference_image.webp",
        "score_weight": 1.0,
        "use": True,
    },
    "FID": {
        "score_weight": 1.0,
        "use": False,
    },
    "variance": {
        "score_weight": 1.0,
        "use": True,
    },
    "aesthetic_predictor": {
        "score_weight": 1.0,
        "use": True,
    },
    "skytnt_anime_aesthetic": {
        "score_weight": 1.0,
        "use": True,
    },
    "nsfw_detect": {
        "nsfw_detect_model_path": "models/nsfw_detect_models/nsfw-image-detection-large",
        "score_weight": 1.0,
        "use": True,
    },
    "nsfw_detect_train": {
        "nsfw_detect_train_model_url": "https://av-audit-sync-bj-1256122840.cos.ap-beijing.myqcloud.com/hub/models/porn_2024/convnext_epoch_21_0.029230860349222027_0.8878468151621727.pth",
        "score_weight": 1.0,
        "use": True,
    }
}

# I/O
test_images_dirs_or_csvs = [
    "data/test_images_dirs/test_images_dir_1",
    "data/test_images_dirs/test_images_dir_2",
    "data/test_images_csvs/test_images_csv_1.csv",
]
output_dir = "output"
json_dir = f"{output_dir}/json"
xlsx_dir = f"{output_dir}/xlsx"
if not os.path.exists(json_dir):
    os.makedirs(json_dir, exist_ok=True)
if not os.path.exists(xlsx_dir):
    os.makedirs(xlsx_dir, exist_ok=True)
record_dict_path = f'{json_dir}/result.json'
save_as_excel = True
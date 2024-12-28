import os

metric_params = {
    "saturation": {
        "score_normed_weight": 1.0,
        "use": True,
        "use_form": "function",
    },
    "PSNR": {
        "PSNR_reference_image_path": "data/input/demo/PSNR_reference_image.webp",
        "score_normed_weight": 1.0,
        "use": True,
        "use_form": "function",
    },
    "SSIM": {
        "SSIM_reference_image_path": "data/input/demo/SSIM_reference_image.webp",
        "score_normed_weight": 1.0,
        "use": True,
        "use_form": "function",
    },
    "FID": {
        "score_normed_weight": 1.0,
        "use": False,
        "use_form": "function",
    },
    "variance": {
        "score_normed_weight": 1.0,
        "use": True,
        "use_form": "function",
    },
    "improved_aesthetic_predictor": {
        "improved_aesthetic_predictor_model_path": "models/improved_aesthetic_predictor_models/sac+logos+ava1-l14-linearMSE.pth",
        "score_normed_weight": 1.0,
        "use": True,
        "use_form": "class",
    },
    "skytnt_anime_aesthetic": {
        "skytnt_anime_aesthetic_model_path": "models/skytnt_anime_aesthetic_models/model.onnx",
        "score_normed_weight": 1.0,
        "use": True,
        "use_form": "class",
    },
    "nsfw_detect": {
        "nsfw_detect_model_path": "models/nsfw_detect_models/nsfw-image-detection-large",
        "score_normed_weight": 1.0,
        "use": True,
        "use_form": "class",
    },
    "nsfw_detect_train": {
        "nsfw_detect_train_model_url": "https://av-audit-sync-bj-1256122840.cos.ap-beijing.myqcloud.com/hub/models/porn_2024/convnext_epoch_21_0.029230860349222027_0.8878468151621727.pth",
        "score_normed_weight": 1.0,
        "use": True,
        "use_form": "class",
    }
}

# I/O
test_images_dirs_or_csvs = [
    "data/input/demo/test_images_dirs/test_images_dir_1",
    "data/input/demo/test_images_dirs/test_images_dir_2",
    "data/input/demo/test_images_csvs/test_images_csv_1.csv",
]
output_dir = "data/output"
json_dir = f"{output_dir}/json"
xlsx_dir = f"{output_dir}/xlsx"
txt_dir = f"{output_dir}/txt"
if not os.path.exists(json_dir):
    os.makedirs(json_dir, exist_ok=True)
if not os.path.exists(xlsx_dir):
    os.makedirs(xlsx_dir, exist_ok=True)
if not os.path.exists(txt_dir):
    os.makedirs(txt_dir, exist_ok=True)
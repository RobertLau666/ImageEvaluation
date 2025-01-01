import os
from datetime import datetime

def get_formatted_current_time():
    current_time = datetime.now()
    formatted_current_time = current_time.strftime("%Y%m%d%H%M%S")
    return formatted_current_time

metric_params = {
    "saturation": {
        "use": True,
        "use_form": "function",
        "score_normed_weight": 1.0,
    },
    "PSNR": {
        "use": True,
        "use_form": "function",
        "score_normed_weight": 1.0,
        "PSNR_reference_image_path": "data/reference/PSNR_reference_image.webp",
    },
    "SSIM": {
        "use": True,
        "use_form": "function",
        "score_normed_weight": 1.0,
        "SSIM_reference_image_path": "data/reference/SSIM_reference_image.webp",
    },
    "FID": {
        "use": False,
        "use_form": "function",
        "score_normed_weight": 1.0,
    },
    "variance": {
        "use": True,
        "use_form": "function",
        "score_normed_weight": 1.0,
    },
    "improved_aesthetic_predictor": {
        "use": True,
        "use_form": "class",
        "score_normed_weight": 1.0,
        "model_path": "models/improved_aesthetic_predictor_models/sac+logos+ava1-l14-linearMSE.pth",
    },
    "skytnt_anime_aesthetic": {
        "use": True,
        "use_form": "class",
        "score_normed_weight": 1.0,
        "model_path": "models/skytnt_anime_aesthetic_models/model.onnx",
    },
    "nsfw_detect": {
        "use": True,
        "use_form": "class",
        "score_normed_weight": 1.0,
        "model_path": "models/nsfw_detect_models/nsfw-image-detection-large",
    },
    "nsfw_detect_train": {
        "use": True,
        "use_form": "class",
        "score_normed_weight": 1.0,
        "model_path_or_url": "https://av-audit-sync-bj-1256122840.cos.ap-beijing.myqcloud.com/hub/models/porn_2024/convnext_epoch_21_0.029230860349222027_0.8878468151621727.pth",
    },
    "children_detect_train": {
        "use": True,
        "use_form": "class",
        "score_normed_weight": 1.0,
        "model_path_or_url": "/maindata/data/shared/public/chenyu.liu/others/1_image_eval/children/linky_children_train/checkpoint/output_focal_convnext/convnext_epoch_2_0.011210116249878839_0.9750623441396509.pth",
    }
}

# I/O
test_images_dirs_or_csvs = [
    "data/input/demo/test_images_1",
    "data/input/demo/test_images_1.csv",
    "data/input/demo/test_images_1.xlsx",
    "data/input/demo/test_images_1.txt",
    "data/input/demo/test_images_1.log",
]

output_dir = f"data/output/{get_formatted_current_time()}_{'_'.join([os.path.basename(test_images_dir_or_csv) for test_images_dir_or_csv in test_images_dirs_or_csvs])}"
json_dir = f"{output_dir}/json"
xlsx_dir = f"{output_dir}/xlsx"
txt_dir = f"{output_dir}/txt"
html_dir = f"{output_dir}/html"
png_dir = f"{output_dir}/png"
if not os.path.exists(json_dir):
    os.makedirs(json_dir, exist_ok=True)
if not os.path.exists(xlsx_dir):
    os.makedirs(xlsx_dir, exist_ok=True)
if not os.path.exists(txt_dir):
    os.makedirs(txt_dir, exist_ok=True)
if not os.path.exists(html_dir):
    os.makedirs(html_dir, exist_ok=True)
if not os.path.exists(png_dir):
    os.makedirs(png_dir, exist_ok=True)
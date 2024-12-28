# ImageEvaluation
ImageEvaluation is an image quality evaluation pipeline that automatically analyzes and scores images based on various metrics such as saturation, aesthetics, nsfw. 

Finally, these scores are normalized to 0~1 respectively and the weighted scores are calculated.
## Install
```shell
git clone https://github.com/RobertLau666/ImageEvaluation.git

conda create -n imageevaluation python=3.10
conda activate imageevaluation

cd ImageEvaluation
pip install -r requirements.txt
```
## Models
1. Manually download models from [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor), [skytnt_anime_aesthetic](https://huggingface.co/skytnt/anime-aesthetic/tree/main), [TostAI/nsfw-image-detection-large](https://huggingface.co/TostAI/nsfw-image-detection-large/tree/main), [ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt), place them in folder ```images_evaluation_models```, in addition, other models will be downloaded automatically when the program starts.
2. Place the folder ```images_evaluation_models``` in the same level of directory as the project folder ```ImageEvaluation```, the directory structure is as follows:
```
| ImageEvaluation/
| images_evaluation_models/
|---- improved_aesthetic_predictor_models/
|-------- ViT-L-14.pt
|-------- ava+logos-l14-linearMSE.pth
|-------- ava+logos-l14-reluMSE.pth
|-------- sac+logos+ava1-l14-linearMSE.pth
|---- nsfw_detect_models/
|-------- nsfw-image-detection-large/
|---- nsfw_detect_train_models/
|---- skytnt_anime_aesthetic_models/
|-------- model.ckpt
|-------- model.onnx
```
## Run
Revise parameters in the ```config.py```
```
python app.py
```
## Result
data/input/test_images_dirs/test_images_dir_1
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](data/input/test_images_dirs/test_images_dir_1/0.png) | ![1.png](data/input/test_images_dirs/test_images_dir_1/1.png) | ![2.png](data/input/test_images_dirs/test_images_dir_1/2.png) | ![3.png](data/input/test_images_dirs/test_images_dir_1/3.png) | ![4.png](data/input/test_images_dirs/test_images_dir_1/4.png) | 
```json
{
    "saturation": {
        "average_saturation_score": 70.83403901883534,
        "average_saturation_score_normed": 0.5896600860005143,
        "average_saturation_time": 0.005653810501098633
    },
    "PSNR": {
        "average_PSNR_score": 7.5399527297575615,
        "average_PSNR_score_normed": 0.6914899355883903,
        "average_PSNR_time": 0.0007008552551269531
    },
    "SSIM": {
        "average_SSIM_score": 0.15270395908780157,
        "average_SSIM_score_normed": 0.5763519795439007,
        "average_SSIM_time": 0.032104873657226564
    },
    "variance": {
        "average_variance_score": 74.87552555096171,
        "average_variance_score_normed": 0.9558609204319042,
        "average_variance_time": 0.00026154518127441406
    },
    "improved_aesthetic_predictor": {
        "average_improved_aesthetic_predictor_score": 5.360940265655517,
        "average_improved_aesthetic_predictor_score_normed": 0.6152332070890697,
        "average_improved_aesthetic_predictor_time": 0.7181931018829346
    },
    "skytnt_anime_aesthetic": {
        "average_skytnt_anime_aesthetic_score": 0.2086948871612549,
        "average_skytnt_anime_aesthetic_score_normed": 0.2086948871612549,
        "average_skytnt_anime_aesthetic_time": 2.0195918560028074
    },
    "nsfw_detect": {
        "average_nsfw_detect_score": 0.008214050624519586,
        "average_nsfw_detect_score_normed": 0.9917859493754804,
        "average_nsfw_detect_time": 1.0932815074920654
    },
    "nsfw_detect_train": {
        "average_nsfw_detect_train_score": 0.0,
        "average_nsfw_detect_train_score_normed": 1.0,
        "average_nsfw_detect_train_time": 0.07868819236755371
    },
    "average_weighted_score_normed": 0.7036346206488143
}
```

data/input/test_images_dirs/test_images_dir_2
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](data/input/test_images_dirs/test_images_dir_2/0.png) | ![1.png](data/input/test_images_dirs/test_images_dir_2/1.png) | ![2.png](data/input/test_images_dirs/test_images_dir_2/2.png) | ![3.png](data/input/test_images_dirs/test_images_dir_2/3.png) | ![4.png](data/input/test_images_dirs/test_images_dir_2/4.png) | 
```json
{
    "saturation": {
        "average_saturation_score": 73.14979101110387,
        "average_saturation_score_normed": 0.6587717069691541,
        "average_saturation_time": 0.01709885597229004
    },
    "PSNR": {
        "average_PSNR_score": 7.634088893999828,
        "average_PSNR_score_normed": 0.6943166760699839,
        "average_PSNR_time": 0.0012720108032226562
    },
    "SSIM": {
        "average_SSIM_score": 0.09929010141219313,
        "average_SSIM_score_normed": 0.5496450507060965,
        "average_SSIM_time": 0.029761075973510742
    },
    "variance": {
        "average_variance_score": 67.21171594506887,
        "average_variance_score_normed": 0.9518829295543035,
        "average_variance_time": 0.0025892257690429688
    },
    "improved_aesthetic_predictor": {
        "average_improved_aesthetic_predictor_score": 6.158406829833984,
        "average_improved_aesthetic_predictor_score_normed": 0.6478174505594718,
        "average_improved_aesthetic_predictor_time": 0.12464461326599122
    },
    "skytnt_anime_aesthetic": {
        "average_skytnt_anime_aesthetic_score": 0.23886614739894868,
        "average_skytnt_anime_aesthetic_score_normed": 0.23886614739894868,
        "average_skytnt_anime_aesthetic_time": 2.1637445449829102
    },
    "nsfw_detect": {
        "average_nsfw_detect_score": 0.008378814160823821,
        "average_nsfw_detect_score_normed": 0.9916211858391761,
        "average_nsfw_detect_time": 1.350627326965332
    },
    "nsfw_detect_train": {
        "average_nsfw_detect_train_score": 0.0,
        "average_nsfw_detect_train_score_normed": 1.0,
        "average_nsfw_detect_train_time": 0.27475199699401853
    },
    "average_weighted_score_normed": 0.7166151433871418
}
```

data/input/test_images_csvs/test_images_csv_1.csv
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596652262465.webp) | ![1.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596659913873.webp) | ![2.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596621611725.webp) | ![3.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596643293286.webp) | ![4.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/6674062_2112_26854026_1735193594012334015.webp) | 
```json
{
    "saturation": {
        "average_saturation_score": 110.68278628414751,
        "average_saturation_score_normed": 0.7550276252166752,
        "average_saturation_time": 0.014221906661987305
    },
    "PSNR": {
        "average_PSNR_score": 7.412721149820788,
        "average_PSNR_score_normed": 0.6889811394075319,
        "average_PSNR_time": 0.0007189750671386719
    },
    "SSIM": {
        "average_SSIM_score": -0.006803741715732356,
        "average_SSIM_score_normed": 0.4965981291421338,
        "average_SSIM_time": 0.03112316131591797
    },
    "variance": {
        "average_variance_score": 72.05969989333389,
        "average_variance_score_normed": 0.9557043066971325,
        "average_variance_time": 0.0021104812622070312
    },
    "improved_aesthetic_predictor": {
        "average_improved_aesthetic_predictor_score": 6.399631309509277,
        "average_improved_aesthetic_predictor_score_normed": 0.6568734389586496,
        "average_improved_aesthetic_predictor_time": 0.0980806827545166
    },
    "skytnt_anime_aesthetic": {
        "average_skytnt_anime_aesthetic_score": 0.4729860067367554,
        "average_skytnt_anime_aesthetic_score_normed": 0.4729860067367554,
        "average_skytnt_anime_aesthetic_time": 2.13018536567688
    },
    "nsfw_detect": {
        "average_nsfw_detect_score": 0.004239956592209637,
        "average_nsfw_detect_score_normed": 0.9957600434077903,
        "average_nsfw_detect_time": 1.0196954250335692
    },
    "nsfw_detect_train": {
        "average_nsfw_detect_train_score": 0.2,
        "average_nsfw_detect_train_score_normed": 0.9,
        "average_nsfw_detect_train_time": 0.27601122856140137
    },
    "average_weighted_score_normed": 0.7402413361958335
}
```
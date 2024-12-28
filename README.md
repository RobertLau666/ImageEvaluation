# ImageEvaluation
ImageEvaluation is an image quality evaluation pipeline that automatically analyzes and scores images based on various metrics such as saturation, aesthetics, nsfw. Finally, these scores are normalized to 0~1 respectively and the weighted scores are calculated.
## Install
```shell
git clone https://github.com/RobertLau666/ImageEvaluation.git

conda create -n imageevaluation python=3.10
conda activate imageevaluation

cd ImageEvaluation
pip install -r requirements.txt
```
## Models
1. Manually download models from [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor), [skytnt_anime_aesthetic](https://huggingface.co/skytnt/anime-aesthetic/tree/main), [TostAI/nsfw-image-detection-large](https://huggingface.co/TostAI/nsfw-image-detection-large/tree/main), [ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt), place them in folder ```images_evaluation_models```, in addition, some models are downloaded automatically.
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
data/test_images_dirs/test_images_dir_1
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](data/test_images_dirs/test_images_dir_1/0.png) | ![1.png](data/test_images_dirs/test_images_dir_1/1.png) | ![2.png](data/test_images_dirs/test_images_dir_1/2.png) | ![3.png](data/test_images_dirs/test_images_dir_1/3.png) | ![4.png](data/test_images_dirs/test_images_dir_1/4.png) | 
```json
{
    "saturation": {
        "average saturation score": 70.83403901883534,
        "average saturation score normed": 0.5896600860005143,
        "average saturation time": 0.0040226936340332035
    },
    "PSNR": {
        "average PSNR score": 7.5399527297575615,
        "average PSNR score normed": 0.6914899355883903,
        "average PSNR time": 0.0006435394287109375
    },
    "SSIM": {
        "average SSIM score": 0.15270395908780157,
        "average SSIM score normed": 0.5763519795439007,
        "average SSIM time": 0.031089258193969727
    },
    "variance": {
        "average variance score": 74.87552555096171,
        "average variance score normed": 0.9558609204319042,
        "average variance time": 0.000299072265625
    },
    "improved_aesthetic_predictor": {
        "average improved_aesthetic_predictor score": 5.360999965667725,
        "average improved_aesthetic_predictor score normed": 0.6152347618901257,
        "average improved_aesthetic_predictor time": 0.6183845043182373
    },
    "skytnt_anime_aesthetic": {
        "average skytnt_anime_aesthetic score": 0.2086948871612549,
        "average skytnt_anime_aesthetic score normed": 0.2086948871612549,
        "average skytnt_anime_aesthetic time": 1.8350161552429198
    },
    "nsfw_detect": {
        "average nsfw_detect score": 0.008214050624519586,
        "average nsfw_detect score normed": 0.9917859493754804,
        "average nsfw_detect time": 1.353853178024292
    },
    "nsfw_detect_train": {
        "average nsfw_detect_train score": 0.0,
        "average nsfw_detect_train score normed": 1.0,
        "average nsfw_detect_train time": 0.08027291297912598
    },
    "average weighted score normed": 0.7036348149989463
}
```

data/test_images_dirs/test_images_dir_2
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](data/test_images_dirs/test_images_dir_2/0.png) | ![1.png](data/test_images_dirs/test_images_dir_2/1.png) | ![2.png](data/test_images_dirs/test_images_dir_2/2.png) | ![3.png](data/test_images_dirs/test_images_dir_2/3.png) | ![4.png](data/test_images_dirs/test_images_dir_2/4.png) | 
```json
{
    "saturation": {
        "average saturation score": 73.14979101110387,
        "average saturation score normed": 0.6587717069691541,
        "average saturation time": 0.02006239891052246
    },
    "PSNR": {
        "average PSNR score": 7.634088893999828,
        "average PSNR score normed": 0.6943166760699839,
        "average PSNR time": 0.0008485794067382812
    },
    "SSIM": {
        "average SSIM score": 0.09929010141219313,
        "average SSIM score normed": 0.5496450507060965,
        "average SSIM time": 0.03200430870056152
    },
    "variance": {
        "average variance score": 67.21171594506887,
        "average variance score normed": 0.9518829295543035,
        "average variance time": 0.0024723529815673826
    },
    "improved_aesthetic_predictor": {
        "average improved_aesthetic_predictor score": 6.158458995819092,
        "average improved_aesthetic_predictor score normed": 0.6478174253901161,
        "average improved_aesthetic_predictor time": 0.13774194717407226
    },
    "skytnt_anime_aesthetic": {
        "average skytnt_anime_aesthetic score": 0.23886614739894868,
        "average skytnt_anime_aesthetic score normed": 0.23886614739894868,
        "average skytnt_anime_aesthetic time": 2.178910160064697
    },
    "nsfw_detect": {
        "average nsfw_detect score": 0.008378814160823821,
        "average nsfw_detect score normed": 0.9916211858391761,
        "average nsfw_detect time": 1.3233444213867187
    },
    "nsfw_detect_train": {
        "average nsfw_detect_train score": 0.0,
        "average nsfw_detect_train score normed": 1.0,
        "average nsfw_detect_train time": 0.2500930309295654
    },
    "average weighted score normed": 0.7166151402409724
}
```

data/test_images_csvs/test_images_csv_1.csv
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596652262465.webp) | ![1.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596659913873.webp) | ![2.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596621611725.webp) | ![3.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596643293286.webp) | ![4.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/6674062_2112_26854026_1735193594012334015.webp) | 
```json
{
    "saturation": {
        "average saturation score": 110.68278628414751,
        "average saturation score normed": 0.7550276252166752,
        "average saturation time": 0.012196922302246093
    },
    "PSNR": {
        "average PSNR score": 7.412721149820788,
        "average PSNR score normed": 0.6889811394075319,
        "average PSNR time": 0.0006587982177734375
    },
    "SSIM": {
        "average SSIM score": -0.006803741715732356,
        "average SSIM score normed": 0.4965981291421338,
        "average SSIM time": 0.029495048522949218
    },
    "variance": {
        "average variance score": 72.05969989333389,
        "average variance score normed": 0.9557043066971325,
        "average variance time": 0.0018590927124023438
    },
    "improved_aesthetic_predictor": {
        "average improved_aesthetic_predictor score": 6.39928674697876,
        "average improved_aesthetic_predictor score normed": 0.6568607965578881,
        "average improved_aesthetic_predictor time": 0.09588418006896973
    },
    "skytnt_anime_aesthetic": {
        "average skytnt_anime_aesthetic score": 0.4729860067367554,
        "average skytnt_anime_aesthetic score normed": 0.4729860067367554,
        "average skytnt_anime_aesthetic time": 2.0671417713165283
    },
    "nsfw_detect": {
        "average nsfw_detect score": 0.004239956592209637,
        "average nsfw_detect score normed": 0.9957600434077903,
        "average nsfw_detect time": 1.2007087230682374
    },
    "nsfw_detect_train": {
        "average nsfw_detect_train score": 0.2,
        "average nsfw_detect_train score normed": 0.9,
        "average nsfw_detect_train time": 0.2710900783538818
    },
    "average weighted score normed": 0.7402397558957384
}
```
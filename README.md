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
data/input/demo/test_images_1
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](data/input/demo/test_images_1/0.png) | ![1.png](data/input/demo/test_images_1/1.png) | ![2.png](data/input/demo/test_images_1/2.png) | ![3.png](data/input/demo/test_images_1/3.png) | ![4.png](data/input/demo/test_images_1/4.png) | 
```json
{
    "saturation": {
        "average_saturation_score": 70.83403901883534,
        "average_saturation_score_normed": 0.5896600860005143,
        "average_saturation_time": 0.008478355407714844
    },
    "PSNR": {
        "average_PSNR_score": 7.5399527297575615,
        "average_PSNR_score_normed": 0.6914899355883903,
        "average_PSNR_time": 0.0008478164672851562
    },
    "SSIM": {
        "average_SSIM_score": 0.15270395908780157,
        "average_SSIM_score_normed": 0.5763519795439007,
        "average_SSIM_time": 0.03302888870239258
    },
    "variance": {
        "average_variance_score": 74.87552555096171,
        "average_variance_score_normed": 0.9558609204319042,
        "average_variance_time": 0.0002655029296875
    },
    "improved_aesthetic_predictor": {
        "average_improved_aesthetic_predictor_score": 5.3603596687316895,
        "average_improved_aesthetic_predictor_score_normed": 0.6152069520398721,
        "average_improved_aesthetic_predictor_time": 0.6193296432495117
    },
    "skytnt_anime_aesthetic": {
        "average_skytnt_anime_aesthetic_score": 0.2086948871612549,
        "average_skytnt_anime_aesthetic_score_normed": 0.2086948871612549,
        "average_skytnt_anime_aesthetic_time": 1.976353931427002
    },
    "nsfw_detect": {
        "average_nsfw_detect_score": 0.008214050624519586,
        "average_nsfw_detect_score_normed": 0.9917859493754804,
        "average_nsfw_detect_time": 1.127815580368042
    },
    "nsfw_detect_train": {
        "average_nsfw_detect_train_score": 0.0,
        "average_nsfw_detect_train_score_normed": 1.0,
        "average_nsfw_detect_train_time": 0.0810469627380371
    },
    "children_detect_train": {
        "average_children_detect_train_score": 0.4,
        "average_children_detect_train_score_normed": 0.6,
        "average_children_detect_train_time": 0.06031341552734375
    },
    "average_weighted_score_normed": 0.6921167455712574
}
```

data/input/demo/test_images_1.csv
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596652262465.webp) | ![1.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596659913873.webp) | ![2.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596621611725.webp) | ![3.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596643293286.webp) | ![4.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/6674062_2112_26854026_1735193594012334015.webp) | 
```json
{
    "saturation": {
        "average_saturation_score": 100.46739424954454,
        "average_saturation_score_normed": 0.7388593776771656,
        "average_saturation_time": 0.01535409688949585
    },
    "PSNR": {
        "average_PSNR_score": 7.627329569401652,
        "average_PSNR_score_normed": 0.6955068947368226,
        "average_PSNR_time": 0.0006707310676574707
    },
    "SSIM": {
        "average_SSIM_score": 0.017341455897924046,
        "average_SSIM_score_normed": 0.5086707279489621,
        "average_SSIM_time": 0.031845808029174805
    },
    "variance": {
        "average_variance_score": 72.04004586467033,
        "average_variance_score_normed": 0.9556720571735154,
        "average_variance_time": 0.0019645094871520996
    },
    "improved_aesthetic_predictor": {
        "average_improved_aesthetic_predictor_score": 6.4721983671188354,
        "average_improved_aesthetic_predictor_score_normed": 0.6593455872065112,
        "average_improved_aesthetic_predictor_time": 0.13754212856292725
    },
    "skytnt_anime_aesthetic": {
        "average_skytnt_anime_aesthetic_score": 0.5620258525013924,
        "average_skytnt_anime_aesthetic_score_normed": 0.5620258525013924,
        "average_skytnt_anime_aesthetic_time": 2.1099193692207336
    },
    "nsfw_detect": {
        "average_nsfw_detect_score": 0.004821326321689412,
        "average_nsfw_detect_score_normed": 0.9951786736783106,
        "average_nsfw_detect_time": 1.049147605895996
    },
    "nsfw_detect_train": {
        "average_nsfw_detect_train_score": 0.25,
        "average_nsfw_detect_train_score_normed": 0.875,
        "average_nsfw_detect_train_time": 0.2440788745880127
    },
    "children_detect_train": {
        "average_children_detect_train_score": 0.75,
        "average_children_detect_train_score_normed": 0.25,
        "average_children_detect_train_time": 0.2550063729286194
    },
    "average_weighted_score_normed": 0.6933621301025199
}
```

data/input/demo/test_images_1.txt
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/_2109_26853993_1735265741016577435.webp?x-oss-process=image/resize,w_1080/format,webp) | ![1.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/_2109_26853993_1735295392767703058.webp?x-oss-process=image/resize,w_1080/format,webp) | ![2.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/15090317_2109_26853955_1735265749151587895.webp?x-oss-process=image/resize,w_1080/format,webp) | ![3.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/15090317_2109_26853955_1735295400758778905.webp?x-oss-process=image/resize,w_1080/format,webp) | ![4.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_nsfw/14090351_2131_596325691_1735562780415595390.webp?x-oss-process=image/resize,w_1080/format,webp) | 
```json
{
    "saturation": {
        "average_saturation_score": 84.36497830374753,
        "average_saturation_score_normed": 0.7015640981998904,
        "average_saturation_time": 0.00192718505859375
    },
    "PSNR": {
        "average_PSNR_score": 8.638021136277034,
        "average_PSNR_score_normed": 0.7211339274093236,
        "average_PSNR_time": 0.0004558563232421875
    },
    "SSIM": {
        "average_SSIM_score": 0.05933760563501185,
        "average_SSIM_score_normed": 0.5296688028175058,
        "average_SSIM_time": 0.028080034255981445
    },
    "variance": {
        "average_variance_score": 54.261109903046574,
        "average_variance_score_normed": 0.9403668821287157,
        "average_variance_time": 0.00047779083251953125
    },
    "improved_aesthetic_predictor": {
        "average_improved_aesthetic_predictor_score": 6.682973194122314,
        "average_improved_aesthetic_predictor_score_normed": 0.6667610610557777,
        "average_improved_aesthetic_predictor_time": 0.02932314872741699
    },
    "skytnt_anime_aesthetic": {
        "average_skytnt_anime_aesthetic_score": 0.8005118131637573,
        "average_skytnt_anime_aesthetic_score_normed": 0.8005118131637573,
        "average_skytnt_anime_aesthetic_time": 1.6837124824523926
    },
    "nsfw_detect": {
        "average_nsfw_detect_score": 0.005184800742426887,
        "average_nsfw_detect_score_normed": 0.9948151992575731,
        "average_nsfw_detect_time": 0.7770517349243165
    },
    "nsfw_detect_train": {
        "average_nsfw_detect_train_score": 0.0,
        "average_nsfw_detect_train_score_normed": 1.0,
        "average_nsfw_detect_train_time": 0.07996702194213867
    },
    "children_detect_train": {
        "average_children_detect_train_score": 0.8,
        "average_children_detect_train_score_normed": 0.2,
        "average_children_detect_train_time": 0.08295822143554688
    },
    "average_weighted_score_normed": 0.7283135315591714
}
```
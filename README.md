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
Revise parameters ```metric_params``` and ```test_images_dirs_or_csvs``` in the ```config.py```
```
python app.py
```
## Result
The results include the following:
### 1. folder ```xlsx``` or ```csv```
The normalized scores for each metric and normalized weighted scores for each image in each file or folder in parameter ```test_images_dirs_or_csvs```.
![csv.png](data/asset/csv.png)
### 2. folder ```png```
The plots according to the column_titles ```"nsfw_detect_train_score_normed"``` and ```"children_detect_train_score_normed"```, classify by ```type``` in each plot. You can see the proportion of different normalized scores.
![png.png](data/asset/png.png)
### 3. folder ```html```
The html report of normalized scores for each metric and normalized weighted scores for each image in each file or folder in parameter ```test_images_dirs_or_csvs```. You can filter the ```type```, order different metrics in ascending and descending order, and view the image.
![html.png](data/asset/html.png)
### 4. folder ```txt```
During the evaluation process, if the image path or url cannot be loaded, this image will be skipped, and its ```index```, ```img_path_or_url```, ```type``` will be recorded in a txt file.
### 5. folder ```json```
The parameter ```metric_params``` and the normalized average scores for each metric and normalized weighted average scores for each image in each file or folder in parameter ```test_images_dirs_or_csvs```. Some of the information is as follows:

data/input/demo/test_images_1
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](data/input/demo/test_images_1/0.png) | ![1.png](data/input/demo/test_images_1/1.png) | ![2.png](data/input/demo/test_images_1/2.png) | ![3.png](data/input/demo/test_images_1/3.png) | ![4.png](data/input/demo/test_images_1/4.png) | 
```json
{
    "saturation": {
        "average_saturation_score": 70.83403901883534,
        "average_saturation_score_normed": 0.5896600860005143,
        "average_saturation_time": 0.010272836685180664
    },
    "PSNR": {
        "average_PSNR_score": 7.5399527297575615,
        "average_PSNR_score_normed": 0.6914899355883903,
        "average_PSNR_time": 0.0006290912628173828
    },
    "SSIM": {
        "average_SSIM_score": 0.15270395908780157,
        "average_SSIM_score_normed": 0.5763519795439007,
        "average_SSIM_time": 0.03200154304504395
    },
    "variance": {
        "average_variance_score": 74.87552555096171,
        "average_variance_score_normed": 0.9558609204319042,
        "average_variance_time": 0.00024871826171875
    },
    "improved_aesthetic_predictor": {
        "average_improved_aesthetic_predictor_score": 5.360665607452392,
        "average_improved_aesthetic_predictor_score_normed": 0.6152192673438415,
        "average_improved_aesthetic_predictor_time": 0.5425155639648438
    },
    "skytnt_anime_aesthetic": {
        "average_skytnt_anime_aesthetic_score": 0.2086948871612549,
        "average_skytnt_anime_aesthetic_score_normed": 0.2086948871612549,
        "average_skytnt_anime_aesthetic_time": 1.8137116432189941
    },
    "nsfw_detect": {
        "average_nsfw_detect_score": 0.008214050624519586,
        "average_nsfw_detect_score_normed": 0.9917859493754804,
        "average_nsfw_detect_time": 1.008475160598755
    },
    "nsfw_detect_train": {
        "average_nsfw_detect_train_score": 0.0,
        "average_nsfw_detect_train_score_normed": 1.0,
        "average_nsfw_detect_train_time": 0.07926111221313477
    },
    "children_detect_train": {
        "average_children_detect_train_score": 0.4,
        "average_children_detect_train_score_normed": 0.6,
        "average_children_detect_train_time": 0.04039134979248047
    },
    "average_weighted_score_normed": 0.6921181139383651
}
```

data/input/demo/test_images_1.csv
|  |  |  |  |  |
|------|------|------|------|------|
| ![0.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596652262465.webp) | ![1.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596659913873.webp) | ![2.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596621611725.webp) | ![3.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc_fast/6153196_2130_26854028_1735193596643293286.webp) | ![4.png](https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/6674062_2112_26854026_1735193594012334015.webp) | 
```json
{
    "saturation": {
        "average_saturation_score": 110.68278628414751,
        "average_saturation_score_normed": 0.7550276252166752,
        "average_saturation_time": 0.012058877944946289
    },
    "PSNR": {
        "average_PSNR_score": 7.412721149820788,
        "average_PSNR_score_normed": 0.6889811394075319,
        "average_PSNR_time": 0.0007004737854003906
    },
    "SSIM": {
        "average_SSIM_score": -0.006803741715732356,
        "average_SSIM_score_normed": 0.4965981291421338,
        "average_SSIM_time": 0.028401279449462892
    },
    "variance": {
        "average_variance_score": 72.05969989333389,
        "average_variance_score_normed": 0.9557043066971325,
        "average_variance_time": 0.001936054229736328
    },
    "improved_aesthetic_predictor": {
        "average_improved_aesthetic_predictor_score": 6.398706531524658,
        "average_improved_aesthetic_predictor_score_normed": 0.6568395915148125,
        "average_improved_aesthetic_predictor_time": 0.09810209274291992
    },
    "skytnt_anime_aesthetic": {
        "average_skytnt_anime_aesthetic_score": 0.4729860067367554,
        "average_skytnt_anime_aesthetic_score_normed": 0.4729860067367554,
        "average_skytnt_anime_aesthetic_time": 2.116146373748779
    },
    "nsfw_detect": {
        "average_nsfw_detect_score": 0.004239956592209637,
        "average_nsfw_detect_score_normed": 0.9957600434077903,
        "average_nsfw_detect_time": 0.996024227142334
    },
    "nsfw_detect_train": {
        "average_nsfw_detect_train_score": 0.2,
        "average_nsfw_detect_train_score_normed": 0.9,
        "average_nsfw_detect_train_time": 0.26923203468322754
    },
    "children_detect_train": {
        "average_children_detect_train_score": 0.6,
        "average_children_detect_train_score_normed": 0.4,
        "average_children_detect_train_time": 0.24373717308044435
    },
    "average_weighted_score_normed": 0.7024329824580924
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
        "average_saturation_time": 0.0019573688507080076
    },
    "PSNR": {
        "average_PSNR_score": 8.638021136277034,
        "average_PSNR_score_normed": 0.7211339274093236,
        "average_PSNR_time": 0.0004626274108886719
    },
    "SSIM": {
        "average_SSIM_score": 0.05933760563501185,
        "average_SSIM_score_normed": 0.5296688028175058,
        "average_SSIM_time": 0.02807588577270508
    },
    "variance": {
        "average_variance_score": 54.261109903046574,
        "average_variance_score_normed": 0.9403668821287157,
        "average_variance_time": 0.0005202770233154296
    },
    "improved_aesthetic_predictor": {
        "average_improved_aesthetic_predictor_score": 6.683196544647217,
        "average_improved_aesthetic_predictor_score_normed": 0.6667690308621647,
        "average_improved_aesthetic_predictor_time": 0.027332258224487305
    },
    "skytnt_anime_aesthetic": {
        "average_skytnt_anime_aesthetic_score": 0.8005118131637573,
        "average_skytnt_anime_aesthetic_score_normed": 0.8005118131637573,
        "average_skytnt_anime_aesthetic_time": 1.7396596908569335
    },
    "nsfw_detect": {
        "average_nsfw_detect_score": 0.005184800742426887,
        "average_nsfw_detect_score_normed": 0.9948151992575731,
        "average_nsfw_detect_time": 0.9324620246887207
    },
    "nsfw_detect_train": {
        "average_nsfw_detect_train_score": 0.0,
        "average_nsfw_detect_train_score_normed": 1.0,
        "average_nsfw_detect_train_time": 0.06197032928466797
    },
    "children_detect_train": {
        "average_children_detect_train_score": 0.8,
        "average_children_detect_train_score_normed": 0.2,
        "average_children_detect_train_time": 0.09664778709411621
    },
    "average_weighted_score_normed": 0.7283144170932145
}
```
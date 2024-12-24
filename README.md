# ImageEvaluation
ImageEvaluation: A image indicator evaluation pipeline
## Install
```
git clone https://github.com/RobertLau666/ImageEvaluation.git

conda create -n imageevaluation python=3.10
conda activate imagetagger
pip install --upgrade pip

cd ImageEvaluation
pip install -r requirements.txt
```
## Models
1. download models from [improved-aesthetic-predictor](!https://github.com/christophschuhmann/improved-aesthetic-predictor), [skytnt_anime_aesthetic](!https://huggingface.co/skytnt/anime-aesthetic/tree/main), [TostAI/nsfw-image-detection-large](!https://huggingface.co/TostAI/nsfw-image-detection-large/tree/main).
2. place the model folder ```images_evaluation_models``` in the same level of directory as the project folder, the directory structure is as follows:
```
| ImageEvaluation/
| images_evaluation_models/
|---- improved_aesthetic_predictor_models/
|-------- ViT-L-14.pt
|-------- ava+logos-l14-linearMSE.pth
|-------- ava+logos-l14-reluMSE.pth
|-------- sac+logos+ava1-l14-linearMSE.pth
|---- nsfw_detector_models/
|-------- nsfw-image-detection-large
|---- skytnt_anime_aesthetic_models/
|-------- model.ckpt
|-------- model.onnx
```
## Run
```
python app.py
```
import os
import numpy as np
from PIL import Image
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.image_grounding.kosmos2 import *
import torch
from pytorch_fid import fid_score
from transformers import CLIPProcessor, CLIPModel
import cv2
import glob
from torch.cuda import amp
from utils.anime_seg.anime_seg import AnimeSegmentation, net_names
import json
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from textblob import TextBlob
from skimage.metrics import structural_similarity as ssim
from num2words import num2words


clip_model_path = "openai/clip-vit-base-patch32"
device = "cuda:1" if torch.cuda.is_available() else "cpu"

real_images_folder = 'images/comics'
generated_images_folder = 'images/panels'

evaluate_task_template_path = 'templates/evaluate-task-template.json'

def data_preprocess(storyboard):
    description_list = []
    image_prompt_list =[]
    dialogue_list = []
    image_list = []
    for i, (panel_key, panel_info) in enumerate(storyboard['panels'].items()):
        description_list.append(panel_info["panel_description"])
        image_prompt_list.append(panel_info["character_description"])
        dialogue_list.append(panel_info["dialogue_list"][0]["content"].split(' - ')[-1])
        image_list.append(Image.open(panel_info["img_path"]))
    comic_image = Image.open(f'{generated_images_folder.split("/")[0]}/comics/comic.png')

    return storyboard, description_list, image_prompt_list, dialogue_list, image_list, comic_image

# def extract_text_from_image(image_path):
#     img = Image.open(image_path)
#     text = pytesseract.image_to_string(img)
#     return text

def evaluate_image_with_clip(image_path):
    # 使用CLIP或其他图像识别模型对图像进行评估
    # 返回图像的评价结果，例如画风、技巧、内容等
    pass

def overall_evaluation(text_evaluation, image_evaluation):
    # 根据文本和图像的评估结果，进行综合评估
    # 返回最终的评价结果
    pass

def get_mask(model, input_img, use_amp=True, s=640):
    h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
    ph, pw = s - h, s - w
    tmpImg = np.zeros([s, s, 3], dtype=np.float32)
    tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred.cpu().numpy().transpose((1, 2, 0)), (w0, h0))[:, :, np.newaxis]
        return pred

def matting_process(image_list):
    model = AnimeSegmentation.try_load('isnet_is', './models/anime_seg/isnetis.ckpt', 'cpu', img_size=1024)
    model.eval()

    matted_image_list = []
    for img in image_list:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = get_mask(model, img, use_amp=True, s=1024)

        white_mask = np.ones_like(mask) * 255
        img = np.concatenate((mask * img + (white_mask * (1 - mask)), mask * 255),axis=2).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))

        matted_image_list.append(img)

    return matted_image_list

def extract_image_feature(image):
    model = CLIPModel.from_pretrained(clip_model_path)
    processor = CLIPProcessor.from_pretrained(clip_model_path)
    model.to(device)

    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs.to(device)

    with torch.no_grad():
        image_feature = model.get_image_features(**inputs) # torch.Size([1, 512])

    return image_feature

def calculate_multi_feature_consistency(feature_list):
    num_features = len(feature_list) # range of num_features: 1~4
    
    total_similarity = 0.0
    
    if num_features == 1:
        total_similarity = 1.0
        average_similarity = total_similarity
    else:
        for i in range(num_features):
            for j in range(i + 1, num_features):
                similarity = torch.cosine_similarity(feature_list[i], feature_list[j], dim=1)
                total_similarity += similarity
        average_similarity = total_similarity / (num_features * (num_features - 1) / 2)
        average_similarity = average_similarity.item()

    multi_feature_consistency_score = average_similarity

    return multi_feature_consistency_score

def calculate_character_consistency(description_list, image_prompt_list, image_list, comic_image):
    # 1. 先扣取panel图片中的人物，背景换成白色
    image_list = matting_process(image_list) # return: PIL

    # 2. 按照image_prompt_list划分成不同的人物组
    grouped_indices = {}
    for index, item in enumerate(image_prompt_list):
        if item not in grouped_indices:
            grouped_indices[item] = [index]
        else:
            grouped_indices[item].append(index)

    multi_feature_consistency = []
    for key, value in grouped_indices.items():
        single_character_features = []
        for v in value:
            # 3. 提取每组图片的特征
            single_character_feature = extract_image_feature(image_list[v])
            single_character_features.append(single_character_feature)
        # 4. 计算每组图片的相似度
        single_character_consistency = calculate_multi_feature_consistency(single_character_features)
        multi_feature_consistency.append(single_character_consistency)

    character_consistency_score = sum(multi_feature_consistency) / len(multi_feature_consistency)
    print(f'character_consistency_score: {character_consistency_score}')

    return character_consistency_score

def calculate_panel_coherent(description_list, image_prompt_list, image_list, comic_image):

    panel_coherent_score = 1.0

    print(f'content_coherent_score: {panel_coherent_score}')

    return panel_coherent_score

def calculate_story_sentiment(story_text):
    # 使用TextBlob进行情感分析
    blob = TextBlob(story_text)

    story_sentiment_score = blob.sentiment.polarity # (-1~1)
    story_sentiment_score = (story_sentiment_score + 1) / 2 # (0~1)
    print(f'story_sentiment_score: {story_sentiment_score}')

    return story_sentiment_score

def calculate_content_evaluate(description_list, image_prompt_list, image_list, comic_image, storyboard):
    content_evaluate_score_list = []

    panel_coherent_score = calculate_panel_coherent(description_list, image_prompt_list, image_list, comic_image)
    content_evaluate_score_list.append(panel_coherent_score)
    story_sentiment_score = calculate_story_sentiment(storyboard["raw_story"])
    content_evaluate_score_list.append(story_sentiment_score)

    content_evaluate_score = sum(content_evaluate_score_list) / len(content_evaluate_score_list)
    print(f'content_evaluate_score: {content_evaluate_score}')

    return content_evaluate_score

def calculate_PCA_similarity(image_list):
    '''
    range: 0~1, but almost in 0.3~0.5, 0.3: all diff, 0.5: all same
    '''
    images = []
    for img in image_list:
        img = img.resize((100, 100))  # 调整图片大小为100x100
        img_array = np.array(img)
        images.append(img_array)

    # 将图片转换为特征矩阵
    features = np.reshape(images, (len(images), -1))

    # 创建PCA对象并指定主成分数量
    n_components = len(image_list)  # 指定要保留的主成分数量
    pca = PCA(n_components=n_components)

    # 执行PCA降维
    reduced_features = pca.fit_transform(features)

    # 归一化特征矩阵
    normalized_features = normalize(reduced_features)

    # 计算特征矩阵之间的余弦相似度
    similarity_matrix = cosine_similarity(normalized_features)
    similarity_matrix = (similarity_matrix + 1) / 2

    # 计算右上三角形部分
    upper_triangle = np.triu(similarity_matrix, k=1)

    # 计算右上三角形部分的和
    upper_triangle_sum = np.sum(upper_triangle)

    # 计算右上角元素的个数
    n = len(upper_triangle)
    count = n * (n - 1) / 2

    # 计算右上角元素的平均值
    PCA_similarity_score = upper_triangle_sum / count

    return PCA_similarity_score

def calculate_style_unity(description_list, image_prompt_list, image_list, comic_image):
    style_unity_score_list = []
    # a. 直方图比较：通过计算每张图片的颜色直方图，然后比较直方图之间的相似度。如果直方图之间的差异较小，则说明画风比较统一。

    # b. 特征提取与聚类：使用计算机视觉中的特征提取算法，如卷积神经网络（CNN）或预训练的图像分类模型，提取每张图片的特征向量，然后将这些特征向量进行聚类分析。如果聚类结果中的不同类别对应于不同的画风，则说明画风不太统一。

    # c. 图像生成对抗网络（GAN）：使用生成对抗网络来生成新的图片，使其与原始图片具有相同的画风。通过对生成的图片进行视觉评估，判断是否与原始图片的画风保持一致。

    # d. 风格迁移：使用图像风格迁移技术，将多张图片的风格迁移到一张参考图片上，然后观察生成的图片是否具有统一的画风。

    # e. 主成分分析（PCA）：将每张图片转换为特征空间，在特征空间中计算主成分，并比较各个图片的主成分之间的相似性。如果主成分相似度较高，则说明画风比较统一。
    PCA_similarity_score = calculate_PCA_similarity(image_list)
    style_unity_score_list.append(PCA_similarity_score)

    style_unity_score = sum(style_unity_score_list) / len(style_unity_score_list)
    print(f'style_unity_score: {style_unity_score}')

    return style_unity_score

def calculate_FID_score(real_images_folder, generated_images_folder):
    '''
    FID（Frechet Inception Distance）分数是一种用于衡量生成模型与真实数据集之间相似性的指标，它是通过计算生成的样本与真实样本在Inception网络中特征表示上的差异程度来计算得出的。FID分数越低，表示生成的样本与真实样本之间的差异越小，生成模型的性能越好。
    '''
    # inception_model = torchvision.models.inception_v3(pretrained=True)
    FID_ = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], # (0~INF)
                                                    batch_size=1,
                                                    device='cpu',
                                                    dims=2048, num_workers=0,
                                                    )
    FID_score = 1 - FID_ / 1000.0 # (0~1)
    print(f'FID_score: {FID_score}')

    return FID_score

def calculate_PSNR_SSIM_score(generated_images_folder):
    PSNR_score_list = []
    SSIM_score_list = []

    image_list = glob.glob(os.path.join(generated_images_folder, 'panel_*.png'))
    reference_image = cv2.imread(image_list[0])

    psnr_standard = {
        "excellent": 1.0,
        "good": 0.8,
        "bad": 0.5,
        "unacceptable": 0.2
    }

    for image in image_list[1:]:
        test_image = cv2.imread(image)
        
        # PSNR
        psnr_ = cv2.PSNR(reference_image, test_image) # (0~INF)
        if psnr_ > 40:
            psnr_score = psnr_standard["excellent"]
        elif 40 >= psnr_ > 30:
            psnr_score = psnr_standard["good"]
        elif 30 >= psnr_ > 20:
            psnr_score = psnr_standard["bad"]
        elif 20 >= psnr_:
            psnr_score = psnr_standard["unacceptable"]
        PSNR_score_list.append(psnr_score)

        # SSIM
        ssim_ = ssim(reference_image, test_image, win_size=3, data_range=255, multichannel=True) # (-1~1)
        ssim_score = (ssim_ + 1) / 2
        SSIM_score_list.append(ssim_score)

    PSNR_score = sum(PSNR_score_list) / len(PSNR_score_list) # (0~1)
    SSIM_score = sum(SSIM_score_list) / len(SSIM_score_list) # (0~1)
    print(f'PSNR_score: {PSNR_score}\nSSIM_score: {SSIM_score}')

    return PSNR_score, SSIM_score

def calculate_variance_score(generated_images_folder):
    variance_score_list = []
    image_list = glob.glob(os.path.join(generated_images_folder, 'panel_*.png'))

    for image in image_list:
        test_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        variance = cv2.meanStdDev(test_image)[1] # (0～INF)
        variance = variance[0][0]
        variance_score = 1 - variance / 1000.0
        variance_score_list.append(variance_score)
    variance_score = sum(variance_score_list) / len(variance_score_list) # (0~1)
    print(f'variance_score: {variance_score}')

    return variance_score

def calculate_image_quality(real_images_folder, generated_images_folder):
    image_quality_score_list = []

    # a. FID
    FID_score = calculate_FID_score(real_images_folder, generated_images_folder)
    image_quality_score_list.append(FID_score)

    # b. PSNR & SSIM
    PSNR_score, SSIM_score = calculate_PSNR_SSIM_score(generated_images_folder)
    image_quality_score_list.extend([PSNR_score, SSIM_score])

    # d. LPIPS

    # e. KID

    # f. variance：衡量模糊程度
    variance_score = calculate_variance_score(generated_images_folder)
    image_quality_score_list.append(variance_score)

    image_quality_score = sum(image_quality_score_list) / len(image_quality_score_list)
    print(f'image_quality_score: {image_quality_score}')

    return image_quality_score

def calculate_cosine_similarity(text1, text2):
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer()

    # 将文本转换为TF-IDF向量
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    cosine_similarity_score = cosine_sim[0][0]
    print(f'cosine_similarity_score: {cosine_similarity_score}')

    return cosine_similarity_score

def calculate_jaccard_similarity(text1, text2):
    # 将文本转换为词集合（以空格为分隔符）
    set1 = set(text1.split())
    set2 = set(text2.split())
    
    # 计算共有词数和总词数
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    
    # 计算Jaccard相似度
    jaccard_similarity_score = intersection / union
    print(f'jaccard_similarity_score: {jaccard_similarity_score}')
    
    return jaccard_similarity_score

def calculate_reduction_degree(kosmos2_plain_text, fuse_prompt):
    reduction_degree_score_list = []

    # a. 余弦相似度
    reduction_degree_score_list.append(calculate_cosine_similarity(kosmos2_plain_text, fuse_prompt))

    # b. difflib.SequenceMatcher：基于最长公共子序列（Longest Common Subsequence, LCS）的相似度计算，用于测量字符串之间的相似性。它考虑了字符的顺序和相对位置。
    difflib_sequenceMatcher_score = difflib.SequenceMatcher(None, kosmos2_plain_text, fuse_prompt).ratio()
    reduction_degree_score_list.append(difflib_sequenceMatcher_score)
    print(f'difflib_sequenceMatcher_score: {difflib_sequenceMatcher_score}')

    # c. Jaccard相似度：Jaccard相似性或联合上的交集定义为交叉的大小除以两个联合的大小。基于集合的相似度计算，适用于比较文本或元素集合的相似性。它忽略了元素的顺序，只关注元素的存在与否。
    reduction_degree_score_list.append(calculate_jaccard_similarity(kosmos2_plain_text, fuse_prompt))
    
    # d. 编辑距离算法：

    # e. SimHash算法：

    # f. 抽取名词后比较相似度

    reduction_degree_score = sum(reduction_degree_score_list) / len(reduction_degree_score_list)

    return reduction_degree_score

def calculate_CLIP_similarity(image, text):
    model = CLIPModel.from_pretrained(clip_model_path)
    processor = CLIPProcessor.from_pretrained(clip_model_path)

    inputs = processor(text=[text, "this is a photo of image"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    CLIP_similarity_score = probs[0][0].item()
    print(f'CLIP_similarity_score: {CLIP_similarity_score}')
    
    return CLIP_similarity_score

def calculate_matching_degree(image, fuse_prompt):
    matching_degree_score_list = []

    # a. CLIP
    CLIP_similarity_score = calculate_CLIP_similarity(image, fuse_prompt)
    matching_degree_score_list.append(CLIP_similarity_score)

    # b. 

    matching_degree_score = sum(matching_degree_score_list) / len(matching_degree_score_list)

    return matching_degree_score

def calculate_score(storyboard):
    storyboard, description_list, image_prompt_list, dialogue_list, image_list, comic_image = data_preprocess(storyboard)

    # 0. load json
    with open(evaluate_task_template_path, 'r') as f:
        records = json.load(f)

    # 1. perform calculate tasks
    for key, value in records.items():
        print(f"\n{'='*3}performing task: {key}, {value}")

        if records[key]["task"] == "calculate panels number":
            panel_count = len(storyboard['panels'])
            panel_count_word = num2words(panel_count)
            annotated_image, colored_text, filtered_entities = generate_predictions(comic_image, records[key]["question"])
            print(f'kosmos2_colored_text: {colored_text}')

            records[key]["return"] = ''.join(text[0] for text in colored_text[1:])
            if panel_count_word in records[key]["return"]:
                records[key]["score"] = 1.0
            else:
                records[key]["score"] = 1.0

        elif records[key]["task"] == "calculate character consistency":
            character_consistency_score = calculate_character_consistency(description_list, image_prompt_list, image_list, comic_image)
            records[key]['score'] = character_consistency_score

        elif records[key]["task"] == "calculate content evaluate":
            content_evaluate_score = calculate_content_evaluate(description_list, image_prompt_list, image_list, comic_image, storyboard)
            records[key]['score'] = content_evaluate_score
        
        elif records[key]["task"] == "calculate style unity":
            style_unity_score = calculate_style_unity(description_list, image_prompt_list, image_list, comic_image)
            records[key]['score'] = style_unity_score

        elif records[key]["task"] == "calculate image quality":
            image_quality_score = calculate_image_quality(real_images_folder, generated_images_folder)
            records[key]['score'] = image_quality_score

        elif records[key]["task"] == "calculate reduction degree":
            score_list = []
            for index, (description, image_prompt, image) in enumerate(zip(description_list, image_prompt_list, image_list)):
                records[key]["panels"][f"panel{index+1}"] = {}
                annotated_image, colored_text, filtered_entities = generate_predictions(image, records[key]["question"])
                kosmos2_plain_text = ''.join(text[0] for text in colored_text[1:])
                fuse_prompt = ', '.join([description, image_prompt])
                print(f'panel: {index+1}\nkosmos2_colored_text: {colored_text}\nkosmos2_plain_text: {kosmos2_plain_text}\nfuse_prompt: {fuse_prompt}')
                reduction_degree_score = calculate_reduction_degree(kosmos2_plain_text, fuse_prompt)
                records[key]["panels"][f"panel{index+1}"]["return"] = reduction_degree_score
                records[key]["panels"][f"panel{index+1}"]["score"] = reduction_degree_score
                score_list.append(records[key]["panels"][f"panel{index+1}"]["score"])
            records[key]["score"] = sum(score_list) / len(score_list)
            print(f'reduction_degree_score: {records[key]["score"]}')

        elif records[key]["task"] == "calculate matching degree":
            score_list = []
            for index, (description, image_prompt, image) in enumerate(zip(description_list, image_prompt_list, image_list)):
                records[key]["panels"][f"panel{index+1}"] = {}
                matching_degree_score = calculate_matching_degree(image, fuse_prompt)
                records[key]["panels"][f"panel{index+1}"]["return"] = matching_degree_score
                records[key]["panels"][f"panel{index+1}"]["score"] = matching_degree_score
                score_list.append(records[key]["panels"][f"panel{index+1}"]["score"])
            records[key]["score"] = sum(score_list) / len(score_list)
            print(f'matching_degree_score: {records[key]["score"]}')

    # 2. calculate final_score
    sum_score = 0
    sum_weight = 0
    for key, value in records.items():
        sum_score += value["score"] * value["weight"]
        sum_weight += value["weight"]
    final_score = sum_score / sum_weight
    records["total score"] = final_score

    # 3. add extra info into records
    records["single_panel_time (s)"] = storyboard["single_panel_time (s)"]
    records["all_panel_time (s)"] = storyboard["all_panel_time (s)"]
    records["story title"] = storyboard["title"]
    records["evaluate time"] = datetime.datetime.now()

    # 4. TODO: save records as .json file

    return final_score, records

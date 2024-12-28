import os
import json
from tqdm import tqdm
import config
from metrics import *


class ImageEvaluation():
    def __init__(self):
        self.get_use_model_names()
        self.init_use_models()

    def get_use_model_names(self):
        self.use_model_names = [model_name for model_name, model_param in config.model_params.items() if model_param["use"]]
        print("use_model_names: ", self.use_model_names)
    
    def init_use_models(self):
        print("Initializing use models...")
        if "improved_aesthetic_predictor" in self.use_model_names:
            self.improved_aesthetic_predictor_model = ImprovedAestheticPredictor()
        if "skytnt_anime_aesthetic" in self.use_model_names:
            self.skytnt_anime_aesthetic_model = SkytntAnimeAesthetic()
        if "nsfw_detect" in self.use_model_names:
            self.nsfw_detect_model = API_ViT_v3(model_path=config.model_params["nsfw_detect"]["nsfw_detect_model_path"])
        if "nsfw_detect_train" in self.use_model_names:
            self.nsfw_detect_model_train = NSFWSelfTrainBinary(model_url=config.model_params["nsfw_detect_train"]["nsfw_detect_train_model_url"])

    def get_img_paths_or_urls(self, images_dir_or_csv):
        img_paths_or_urls = []
        if images_dir_or_csv.endswith(('.csv', '.xlsx')):
            self.is_excel_file = True
            all_files = read_excel(images_dir_or_csv, begin_r=0, end_r=-1, url_c=6)
            img_paths_or_urls = get_img_urls(all_files)
        else:
            self.is_excel_file = False
            img_paths_or_urls = [os.path.join(images_dir_or_csv, img_name) for img_name in sorted(os.listdir(images_dir_or_csv))]
        return img_paths_or_urls

    def get_img_numpy(self, img_path_or_url):
        img_numpy = get_image_array_from_img_url(img_path_or_url) if self.is_excel_file else cv2.imread(img_path_or_url)
        return img_numpy

    def __call__(self, images_dir_or_csv):
        img_paths_or_urls = self.get_img_paths_or_urls(images_dir_or_csv)
        
        save_as_excel = config.save_as_excel
        if save_as_excel:
            columns = ["img_path_or_url"] + self.use_model_names
            output_file = f'{config.xlsx_dir}/result_{os.path.splitext(os.path.basename(images_dir_or_csv))[0]}_{get_formatted_current_time()}.xlsx'
            if not os.path.exists(output_file):
                df = pd.DataFrame(columns=columns)
                df.to_excel(output_file, index=False)

        if "saturation" in self.use_model_names:
            saturation_scores = []
        if "PSNR" in self.use_model_names:
            PSNR_scores = []
        if "SSIM" in self.use_model_names:
            SSIM_scores = []
        if "variance" in self.use_model_names:
            variance_scores = []
        if "improved_aesthetic_predictor" in self.use_model_names:
            improved_aesthetic_predictor_scores = []
        if "skytnt_anime_aesthetic" in self.use_model_names:
            skytnt_anime_aesthetic_scores = []
        if "nsfw_detect" in self.use_model_names:
            nsfw_detect_scores = []
        if "nsfw_detect_train" in self.use_model_names:
            nsfw_detect_train_scores = []
        
        # 可以单图评估的
        for img_path_or_url in tqdm(img_paths_or_urls):
            img_numpy = self.get_img_numpy(img_path_or_url)
            if img_numpy is None:
                continue
            if "saturation" in self.use_model_names:
                saturation_score = calculate_saturation_score(img_numpy)
                saturation_scores.append(saturation_score)
            if "PSNR" in self.use_model_names:
                PSNR_score = calculate_PSNR_score(img_numpy)
                PSNR_scores.append(PSNR_score)
            if "SSIM" in self.use_model_names:
                SSIM_score = calculate_SSIM_score(img_numpy)
                SSIM_scores.append(SSIM_score)
            if "variance" in self.use_model_names:
                variance_score = calculate_variance_score(img_numpy)
                variance_scores.append(variance_score)
            if "improved_aesthetic_predictor" in self.use_model_names:
                improved_aesthetic_predictor_score = self.improved_aesthetic_predictor_model(img_numpy)
                improved_aesthetic_predictor_scores.append(improved_aesthetic_predictor_score)
            if "skytnt_anime_aesthetic" in self.use_model_names:
                skytnt_anime_aesthetic_score = self.skytnt_anime_aesthetic_model(img_numpy)
                skytnt_anime_aesthetic_scores.append(skytnt_anime_aesthetic_score)
            if "nsfw_detect" in self.use_model_names:
                is_nsfw_img, nsfw_detect_score = self.nsfw_detect_model(img_numpy)
                nsfw_detect_scores.append(nsfw_detect_score)
            if "nsfw_detect_train" in self.use_model_names:
                nsfw_detect_train_score = self.nsfw_detect_model_train(img_numpy)
                nsfw_detect_train_scores.append(nsfw_detect_train_score)
        
            img_result = {
                "img_path_or_url": img_path_or_url,
            }
            for column in columns[1:]:
                exec(f'img_result["{column}"] = {column}_score')

            if save_as_excel:
                single_df = pd.DataFrame([img_result])
                with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                    existing_df = pd.read_excel(output_file)
                    # 去除全 NA 的列
                    existing_df = existing_df.dropna(axis=1, how='all')
                    single_df = single_df.dropna(axis=1, how='all')
                    result_df = pd.concat([existing_df, single_df], ignore_index=True)
                    result_df.to_excel(writer, index=False)
        
        # # 必须组图评估的
        # if "FID" in self.use_model_names:
        #     FID_score = calculate_FID_score(img_paths_or_urls)

        score_json = {}
        for use_model_name in self.use_model_names:
            if use_model_name == "nsfw_detect_train":
                average_nsfw_detect_train_score = 0
                for nsfw_detect_train_score in nsfw_detect_train_scores:
                    average_nsfw_detect_train_score += (1 if int(nsfw_detect_train_score) >= 1 else 0)
                score_json[f"average {use_model_name} score"] = 1 - average_nsfw_detect_train_score / len(nsfw_detect_train_scores)
            else:
                exec(f'score_json["average {use_model_name} score"] = sum({use_model_name}_scores) / len({use_model_name}_scores)')

        average_weighted_score = sum([config.model_params[use_model_name]["score_weight"] * score_json[f"average {use_model_name} score"] for use_model_name in self.use_model_names]) / sum([config.model_params[use_model_name]["score_weight"] for use_model_name in self.use_model_names])
        score_json["average weighted score"] = average_weighted_score

        return score_json


if __name__ == "__main__":
    img_eval = ImageEvaluation()

    record_dict_path = f"{config.json_dir}/result_{get_formatted_current_time()}.json"
    if not os.path.exists(record_dict_path):
        record_dict = {}
    else:
        try:
            with open(record_dict_path, 'r', encoding='utf-8') as f:
                record_dict = json.load(f)
        except json.decoder.JSONDecodeError as e:
            record_dict = {}
            
    for test_images_dir_or_csv in tqdm(config.test_images_dirs_or_csvs):
        print(f"\nProcessing {test_images_dir_or_csv} ...")
        score_json = img_eval(test_images_dir_or_csv)
        record_dict[test_images_dir_or_csv] = score_json
        
    with open(record_dict_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(record_dict, indent=4, ensure_ascii=False))
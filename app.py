import os
import json
from tqdm import tqdm
import config
from metrics import *
import time


class ImageEvaluation():
    def __init__(self):
        self.get_use_metric_names()
        self.init_use_metrics()

    def get_use_metric_names(self):
        self.use_metric_names = []
        self.use_metric_forms = {}
        for metric_name, metric_param in config.metric_params.items():
            if metric_param["use"]:
                self.use_metric_names.append(metric_name)
            if metric_param["use_form"] not in self.use_metric_forms:
                self.use_metric_forms[metric_param["use_form"]] = [metric_name]
            else:
                self.use_metric_forms[metric_param["use_form"]].append(metric_name)

        print("use_metric_names: ", self.use_metric_names)
        print("use_metric_forms: ", self.use_metric_forms)
    
    def init_use_metrics(self):
        print("Initializing use metrics...")
        if "improved_aesthetic_predictor" in self.use_metric_names:
            self.improved_aesthetic_predictor_model = ImprovedAestheticPredictor(model_path=config.metric_params["improved_aesthetic_predictor"]["improved_aesthetic_predictor_model_path"])
        if "skytnt_anime_aesthetic" in self.use_metric_names:
            self.skytnt_anime_aesthetic_model = SkytntAnimeAesthetic(model_path=config.metric_params["skytnt_anime_aesthetic"]["skytnt_anime_aesthetic_model_path"])
        if "nsfw_detect" in self.use_metric_names:
            self.nsfw_detect_model = API_ViT_v3(model_path=config.metric_params["nsfw_detect"]["nsfw_detect_model_path"])
        if "nsfw_detect_train" in self.use_metric_names:
            self.nsfw_detect_train_model = NSFWSelfTrainBinary(model_url=config.metric_params["nsfw_detect_train"]["nsfw_detect_train_model_url"])

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
        
        # save_as_excel
        column_titles = ["img_path_or_url"] + [f"{use_metric_name}_score_normed" for use_metric_name in self.use_metric_names]
        result_excel_path = f'{config.xlsx_dir}/result_{os.path.splitext(os.path.basename(images_dir_or_csv))[0]}_{get_formatted_current_time()}.xlsx'
        print(f"images metric scores will save at: {result_excel_path}")
        if not os.path.exists(result_excel_path):
            df = pd.DataFrame(columns=column_titles)
            df.to_excel(result_excel_path, index=False)

        if "saturation" in self.use_metric_names:
            saturation_scores, saturation_scores_normed, saturation_times = [], [], []
        if "PSNR" in self.use_metric_names:
            PSNR_scores, PSNR_scores_normed, PSNR_times = [], [], []
        if "SSIM" in self.use_metric_names:
            SSIM_scores, SSIM_scores_normed, SSIM_times = [], [], []
        if "variance" in self.use_metric_names:
            variance_scores, variance_scores_normed, variance_times = [], [], []
        if "improved_aesthetic_predictor" in self.use_metric_names:
            improved_aesthetic_predictor_scores, improved_aesthetic_predictor_scores_normed, improved_aesthetic_predictor_times = [], [], []
        if "skytnt_anime_aesthetic" in self.use_metric_names:
            skytnt_anime_aesthetic_scores, skytnt_anime_aesthetic_scores_normed, skytnt_anime_aesthetic_times = [], [], []
        if "nsfw_detect" in self.use_metric_names:
            nsfw_detect_scores, nsfw_detect_scores_normed, nsfw_detect_times = [], [], []
        if "nsfw_detect_train" in self.use_metric_names:
            nsfw_detect_train_scores, nsfw_detect_train_scores_normed, nsfw_detect_train_times = [], [], []
        
        # 可以单图评估的
        img_path_or_url_contiue_path = f'{config.txt_dir}/contiue_{os.path.splitext(os.path.basename(images_dir_or_csv))[0]}_{get_formatted_current_time()}.txt'
        for index, img_path_or_url in enumerate(tqdm(img_paths_or_urls)):
            img_numpy = self.get_img_numpy(img_path_or_url)
            if img_numpy is None:
                with open(img_path_or_url_contiue_path, 'a', encoding='utf-8') as file:
                    file.write(f"{index} {img_path_or_url}\n")
                continue
            
            result_excel_ = {
                "img_path_or_url": img_path_or_url,
            }
            exec(f'all_weighted_score_normed = 0')
            for use_metric_name in self.use_metric_names:
                exec(f't_{use_metric_name}_start = time.time()')
                if use_metric_name in self.use_metric_forms["function"]:
                    exec(f'{use_metric_name}_score, {use_metric_name}_score_normed = calculate_{use_metric_name}_score(img_numpy)')
                elif use_metric_name in self.use_metric_forms["class"]:
                    exec(f'{use_metric_name}_score, {use_metric_name}_score_normed = self.{use_metric_name}_model(img_numpy)')
                exec(f't_{use_metric_name} = time.time() - t_{use_metric_name}_start')
                exec(f'{use_metric_name}_scores.append({use_metric_name}_score)')
                exec(f'{use_metric_name}_scores_normed.append({use_metric_name}_score_normed)')
                exec(f'{use_metric_name}_times.append(t_{use_metric_name})')

                exec(f'all_weighted_score_normed += {use_metric_name}_score_normed * config.metric_params["{use_metric_name}"]["score_normed_weight"]')
                exec(f'result_excel_["{use_metric_name}_score_normed"] = {use_metric_name}_score_normed')

            # save_as_excel
            exec(f'result_excel_["average_weighted_score_normed"] = all_weighted_score_normed / sum([config.metric_params[use_metric_name]["score_normed_weight"] for use_metric_name in self.use_metric_names])')
            single_df = pd.DataFrame([result_excel_])
            with pd.ExcelWriter(result_excel_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                existing_df = pd.read_excel(result_excel_path)
                existing_df = existing_df.dropna(axis=1, how='all')
                single_df = single_df.dropna(axis=1, how='all')
                result_df = pd.concat([existing_df, single_df], ignore_index=True)
                result_df.to_excel(writer, index=False)
        
        # # 必须组图评估的
        # if "FID" in self.use_metric_names:
        #     FID_score, FID_score_normed = calculate_FID_score(img_paths_or_urls)

        result_json_ = {}
        for use_metric_name in self.use_metric_names:
            exec(f'result_json_["{use_metric_name}"] = {{}}')
            exec(f'result_json_["{use_metric_name}"][f"average_{use_metric_name}_score"] = sum({use_metric_name}_scores) / len({use_metric_name}_scores)')
            exec(f'result_json_["{use_metric_name}"][f"average_{use_metric_name}_score_normed"] = sum({use_metric_name}_scores_normed) / len({use_metric_name}_scores_normed)')
            exec(f'result_json_["{use_metric_name}"][f"average_{use_metric_name}_time"] = sum({use_metric_name}_times) / len({use_metric_name}_times)')
        result_json_["average_weighted_score_normed"] = sum([config.metric_params[use_metric_name]["score_normed_weight"] * result_json_[use_metric_name][f"average_{use_metric_name}_score_normed"] for use_metric_name in self.use_metric_names]) / sum([config.metric_params[use_metric_name]["score_normed_weight"] for use_metric_name in self.use_metric_names])

        return result_json_


if __name__ == "__main__":
    img_eval = ImageEvaluation()

    result_json_path = f"{config.json_dir}/result_{get_formatted_current_time()}.json"
    print(f"images group average metric scores will save at: {result_json_path}")

    for test_images_dir_or_csv in tqdm(config.test_images_dirs_or_csvs):
        if not os.path.exists(result_json_path):
            result_json = {}
        else:
            try:
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    result_json = json.load(f)
            except json.decoder.JSONDecodeError as e:
                result_json = {}

        print(f"\nProcessing {test_images_dir_or_csv}...")
        result_json_ = img_eval(test_images_dir_or_csv)
        result_json[test_images_dir_or_csv] = result_json_
        
        with open(result_json_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result_json, indent=4, ensure_ascii=False))
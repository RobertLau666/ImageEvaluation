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
        # self.use_metric_names = [metric_name for metric_name, metric_param in config.metric_params.items() if metric_param["use"]]
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
            self.improved_aesthetic_predictor_model = ImprovedAestheticPredictor()
        if "skytnt_anime_aesthetic" in self.use_metric_names:
            self.skytnt_anime_aesthetic_model = SkytntAnimeAesthetic()
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
        
        save_as_excel = config.save_as_excel
        if save_as_excel:
            columns = ["img_path_or_url"] + self.use_metric_names
            output_excel_file = f'{config.xlsx_dir}/result_{os.path.splitext(os.path.basename(images_dir_or_csv))[0]}_{get_formatted_current_time()}.xlsx'
            print(f"images metric scores will save at: {output_excel_file}")
            if not os.path.exists(output_excel_file):
                df = pd.DataFrame(columns=columns)
                df.to_excel(output_excel_file, index=False)

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
        for img_path_or_url in tqdm(img_paths_or_urls):
            img_numpy = self.get_img_numpy(img_path_or_url)
            if img_numpy is None:
                continue
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


            # if "saturation" in self.use_metric_names:
            #     t_saturation_start = time.time()
            #     saturation_score, saturation_score_normed = calculate_saturation_score(img_numpy)
            #     t_saturation = time.time() - t_saturation_start
            #     saturation_scores.append(saturation_score)
            #     saturation_scores_normed.append(saturation_score_normed)
            #     saturation_times.append(t_saturation)
            # if "PSNR" in self.use_metric_names:
            #     t_PSNR_start = time.time()
            #     PSNR_score, PSNR_score_normed = calculate_PSNR_score(img_numpy)
            #     t_PSNR = time.time() - t_PSNR_start
            #     PSNR_scores.append(PSNR_score)
            #     PSNR_scores_normed.append(PSNR_score_normed)
            #     PSNR_times.append(t_PSNR)
            # if "SSIM" in self.use_metric_names:
            #     t_SSIM_start = time.time()
            #     SSIM_score, SSIM_score_normed = calculate_SSIM_score(img_numpy)
            #     t_SSIM = time.time() - t_SSIM_start
            #     SSIM_scores.append(SSIM_score)
            #     SSIM_scores_normed.append(SSIM_score_normed)
            #     SSIM_times.append(t_SSIM)
            # if "variance" in self.use_metric_names:
            #     t_variance_start = time.time()
            #     variance_score, variance_score_normed = calculate_variance_score(img_numpy)
            #     t_variance = time.time() - t_variance_start
            #     variance_scores.append(variance_score)
            #     variance_scores_normed.append(variance_score_normed)
            #     variance_times.append(t_variance)
            # if "improved_aesthetic_predictor" in self.use_metric_names:
            #     t_improved_aesthetic_predictor_start = time.time()
            #     improved_aesthetic_predictor_score, improved_aesthetic_predictor_score_normed = self.improved_aesthetic_predictor_model(img_numpy)
            #     t_improved_aesthetic_predictor = time.time() - t_improved_aesthetic_predictor_start
            #     improved_aesthetic_predictor_scores.append(improved_aesthetic_predictor_score)
            #     improved_aesthetic_predictor_scores_normed.append(improved_aesthetic_predictor_score_normed)
            #     improved_aesthetic_predictor_times.append(t_improved_aesthetic_predictor)
            # if "skytnt_anime_aesthetic" in self.use_metric_names:
            #     t_skytnt_anime_aesthetic_start = time.time()
            #     skytnt_anime_aesthetic_score, skytnt_anime_aesthetic_score_normed = self.skytnt_anime_aesthetic_model(img_numpy)
            #     t_skytnt_anime_aesthetic = time.time() - t_skytnt_anime_aesthetic_start
            #     skytnt_anime_aesthetic_scores.append(skytnt_anime_aesthetic_score)
            #     skytnt_anime_aesthetic_scores_normed.append(skytnt_anime_aesthetic_score_normed)
            #     skytnt_anime_aesthetic_times.append(t_skytnt_anime_aesthetic)
            # if "nsfw_detect" in self.use_metric_names:
            #     t_nsfw_detect_start = time.time()
            #     nsfw_detect_score, nsfw_detect_score_normed = self.nsfw_detect_model(img_numpy)
            #     t_nsfw_detect = time.time() - t_nsfw_detect_start
            #     nsfw_detect_scores.append(nsfw_detect_score)
            #     nsfw_detect_scores_normed.append(nsfw_detect_score_normed)
            #     nsfw_detect_times.append(t_nsfw_detect)
            # if "nsfw_detect_train" in self.use_metric_names:
            #     t_nsfw_detect_train_start = time.time()
            #     nsfw_detect_train_score, nsfw_detect_train_score_normed = self.nsfw_detect_train_model(img_numpy)
            #     t_nsfw_detect_train = time.time() - t_nsfw_detect_train_start
            #     nsfw_detect_train_scores.append(nsfw_detect_train_score)
            #     nsfw_detect_train_scores_normed.append(nsfw_detect_train_score_normed)
            #     nsfw_detect_train_times.append(t_nsfw_detect_train)
        
            if save_as_excel:
                img_result = {
                    "img_path_or_url": img_path_or_url,
                }
                for column in columns[1:]:
                    exec(f'img_result["{column}"] = {column}_score')
                single_df = pd.DataFrame([img_result])
                with pd.ExcelWriter(output_excel_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                    existing_df = pd.read_excel(output_excel_file)
                    # 去除全 NA 的列
                    existing_df = existing_df.dropna(axis=1, how='all')
                    single_df = single_df.dropna(axis=1, how='all')
                    result_df = pd.concat([existing_df, single_df], ignore_index=True)
                    result_df.to_excel(writer, index=False)
        
        # # 必须组图评估的
        # if "FID" in self.use_metric_names:
        #     FID_score = calculate_FID_score(img_paths_or_urls)

        score_json = {}
        for use_metric_name in self.use_metric_names:
            # if use_metric_name == "nsfw_detect_train":
            #     average_nsfw_detect_train_score = 0
            #     for nsfw_detect_train_score in nsfw_detect_train_scores:
            #         average_nsfw_detect_train_score += (1 if int(nsfw_detect_train_score) >= 1 else 0)
            #     score_json[f"average {use_metric_name} score"] = 1 - average_nsfw_detect_train_score / len(nsfw_detect_train_scores)
            # else:
            exec(f'score_json["{use_metric_name}"] = {{}}')
            exec(f'score_json["{use_metric_name}"][f"average {use_metric_name} score"] = sum({use_metric_name}_scores) / len({use_metric_name}_scores)')
            exec(f'score_json["{use_metric_name}"][f"average {use_metric_name} score normed"] = sum({use_metric_name}_scores_normed) / len({use_metric_name}_scores_normed)')
            exec(f'score_json["{use_metric_name}"][f"average {use_metric_name} time"] = sum({use_metric_name}_times) / len({use_metric_name}_times)')

        average_weighted_score_normed = sum([config.metric_params[use_metric_name]["score_normed_weight"] * score_json[use_metric_name][f"average {use_metric_name} score normed"] for use_metric_name in self.use_metric_names]) / sum([config.metric_params[use_metric_name]["score_normed_weight"] for use_metric_name in self.use_metric_names])
        score_json["average weighted score normed"] = average_weighted_score_normed

        return score_json


if __name__ == "__main__":
    img_eval = ImageEvaluation()

    record_dict_path = f"{config.json_dir}/result_{get_formatted_current_time()}.json"
    print(f"images group average metric scores will save at: {record_dict_path}")

    for test_images_dir_or_csv in tqdm(config.test_images_dirs_or_csvs):
        if not os.path.exists(record_dict_path):
            record_dict = {}
        else:
            try:
                with open(record_dict_path, 'r', encoding='utf-8') as f:
                    record_dict = json.load(f)
            except json.decoder.JSONDecodeError as e:
                record_dict = {}

        print(f"\nProcessing {test_images_dir_or_csv}...")
        score_json = img_eval(test_images_dir_or_csv)
        record_dict[test_images_dir_or_csv] = score_json
        
        with open(record_dict_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(record_dict, indent=4, ensure_ascii=False))
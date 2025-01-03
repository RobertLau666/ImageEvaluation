import os
import json
import time
from tqdm import tqdm
import config
from metrics import *
import shutil


class ImageEvaluation():
    def __init__(self):
        self.get_metric_names_and_forms()
        self.init_metrics()

    def get_metric_names_and_forms(self):
        self.metric_names = []
        self.metric_forms = {}
        for metric_name, metric_param in config.metric_params.items():
            if metric_param.get("use", False):
                self.metric_names.append(metric_name)
            if metric_param["use_form"] not in self.metric_forms:
                self.metric_forms[metric_param["use_form"]] = [metric_name]
            else:
                self.metric_forms[metric_param["use_form"]].append(metric_name)

        print(f"metric_names: {self.metric_names}")
        print(f"metric_forms: {self.metric_forms}")
    
    def init_metrics(self):
        print("Initializing metrics...")
        if "improved_aesthetic_predictor" in self.metric_names:
            self.improved_aesthetic_predictor_model = ImprovedAestheticPredictor(model_path=config.metric_params["improved_aesthetic_predictor"]["model_path"])
        if "skytnt_anime_aesthetic" in self.metric_names:
            self.skytnt_anime_aesthetic_model = SkytntAnimeAesthetic(model_path=config.metric_params["skytnt_anime_aesthetic"]["model_path"])
        if "nsfw_detect" in self.metric_names:
            self.nsfw_detect_model = API_ViT_v3(model_path=config.metric_params["nsfw_detect"]["model_path"])
        if "nsfw_detect_train" in self.metric_names:
            self.nsfw_detect_train_model = NSFWSelfTrainCls(model_path_or_url=config.metric_params["nsfw_detect_train"]["model_path_or_url"])
        if "children_detect_train" in self.metric_names:
            self.children_detect_train_model = ChildrenSelfTrainCls(model_path_or_url=config.metric_params["children_detect_train"]["model_path_or_url"])

    def get_img_paths_or_urls_and_types(self, images_dir_or_file):
        if images_dir_or_file.endswith(('.csv', '.xlsx', '.txt', '.log')):
            self.is_img_url_file = True
            img_paths_or_urls, types = get_img_urls(images_dir_or_file)
        else:
            self.is_img_url_file = False
            img_paths_or_urls = [os.path.join(images_dir_or_file, img_name) for img_name in sorted(os.listdir(images_dir_or_file))]
            types = ['no_type'] * len(img_paths_or_urls)
        return img_paths_or_urls, types

    def __call__(self, images_dir_or_file):
        img_paths_or_urls, types = self.get_img_paths_or_urls_and_types(images_dir_or_file)
        
        column_titles = []
        result_xlsx_path = os.path.join(config.xlsx_dir, f'{get_formatted_current_time()}_{os.path.basename(images_dir_or_file)}.xlsx')
        print(f"Every image metric scores will save at: {result_xlsx_path}")
        if not os.path.exists(result_xlsx_path):
            df = pd.DataFrame(columns=column_titles)
            df.to_excel(result_xlsx_path, index=False)

        if "saturation" in self.metric_names:
            saturation_scores, saturation_scores_normed, saturation_times = [], [], []
        if "PSNR" in self.metric_names:
            PSNR_scores, PSNR_scores_normed, PSNR_times = [], [], []
        if "SSIM" in self.metric_names:
            SSIM_scores, SSIM_scores_normed, SSIM_times = [], [], []
        if "variance" in self.metric_names:
            variance_scores, variance_scores_normed, variance_times = [], [], []
        if "improved_aesthetic_predictor" in self.metric_names:
            improved_aesthetic_predictor_scores, improved_aesthetic_predictor_scores_normed, improved_aesthetic_predictor_times = [], [], []
        if "skytnt_anime_aesthetic" in self.metric_names:
            skytnt_anime_aesthetic_scores, skytnt_anime_aesthetic_scores_normed, skytnt_anime_aesthetic_times = [], [], []
        if "nsfw_detect" in self.metric_names:
            nsfw_detect_scores, nsfw_detect_scores_normed, nsfw_detect_times = [], [], []
        if "nsfw_detect_train" in self.metric_names:
            nsfw_detect_train_scores, nsfw_detect_train_scores_normed, nsfw_detect_train_times = [], [], []
        if "children_detect_train" in self.metric_names:
            children_detect_train_scores, children_detect_train_scores_normed, children_detect_train_times = [], [], []

        # 1. calculate metrics
        # 1.1 metrics for single image
        img_path_or_url_skip_path = os.path.join(config.txt_dir, f'{get_formatted_current_time()}_{os.path.basename(images_dir_or_file)}_skip.txt')
        print(f"Skipped image paths or urls will save at: {img_path_or_url_skip_path}")
        for index, (img_path_or_url, type) in enumerate(tqdm(zip(img_paths_or_urls, types))):
            img_numpy = get_image_numpy_from_img_url(img_path_or_url) if self.is_img_url_file else cv2.imread(img_path_or_url)
            if img_numpy is None:
                with open(img_path_or_url_skip_path, 'a', encoding='utf-8') as file:
                    file.write(f"{index} {img_path_or_url} {type}\n")
                continue
            
            result_excel_ = {
                "img_path_or_url": img_path_or_url,
                "type": type,
            }
            exec('all_weighted_score_normed = 0')
            for metric_name in self.metric_names:
                exec(f't_{metric_name}_start = time.time()')
                if metric_name in self.metric_forms["function"]:
                    exec(f'{metric_name}_score, {metric_name}_score_normed = calculate_{metric_name}_score(img_numpy)')
                elif metric_name in self.metric_forms["class"]:
                    exec(f'{metric_name}_score, {metric_name}_score_normed = self.{metric_name}_model(img_numpy)')
                exec(f't_{metric_name} = time.time() - t_{metric_name}_start')
                exec(f'{metric_name}_scores.append({metric_name}_score)')
                exec(f'{metric_name}_scores_normed.append({metric_name}_score_normed)')
                exec(f'{metric_name}_times.append(t_{metric_name})')
                exec(f'all_weighted_score_normed += {metric_name}_score_normed * config.metric_params["{metric_name}"]["score_normed_weight"]')
                exec(f'result_excel_["{metric_name}_score_normed"] = {metric_name}_score_normed')
            exec(f'result_excel_["average_weighted_score_normed"] = all_weighted_score_normed / sum([config.metric_params[metric_name]["score_normed_weight"] for metric_name in self.metric_names])')
            single_df = pd.DataFrame([result_excel_])
            with pd.ExcelWriter(result_xlsx_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                existing_df = pd.read_excel(result_xlsx_path)
                existing_df = existing_df.dropna(axis=1, how='all')
                single_df = single_df.dropna(axis=1, how='all')
                result_df = pd.concat([existing_df, single_df], ignore_index=True)
                result_df.to_excel(writer, index=False)
        
        # 1.2 metrics for group images, such as FID

        # 2. convert xlsx to csv
        result_csv_path = os.path.join(config.csv_dir, os.path.splitext(os.path.basename(result_xlsx_path))[0] + '.csv')
        convert_xlsx_to_csv(result_xlsx_path, result_csv_path)
        shutil.rmtree(config.xlsx_dir)
        
        # 3. generate report html
        generate_html_report(result_csv_path)

        # 4. generate result json
        result_json_ = {}
        for metric_name in self.metric_names:
            exec(f'result_json_["{metric_name}"] = {{}}')
            exec(f'result_json_["{metric_name}"][f"average_{metric_name}_score"] = sum({metric_name}_scores) / len({metric_name}_scores)')
            exec(f'result_json_["{metric_name}"][f"average_{metric_name}_score_normed"] = sum({metric_name}_scores_normed) / len({metric_name}_scores_normed)')
            exec(f'result_json_["{metric_name}"][f"average_{metric_name}_time"] = sum({metric_name}_times) / len({metric_name}_times)')
        result_json_["average_weighted_score_normed"] = sum([config.metric_params[metric_name]["score_normed_weight"] * result_json_[metric_name][f"average_{metric_name}_score_normed"] for metric_name in self.metric_names]) / sum([config.metric_params[metric_name]["score_normed_weight"] for metric_name in self.metric_names])

        # 5. generate plot png
        for column_title in ["nsfw_detect_train_score_normed", "children_detect_train_score_normed"]:
            generate_plot_by_column_title(result_csv_path, column_title)

        return result_json_


def main():
    img_eval = ImageEvaluation()
    config.create_dirs(config.test_images_dirs_or_files)
    result_json_path = os.path.join(config.json_dir, f"{get_formatted_current_time()}_{'_'.join([os.path.basename(test_images_dir_or_file) for test_images_dir_or_file in config.test_images_dirs_or_files])}.json")
    for test_images_dir_or_file in tqdm(config.test_images_dirs_or_files):
        if not os.path.exists(result_json_path):
            result_json = {
                "metric_params": config.metric_params,
                "test_images_dirs_or_files": {}
            }
        else:
            try:
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    result_json = json.load(f)
            except json.decoder.JSONDecodeError as e:
                result_json = {
                    "metric_params": config.metric_params,
                    "test_images_dirs_or_files": {}
                }

        print(f"\n\nProcessing {test_images_dir_or_file}...")
        result_json["test_images_dirs_or_files"][test_images_dir_or_file] = img_eval(test_images_dir_or_file)
        
        with open(result_json_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result_json, indent=4, ensure_ascii=False))
    print(f"Group image metric average scores saved at: {result_json_path}")

if __name__ == "__main__":
    main()
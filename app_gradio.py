import gradio as gr
import pandas as pd
import app
import config
import os
import shutil
from metrics.file_process import get_formatted_current_time, concatenate_images

def process(upload_file, checked_metric_names):
    if upload_file is None:
        return 'Warning: please upload file!', None, None, None, None, None, None
    if len(checked_metric_names) == 0:
        return 'Warning: please check metric names!', None, None, None, None, None, None
    
    gradio_input_dir = "data/input/gradio"
    if not os.path.exists(gradio_input_dir):
        os.makedirs(gradio_input_dir, exist_ok=True)
    upload_file_save_path = os.path.join(gradio_input_dir, '_'.join(['uploadtime:' + get_formatted_current_time(), os.path.basename(upload_file.name)]))
    shutil.move(upload_file.name, upload_file_save_path)

    for metric_name in list(config.metric_params.keys()):
        config.metric_params[metric_name]["use"] = True if metric_name in checked_metric_names else False
    config.test_images_dirs_or_files = [upload_file_save_path]
    config.create_dirs(config.test_images_dirs_or_files)

    app.main()

    csv_file_path = os.path.join(config.csv_dir, os.listdir(config.csv_dir)[0]) if len(os.listdir(config.csv_dir)) != 0 else None
    html_file_path = os.path.join(config.html_dir, os.listdir(config.html_dir)[0]) if len(os.listdir(config.html_dir)) != 0 else None
    png_file_path = concatenate_images(config.png_dir)
    txt_file_path = os.path.join(config.txt_dir, os.listdir(config.txt_dir)[0]) if len(os.listdir(config.txt_dir)) != 0 else None
    json_file_path = os.path.join(config.json_dir, os.listdir(config.json_dir)[0]) if len(os.listdir(config.json_dir)) != 0 else None
    shutil.make_archive(config.output_dir, 'zip', config.output_dir)
    zip_file_path = config.output_dir + '.zip'

    return "Process done!", csv_file_path, html_file_path, png_file_path, txt_file_path, json_file_path, zip_file_path

def get_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            Image Analysis Tool
            </div>  
            """
        )
        upload_file = gr.File(label="Upload file: 1. upload file which suffix in ['.csv', '.xlsx', '.txt', '.log'] 2. the format of each line must be either 'img_url' or 'img_url type' 3. column titles are not required", file_types=[".csv", ".xlsx", ".txt", ".log"])
        checked_metric_names = gr.CheckboxGroup(list(config.metric_params.keys()), label="metric_names", info="Check the metric names you want to detect")
        process_button = gr.Button("Process")
        status = gr.Textbox(label="Status", value="Processing not started", interactive=True)
        with gr.Row():
            csv_file = gr.File(label="Download csv file")
            html_file = gr.File(label="Download html file")
            png_file = gr.File(label="Download png file")
            txt_file = gr.File(label="Download txt file")
            json_file = gr.File(label="Download json file")
            zip_file = gr.File(label="Download zip file")

        process_button.click(
            process,
            inputs=[upload_file, checked_metric_names],
            outputs=[status, csv_file, html_file, png_file, txt_file, json_file, zip_file]
        )
    return demo

if __name__ == "__main__":
    get_demo().launch(share=True)
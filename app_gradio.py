import gradio as gr
import pandas as pd
import app
import config
import os
import shutil
from metrics.file_process import get_formatted_current_time

def process_file(file, checked_metric_names):
    print("file:", file)
    print("file.name:", file.name)
    print("checked_metric_names:", checked_metric_names)

    save_dir = "/maindata/data/shared/public/chenyu.liu/others/ImageEvaluation/data/input"
    save_path = os.path.join(save_dir, '_'.join(['uploadtime:' + get_formatted_current_time(), os.path.basename(file.name)]))
    shutil.move(file.name, save_path)

    for metric_name in list(config.metric_params.keys()):
        config.metric_params[metric_name]["use"] = True if metric_name in checked_metric_names else False
    config.test_images_dirs_or_files = [save_path]

    app.main()

    # df = pd.read_csv(file.name)

    # with open(save_path, "wb") as f:
    #     f.write(file.read())  # 保存文件内容

    # analysis = df.groupby('event').agg(
    #     total_images=('url', 'count'),
    #     blocked_images=('blocked', 'sum')
    # )
    # analysis['block_rate'] = (analysis['blocked_images'] / analysis['total_images'] * 100).round(2).astype(str) + '%'
    # analysis = analysis.reset_index()
    
    # output_csv = "results.csv"
    # analysis.to_csv(output_csv, index=False)
    status = "process done!"
    return status #, analysis, output_csv

with gr.Blocks() as demo:
    gr.Markdown("#Image Analysis Tool")
    
    with gr.Row():
        file = gr.File(label="Upload File", file_types=[".csv", ".xlsx", ".txt", ".log"])
    checked_metric_names = gr.CheckboxGroup(list(config.metric_params.keys()), label="metric_names", info="Check the metric names you want to detect")
    process_button = gr.Button("Process")
    status = gr.Textbox(label="Status", value="Processing not started", interactive=False)
    result_table = gr.Dataframe(label="Analysis Results")
    download_csv = gr.File(label="Download Results CSV")
    
    process_button.click(
        process_file,
        inputs=[file, checked_metric_names], 
        outputs=[status] # , result_table, download_csv
    )
    
if __name__ == "__main__":
    demo.launch(share=True)
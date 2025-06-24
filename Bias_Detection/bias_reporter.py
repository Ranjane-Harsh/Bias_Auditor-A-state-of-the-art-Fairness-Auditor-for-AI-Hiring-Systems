import os
import pandas as pd
import matplotlib.pyplot as plt

def save_summary_to_csv(attribute_name, summary_df,output_dir):
    file_name = f"{attribute_name}_bias_summary.csv"
    full_path = os.path.join(output_dir, file_name)
    summary_df.to_csv(full_path, index= True)
    return full_path

def plot_disparities(attribute_name, summary_df, output_dir):
    metrices = ['Selection_rate','Disparate_impact_ratio', 'True_postive_rate','False_postive_rate']
    for metric in metrices:
        values = summary_df.loc[metric]
        values = values.dropna()

        plt.figure()
        plt.bar(values.index, values.values)
        plt.xlabel(attribute_name)
        plt.ylabel(metric)
        plt.title(f"{metric} - {attribute_name}")
        plt.tight_layout()

        file_name = f"{attribute_name}_{metric}.png"
        full_path = os.path.join(output_dir, file_name)
        plt.savefig(full_path)
        plt.close()

def run_bias_report(summary_dict,output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for attribute_name, summary_df in summary_dict.items():
        csv_path = save_summary_to_csv(attribute_name, summary_df,output_dir)
        plot_disparities(attribute_name,summary_df, output_dir)
    return 
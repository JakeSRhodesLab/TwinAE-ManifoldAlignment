## Regression Vizualization helpers
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from Helpers.utils import subset_df, plot_in_fig
import json

def read_json_files_to_dataframe(directory_path):
    # List to store data from all JSON files
    data_list = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                # Construct full file path
                file_path = os.path.join(root, file)
                
                # Read JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data_list.append(data)
    
    # Create a DataFrame from the list of JSON data
    df = pd.DataFrame(data_list)
    
    return df

def report_missing_json_files(directory_path):
    print("Missing Data Files:")

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory_path):
        print(f"    Entering directory: {root}")

        found_table = {

        }

        for file in files:
            if file.endswith('.json'):
                # Construct full file path
                file_path = os.path.join(root, file)
                
                


def plot_radial(df, columns):
    unique_methods = df["method"].unique()

    # Calculate the sum of scores for each method for sorting
    method_scores = {}
    for method_name in unique_methods:
        method_data = df[df["method"] == method_name][columns]
        method_data = method_data.mean()

        if "FOSCTTM" in columns:
            method_data["FOSCTTM"] = 1 - method_data["FOSCTTM"]
        if "Nearest Neighbor (F1 score or RMSE)" in columns:
            method_data["Nearest Neighbor (F1 score or RMSE)"] = 1 - method_data["Nearest Neighbor (F1 score or RMSE)"]
        if "Random Forest (F1 score or RMSE)" in columns:
            method_data["Random Forest (F1 score or RMSE)"] = 1 - method_data["Random Forest (F1 score or RMSE)"]
        if "Grae-KNN-metric" in columns:
            method_data["Grae-KNN-metric"] = 1 - method_data["Grae-KNN-metric"]
        if "Grae-RF-metric" in columns:
            method_data["Grae-RF-metric"] = 1 - method_data["Grae-RF-metric"]

        score_sum = method_data.sum()
        method_scores[method_name] = score_sum

    # Sort methods by total scores (descending order)
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_method_names = [item[0] for item in sorted_methods]

    # Define a color palette
    colors = sns.color_palette("husl", n_colors=len(unique_methods))

    # Calculate grid size
    grid_size = int(len(unique_methods)**0.5) + 1

    # Create polar subplots
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(20, 20), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    # Loop through each sorted method
    for i, (method_name, color) in enumerate(zip(sorted_method_names, colors)):
        # Select and preprocess data
        method_data = df[df["method"] == method_name][columns]

        if "FOSCTTM" in columns:
            method_data["FOSCTTM"] = 1 - method_data["FOSCTTM"]
        if "Nearest Neighbor (F1 score or RMSE)" in columns:
            method_data["Nearest Neighbor (F1 score or RMSE)"] = 1 - method_data["Nearest Neighbor (F1 score or RMSE)"]
        if "Random Forest (F1 score or RMSE)" in columns:
            method_data["Random Forest (F1 score or RMSE)"] = 1 - method_data["Random Forest (F1 score or RMSE)"]
        if "Grae-KNN-metric" in columns:
            method_data["Grae-KNN-metric"] = 1 - method_data["Grae-KNN-metric"]
        if "Grae-RF-metric" in columns:
            method_data["Grae-RF-metric"] = 1 - method_data["Grae-RF-metric"]

        method_data = method_data.mean()  # Take the mean for radar chart values

        # Prepare radar chart data
        categories = columns  # Use the column names for categories
        values = method_data.tolist()
        values += values[:1]  # Close the loop

        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]

        # Plot the radar chart
        ax = axes[i]
        ax.plot(angles, values, label=method_name, color=color)
        ax.fill(angles, values, color=color, alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(-0.5, 1)
        ax.set_title(f"Method: {method_name}\nTotal Score: {method_scores[method_name]:.2f}", 
                    fontsize=10, color=color)


    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

move_index = -0.03
def get_index_pos(agregate_df):
    global move_index
    move_index += 0.03

    return agregate_df.index + move_index
    
def plt_methods_by_CSV_max(df, sort_by = "MASH", metric = "Combined_Metric", return_df =False, plot_methods = ["MASH", "SPUD"]):
    """df should equal the dataframe. It can be subsetted already
    
    Plots the max of the combined metric for each method to each CSV_File
    
    sort_by should be the string of what the method you want"""

    agregate_df = pd.DataFrame({
            'SSMA': df[df["method"] == "SSMA"].groupby("csv_file")[metric].max(),
            'MAGAN': df[df["method"] == "MAGAN"].groupby("csv_file")[metric].max(),
            'DTA': df[df["method"] == "DTA"].groupby("csv_file")[metric].max(),
            'SPUD': df[df["method"] == "SPUD"].groupby("csv_file")[metric].max(),
            'MASH': df[df["method"] == "MASH"].groupby("csv_file")[metric].max(),
            'MASH-': df[df["method"] == "MASH-"].groupby("csv_file")[metric].max(),
            'RF-MASH-': df[df["method"] == "RF-MASH-"].groupby("csv_file")[metric].max(),
            'NAMA': df[df["method"] == "NAMA"].groupby("csv_file")[metric].max(),
            'RF-NAMA': df[df["method"] == "RF-NAMA"].groupby("csv_file")[metric].max(),
            'PCR': df[df["method"].isin(["PCR", "MAPA"])].groupby("csv_file")[metric].max(),
            'JLMA': df[df["method"] == "JLMA"].groupby("csv_file")[metric].max(),
            'MASH_RF': df[df["method"].isin(["MASH_RF", "RF-MASH"])].groupby("csv_file")[metric].max(),
            'MALI_RF': df[df["method"].isin(["MALI_RF", "MALI-RF"])].groupby("csv_file")[metric].max(),
            'MALI': df[df["method"] == "MALI"].groupby("csv_file")[metric].max(),
            'SPUD_RF': df[df["method"].isin(["SPUD_RF", "RF-SPUD"])].groupby("csv_file")[metric].max(),
            'BL_A': df.groupby("csv_file")["A_Classification_Score"].max(),
            'BL_B': df.groupby("csv_file")["B_Classification_Score"].max(),
    })

    agregate_df = agregate_df.sort_values(by = sort_by).reset_index()

    #If we only want the df
    if return_df:
        return agregate_df

    #To make it easier to add edits
    key_words = {
                "s" : 70,
                "alpha" : .60,
                }

    plt.figure(figsize=(16, 6))
    
    if "MASH" in plot_methods:
        ax = plt.scatter(y = agregate_df["MASH"], marker = '^', color = "green", label = "MASH", x = get_index_pos(agregate_df), **key_words)
    if "MAGAN" in plot_methods:
        ax = plt.scatter(y = agregate_df["MAGAN"], marker = '^', color = "red", label = "MAGAN",x = get_index_pos(agregate_df), **key_words)
    if "JLMA" in plot_methods:
        ax = plt.scatter(y = agregate_df["JLMA"], marker = '^', label = "JLMA",x = get_index_pos(agregate_df), **key_words)
    if "SPUD" in plot_methods:
        ax = plt.scatter(y = agregate_df["SPUD"], marker = "^", label = "SPUD",x = get_index_pos(agregate_df), **key_words)
    if "MASH-" in plot_methods:
        ax = plt.scatter(y = agregate_df["MASH-"], marker = '^', label = "MASH-",x = get_index_pos(agregate_df), **key_words)
    if "NAMA" in plot_methods:
        ax = plt.scatter(y = agregate_df["NAMA"], marker = '^', label = "NAMA",x = get_index_pos(agregate_df), **key_words)
    if "PCR" in plot_methods:
        ax = plt.scatter(y = agregate_df["PCR"], marker = '^', color = "brown",x = get_index_pos(agregate_df), label = "Procrutees", **key_words)
    if "DTA" in plot_methods:
        ax = plt.scatter(y = agregate_df["DTA"], marker = "^", color = "orange",x = get_index_pos(agregate_df), label = "DTA", **key_words)
    if "SSMA" in plot_methods:
        ax = plt.scatter(y = agregate_df["SSMA"],  marker = '^', label = "SSMA",x = get_index_pos(agregate_df), **key_words) 
    if "BL_A" in plot_methods:
        ax = plt.scatter(y = agregate_df["BL_A"], marker = '.', color = "red",x = get_index_pos(agregate_df), label = "BL_A", **key_words)
    if "BL_B" in plot_methods:
        ax = plt.scatter(y = agregate_df["BL_B"], marker = '.', color = "pink", x = get_index_pos(agregate_df), label = "BL_B", **key_words)

    #To make it easier to add edits
    key_words = {
                "s" : 50,
                "alpha" : .90 }
    
    if "RF-NAMA" in plot_methods:
        ax = plt.scatter(y = agregate_df["RF-NAMA"], marker = 'o', color = "red", x = get_index_pos(agregate_df),label = "RF-NAMA", **key_words)
    if "RF-MASH-" in plot_methods:
        ax = plt.scatter(y = agregate_df["RF-MASH-"], marker = 'o', color = "orange", x = get_index_pos(agregate_df),label = "RF-MASH-", **key_words)
    if "MASH_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["MASH_RF"], marker = 'o', color = "green", x = get_index_pos(agregate_df),label = "MASH_RF", **key_words)
    if "MALI_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["MALI_RF"], marker = 'o',x = get_index_pos(agregate_df), label = "MALI_RF", **key_words)
    if "MALI" in plot_methods:
        ax = plt.scatter(y = agregate_df["MALI"], marker = 'o', x = get_index_pos(agregate_df),label = "MALI", **key_words)
    if "SPUD_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["SPUD_RF"], x = get_index_pos(agregate_df), label = "SPUD_RF", marker = 'o', color = "blue", **key_words)

    global move_index
    move_index = -0.03

    #Show Legend
    plt.xticks(ticks= agregate_df.index,labels=agregate_df["csv_file"], rotation = 90)
    plt.title(f"{metric} Scores vs. CSV Files (MAX)")
    plt.ylabel(metric)
    plt.grid(visible=True, axis = "x")
    plt.legend()
    #plt.show()

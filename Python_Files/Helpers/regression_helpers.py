## Regression Vizualization helpers
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from Python_Files.Helpers.utils import subset_df, plot_in_fig
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
            'NAMA': df[df["method"] == "NAMA"].groupby("csv_file")[metric].max(),
            'PCR': df[df["method"] == "PCR"].groupby("csv_file")[metric].max(),
            'JLMA': df[df["method"] == "JLMA"].groupby("csv_file")[metric].max(),
            'MASH_RF': df[df["method"] == "MASH_RF"].groupby("csv_file")[metric].max(),
            'MALI_RF': df[df["method"] == "MALI_RF"].groupby("csv_file")[metric].max(),
            'MALI': df[df["method"] == "MALI"].groupby("csv_file")[metric].max(),
            'KEMA_RF': df[df["method"] == "KEMA_RF"].groupby("csv_file")[metric].max(),
            'SPUD_RF': df[df["method"] == "SPUD_RF"].groupby("csv_file")[metric].max(),
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
    if "SPUD" in plot_methods:
        ax = plt.scatter(y = agregate_df["SPUD"], label = "SPUD", marker = '^',x = get_index_pos(agregate_df), color = "blue", **key_words)
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
    
    if "MASH_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["MASH_RF"], marker = 'o', color = "green", x = get_index_pos(agregate_df),label = "MASH_RF", **key_words)
    if "MALI_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["MALI_RF"], marker = 'o',x = get_index_pos(agregate_df), label = "MALI_RF", **key_words)
    if "MALI" in plot_methods:
        ax = plt.scatter(y = agregate_df["MALI"], marker = 'o', x = get_index_pos(agregate_df),label = "MALI", **key_words)
    if "SPUD_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["SPUD_RF"], x = get_index_pos(agregate_df), label = "SPUD_RF", marker = 'o', color = "blue", **key_words)
    if "KEMA_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["KEMA_RF"],x = get_index_pos(agregate_df), marker = 'o', color = "purple", label = "KEMA", **key_words)

    global move_index
    move_index = -0.03

    #Show Legend
    plt.xticks(ticks= agregate_df.index,labels=agregate_df["csv_file"], rotation = 90)
    plt.title(f"{metric} Scores vs. CSV Files (MAX)")
    plt.ylabel(metric)
    plt.grid(visible=True, axis = "x")
    plt.legend()
    #plt.show()

import numpy as np

def discretize_labels(regression_labels):
    """
    Transforms regression labels into ten discrete labels.
    
    Parameters:
    regression_labels (list or np.array): Array of regression labels.
    
    Returns:
    np.array: Array of discretized labels.
    """
    # Convert to numpy array if not already
    regression_labels = np.array(regression_labels)
    
    # Calculate the percentiles for discretization
    percentiles = np.percentile(regression_labels, np.arange(0, 101, 10))
    
    # Digitize the regression labels into 10 bins
    discrete_labels = np.digitize(regression_labels, percentiles, right=True) - 1
    
    # Ensure labels are in the range 0-9
    discrete_labels = np.clip(discrete_labels, 0, 9)
    
    return discrete_labels
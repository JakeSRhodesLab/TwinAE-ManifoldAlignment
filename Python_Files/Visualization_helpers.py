# Visulization_Helper
import seaborn as sns
import test_manifold_algorithms as tma
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import subset_df, plot_in_fig


#If there is no new data, we could just read in the old csvfile
df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/All_Data_DataFrame.csv", keep_default_na=False, na_values=['', 'NaN'], index_col= None)

#If there is no new data, we could just read in the old csvfile
og_df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/ManifoldData/Data_DataFrame.csv", keep_default_na=False, na_values=['', 'NaN'], index_col= None)

#If there is no new data, we could just read in the old csvfile
rf_df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/ManifoldData_RF/Data_DataFrame.csv", keep_default_na=False, na_values=['', 'NaN'], index_col= None)

def create_DataFrames():
    """Uploads and creates dataframe files"""

    global df
    global og_df
    global rf_df
    
    #Veiwing with DataFrame
    rf_df = tma.upload_to_DataFrame(directory = "Not_default")
    
    #Add a combined metric to help see (The closer to 1 the better)
    rf_df["Combined_Metric"] = rf_df["Cross_Embedding_KNN"] - rf_df["FOSCTTM"]

    #Fix DataFrame if it was bad values
    rf_df = rf_df[~(rf_df["Percent_of_Anchors"].astype(float) > 0.5)]

    #Save the Data Frame
    rf_df.to_csv(os.getcwd()[:-12] + "ManifoldData_RF/Data_DataFrame.csv", index=False, na_rep='NaN')

    #Veiwing with DataFrame
    og_df = tma.upload_to_DataFrame(directory = "default")

    #Add a combined metric to help see (The closer to 1 the better)
    og_df["Combined_Metric"] = og_df["Cross_Embedding_KNN"] - og_df["FOSCTTM"]

    #Fix DataFrame if it was bad values
    og_df = og_df[~(og_df["Percent_of_Anchors"].astype(float) > 0.5)]

    #Save the Data Frame
    og_df.to_csv(os.getcwd()[:-12] + "ManifoldData/Data_DataFrame.csv", index=False, na_rep='NaN')

    #Concat data frames
    df = pd.concat([og_df, rf_df], ignore_index=True)

    #Save the Data Frame
    df.to_csv(os.getcwd()[:-12] + "/All_Data_DataFrame.csv", index=False, na_rep='NaN')

def plt_methods_by_CSV_max(df, sort_by = "DIG", metric = "Combined_Metric", return_df =False, plot_methods = ["SSMA", "MAGAN", "DTA", "SPUD_D", "SPUD_M", "DIG", "CwDIG", "NAMA", "PCR", "JLMA", "Split_A", "Split_B"]):
    """df should equal the dataframe. It can be subsetted already
    
    Plots the max of the combined metric for each method to each CSV_File
    
    sort_by should be the string of what the method you want"""

    # Filter the DataFrame
    #df = df[~df["csv_file"].isin(["b", "blobs", "S-c", "S-curve"])]

    global og_df
    global rf_df

    if metric == "FOSCTTM": #Because for the FOSCTTM the smaller score is better
        agregate_df = pd.DataFrame({
            'SSMA': df[df["method"] == "SSMA"].groupby("csv_file")[metric].min(),
            'MAGAN': df[df["method"] == "MAGAN"].groupby("csv_file")[metric].min(),
            'DTA': df[df["method"] == "DTA"].groupby("csv_file")[metric].min(),
            'SPUD_D': df[df["algorithm"]== "distance"].groupby("csv_file")[metric].min(),
            'SPUD': df[df["algorithm"] == "merge"].groupby("csv_file")[metric].min(),
            'DIG': df[df["method"] == "DIG"].groupby("csv_file")[metric].min(),
            'CwDIG': df[df["method"] == "CwDIG"].groupby("csv_file")[metric].min(),
            'NAMA': df[df["method"] == "NAMA"].groupby("csv_file")[metric].min(),
            'PCR': df[df["method"] == "PCR"].groupby("csv_file")[metric].min(),
            'JLMA': df[df["method"] == "JLMA"].groupby("csv_file")[metric].min(),

            # 'Split_A': og_df.groupby("csv_file")["A_Classification_Score"].min(), #These Don't make sense
            # 'Split_B': og_df.groupby("csv_file")["B_Classification_Score"].min(),
            # 'RFBL2': rf_df.groupby("csv_file")["A_Classification_Score"].max(),
            # 'RFBL2': rf_df.groupby("csv_file")["B_Classification_Score"].min(),

            'MASH_RF': df[df["method"] == "MASH_RF"].groupby("csv_file")[metric].min(),
            'MALI_RF': df[df["method"] == "MALI_RF"].groupby("csv_file")[metric].min(),
            'MALI': df[df["method"] == "MALI"].groupby("csv_file")[metric].min(),
            'KEMA_RF': df[df["method"] == "KEMA_RF"].groupby("csv_file")[metric].min(),
            'SPUD_RF': df[df["method"] == "SPUD_RF"].groupby("csv_file")[metric].min()
        })
    else:
        agregate_df = pd.DataFrame({
            'SSMA': df[df["method"] == "SSMA"].groupby("csv_file")[metric].max(),
            'MAGAN': df[df["method"] == "MAGAN"].groupby("csv_file")[metric].max(),
            'DTA': df[df["method"] == "DTA"].groupby("csv_file")[metric].max(),
            'SPUD_D': df[df["algorithm"]== "distance"].groupby("csv_file")[metric].max(),
            'SPUD': df[df["algorithm"] == "merge"].groupby("csv_file")[metric].max(),
            'DIG': df[df["method"] == "DIG"].groupby("csv_file")[metric].max(),
            'CwDIG': df[df["method"] == "CwDIG"].groupby("csv_file")[metric].max(),
            'NAMA': df[df["method"] == "NAMA"].groupby("csv_file")[metric].max(),
            'PCR': df[df["method"] == "PCR"].groupby("csv_file")[metric].max(),
            'JLMA': df[df["method"] == "JLMA"].groupby("csv_file")[metric].max(),

            'Split_A': og_df.groupby("csv_file")["A_Classification_Score"].max(),
            'Split_B': og_df.groupby("csv_file")["B_Classification_Score"].max(),
            'RFBL1': rf_df.groupby("csv_file")["A_Classification_Score"].max(),
            'RFBL2': rf_df.groupby("csv_file")["B_Classification_Score"].max(),

            'MASH_RF': df[df["method"] == "MASH_RF"].groupby("csv_file")[metric].max(),
            'MALI_RF': df[df["method"] == "MALI_RF"].groupby("csv_file")[metric].max(),
            'MALI': df[df["method"] == "MALI"].groupby("csv_file")[metric].max(),
            'KEMA_RF': df[df["method"] == "KEMA_RF"].groupby("csv_file")[metric].max(),
            'SPUD_RF': df[df["method"] == "SPUD_RF"].groupby("csv_file")[metric].max()
        })

    agregate_df = agregate_df.sort_values(by = sort_by).reset_index()

    #If we only want the df
    if return_df:
        return agregate_df

    #To make it easier to add edits
    key_words = {"x" : agregate_df.index - 0.13,
                "s" : 50,
                "alpha" : .90,
                #"edgecolor" : "black",
                #"linewidth": 0.5,
                #"facecolor": "None"
                }

    plt.figure(figsize=(16, 6))
    
    if "DIG" in plot_methods:
        ax = plt.scatter(y = agregate_df["DIG"], marker = '^', color = "green", label = "MASH", **key_words)
    if "MAGAN" in plot_methods:
        ax = plt.scatter(y = agregate_df["MAGAN"], marker = '^', color = "red", label = "MAGAN", **key_words)
    if "JLMA" in plot_methods:
        ax = plt.scatter(y = agregate_df["JLMA"], marker = '^', label = "JLMA", **key_words)
    if "SPUD_D" in plot_methods:
        ax = plt.scatter(y = agregate_df["SPUD_D"], marker = "^", color = "None", edgecolor = "blue", linewidth= 1.3, label = "SPUD_D", **key_words)
    if "Split_A" in plot_methods:
        ax = plt.scatter(y = agregate_df["Split_A"], marker = '.', color = "black", label = "Split A", **key_words)
    if "CwDIG" in plot_methods:
        ax = plt.scatter(y = agregate_df["CwDIG"], marker = '^', color = "None",  edgecolor = "green", linewidth= 1.3, label = "MASH-", **key_words)
    if "NAMA" in plot_methods:
        ax = plt.scatter(y = agregate_df["NAMA"], marker = '^', label = "NAMA", **key_words)
    if "PCR" in plot_methods:
        ax = plt.scatter(y = agregate_df["PCR"], marker = '^', color = "brown", label = "Procrutees", **key_words)
    if "DTA" in plot_methods:
        ax = plt.scatter(y = agregate_df["DTA"], marker = "^", color = "orange", label = "DTA", **key_words)
    if "SPUD" in plot_methods:
        ax = plt.scatter(y = agregate_df["SPUD"], label = "SPUD", marker = '^', color = "blue", **key_words)
    if "SSMA" in plot_methods:
        ax = plt.scatter(y = agregate_df["SSMA"],  marker = '^', label = "SSMA", **key_words) 
    if "Split_B" in plot_methods:
        ax = plt.scatter(y = agregate_df["Split_B"], marker = '.', color = "red", label = "Split B", **key_words)


    #To make it easier to add edits
    key_words = {"x" : agregate_df.index + 0.13,
                "s" : 50,
                "alpha" : .90,
                #"edgecolor" : "black",
                #"linewidth": 0.5,
                #"facecolor": "None"
                }
    
    if "MASH_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["MASH_RF"], marker = 'o', color = "green", label = "MASH_RF", **key_words)
    if "MALI_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["MALI_RF"], marker = 'o', label = "MALI_RF", **key_words)
    if "MALI" in plot_methods:
        ax = plt.scatter(y = agregate_df["MALI"], marker = 'o', label = "MALI", **key_words)
    if "SPUD_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["SPUD_RF"], label = "SPUD_RF", marker = 'o', color = "blue", **key_words)
    if "RFBL2" in plot_methods:
        ax = plt.scatter(y = agregate_df["RFBL2"], marker = '.', color = "None", edgecolor = "red", linewidth= 1.3, label = "RFBL1", **key_words)
    if "KEMA_RF" in plot_methods:
        ax = plt.scatter(y = agregate_df["KEMA_RF"], marker = 'o', color = "purple", label = "KEMA", **key_words)
    if "RFBL1" in plot_methods:
        ax = plt.scatter(y = agregate_df["RFBL1"], marker = '.', color = "None", edgecolor = "black", linewidth= 1.3, label = "RFBL2", **key_words)


    #Show Legend
    plt.xticks(ticks= agregate_df.index,labels=agregate_df["csv_file"], rotation = 90)
    plt.title(f"{metric} Scores vs. CSV Files (MAX)")
    plt.ylabel(metric)
    plt.grid(visible=True, axis = "x")
    plt.legend()
    #plt.show()

def plt_methods_by_CSV_mean(df, sort_by = "SPUD", metric = "Combined_Metric", return_df = False, plot_methods = ["SSMA", "MASH_RF", "MALI_RF", "SPUD_RF", "MAGAN", "DTA", "SPUD_D", "SPUD_M", "DIG", "CwDIG", "NAMA", "PCR", "JLMA", "Split_A", "Split_B"]):
    """Plots 95 percent confident intervals for each method against csv files
    
    sort_by should be the string of what the method you want"""

    global og_df
    global rf_df

    # Filter the DataFrame
    #df = df[~df["csv_file"].isin(["b", "blobs", "S-c", "S-curve"])]

    agregate_df = pd.DataFrame({
        'SSMA': df[df["method"] == "SSMA"].groupby("csv_file")[metric].mean(),
        'MAGAN': df[df["method"] == "MAGAN"].groupby("csv_file")[metric].mean(),
        'DTA': df[df["method"] == "DTA"].groupby("csv_file")[metric].mean(),
        'SPUD': df[df["method"] == "SPUD"].groupby("csv_file")[metric].mean(),
        'DIG': df[df["method"] == "DIG"].groupby("csv_file")[metric].mean(),
        'CwDIG': df[df["method"] == "CwDIG"].groupby("csv_file")[metric].mean(),
        'NAMA': df[df["method"] == "NAMA"].groupby("csv_file")[metric].mean(),
        'JLMA': df[df["method"] == "JLMA"].groupby("csv_file")[metric].mean(),
        'PCR': df[df["method"] == "PCR"].groupby("csv_file")[metric].mean(),

        'Split_A': og_df.groupby("csv_file")["A_Classification_Score"].mean(),
        'Split_B': og_df.groupby("csv_file")["B_Classification_Score"].mean(),
        'RFBL1': rf_df.groupby("csv_file")["A_Classification_Score"].mean(),
        'RFBL2': rf_df.groupby("csv_file")["B_Classification_Score"].mean(),

        'MASH_RF': df[df["method"] == "MASH_RF"].groupby("csv_file")[metric].mean(),
        'MALI_RF': df[df["method"] == "MALI_RF"].groupby("csv_file")[metric].mean(),
        'MALI': df[df["method"] == "MALI"].groupby("csv_file")[metric].mean(),
        'KEMA_RF': df[df["method"] == "KEMA_RF"].groupby("csv_file")[metric].mean(),
        'SPUD_RF': df[df["method"] == "SPUD_RF"].groupby("csv_file")[metric].mean()
    })

    #Calculate error bars
    err_df = pd.DataFrame({
        'SSMA': df[df["method"] == "SSMA"].groupby("csv_file")[metric].std(),
        'MAGAN': df[df["method"] == "MAGAN"].groupby("csv_file")[metric].std(),
        'DTA': df[df["method"] == "DTA"].groupby("csv_file")[metric].std(),
        'SPUD': df[df["method"] == "SPUD"].groupby("csv_file")[metric].std(),
        'DIG': df[df["method"] == "DIG"].groupby("csv_file")[metric].std(),
        'CwDIG': df[df["method"] == "CwDIG"].groupby("csv_file")[metric].std(),
        'NAMA': df[df["method"] == "NAMA"].groupby("csv_file")[metric].std(),
        'JLMA': df[df["method"] == "JLMA"].groupby("csv_file")[metric].std(),
        'PCR': df[df["method"] == "PCR"].groupby("csv_file")[metric].std(),

        'Split_A': og_df.groupby("csv_file")["A_Classification_Score"].std(),
        'Split_B': og_df.groupby("csv_file")["B_Classification_Score"].std(),
        'RFBL1': rf_df.groupby("csv_file")["A_Classification_Score"].std(),
        'RFBL2': rf_df.groupby("csv_file")["B_Classification_Score"].std(),

        'MASH_RF': df[df["method"] == "MASH_RF"].groupby("csv_file")[metric].std(),
        'MALI_RF': df[df["method"] == "MALI_RF"].groupby("csv_file")[metric].std(),
        'MALI': df[df["method"] == "MALI"].groupby("csv_file")[metric].std(),
        'KEMA_RF': df[df["method"] == "KEMA_RF"].groupby("csv_file")[metric].std(),
        'SPUD_RF': df[df["method"] == "SPUD_RF"].groupby("csv_file")[metric].std()
    })

    agregate_df = agregate_df.sort_values(by = sort_by).reset_index()

    #If we only want the df
    if return_df:
        return agregate_df

    #To make it easier to add edits
    key_words = {"ms" : 8,
                "alpha" : .75}

    plt.figure(figsize=(20, 6))
    if "DTA" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index - 0.1, y = agregate_df["DTA"], yerr = err_df["DTA"], fmt = ".", label = "DTA", **key_words)
    if "SPUD" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index, y = agregate_df["SPUD"], yerr = err_df["SPUD"], fmt = ".", label = "SPUD", **key_words) 
    if "SPUD_RF" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index, y = agregate_df["SPUD_RF"], yerr = err_df["SPUD_RF"], fmt = ".", label = "SPUD_RF", **key_words) 
    if "DIG" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.2, y = agregate_df["DIG"], yerr = err_df["DIG"],fmt = '.', label = "DIG", **key_words)
    if "MASH_RF" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.2, y = agregate_df["MASH_RF"], yerr = err_df["MASH_RF"],fmt = '.', label = "MASH_RF", **key_words)
    if "CwDIG" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.2, y = agregate_df["CwDIG"], yerr = err_df["CwDIG"],fmt = '.', label = "CwDIG", **key_words)
    if "SSMA" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.1, y = agregate_df["SSMA"], yerr = err_df["SSMA"],fmt = '.', label = "SSMA", **key_words)
    if "NAMA" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index - 0.2, y = agregate_df["NAMA"], yerr = err_df["NAMA"],fmt = '.', label = "NAMA", **key_words)
    if "MAGAN" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.3, y = agregate_df["MAGAN"], yerr = err_df["MAGAN"],fmt = '.', color = "black", label = "MAGAN", **key_words)
    if "MALI_RF" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.3, y = agregate_df["MALI_RF"], yerr = err_df["MALI_RF"],fmt = '.', color = "gray", label = "MALI_RF", **key_words)
    if "MALI" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.2, y = agregate_df["MALI"], yerr = err_df["MALI"],fmt = '.', color = "black", label = "MALI", **key_words)
    if "KEMA_RF" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.1, y = agregate_df["KEMA_RF"], yerr = err_df["KEMA_RF"],fmt = '.', label = "KEMA_RF", **key_words)
    if "JLMA" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index, y = agregate_df["JLMA"], yerr = err_df["JLMA"], fmt = ".", label = "JMLA", **key_words) 
    if "PCR" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index - 0.3, y = agregate_df["PCR"], yerr = err_df["PCR"], fmt = ".", label = "Procrustees", **key_words) 

    if "Split_A" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.4, y = agregate_df["Split_A"], yerr = err_df["Split_A"], fmt = "_", label = "Split_A", **key_words) 
    if "Split_B" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index - 0.4, y = agregate_df["Split_B"], yerr = err_df["Split_B"], fmt = "_", label = "Split_B", **key_words) 
    if "RFBL1" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index + 0.4, y = agregate_df["RFBL1"], yerr = err_df["RFBL1"], fmt = "_", label = "RFBL1", **key_words) 
    if "RFBL2" in plot_methods:
        ax = plt.errorbar(x = agregate_df.index - 0.2, y = agregate_df["RFBL2"], yerr = err_df["RFBL2"], fmt = "_", label = "RFBL2", **key_words) 

    plt.ylim([-0.3, 1])

    #Show Legend
    plt.xticks(ticks= agregate_df.index,labels=agregate_df["csv_file"], rotation = 90)
    plt.title(f"{metric} Scores vs. CSV Files (MEAN)")
    plt.ylabel(metric)
    plt.grid(visible=True, axis = "x")
    plt.legend()
    plt.show()

def compare_with_baseline(scoring = "Combined_Metric", verbose = 0,  **kwargs):
    """Tells us how often our Methods are better than the Baseline
    
    Returns a pandas dataframe"""

    #Add DF 
    if "df" not in kwargs.keys():
        kwargs["df"] = df

    #Get resulting Df and get the split data
    rankdf = plt_methods_by_CSV_max(df = subset_df(**kwargs), metric = scoring, return_df=True)

    #Only use applicable files
    # Filter DataFrame to keep only rows where `csv_file` matches entries in `files_to_keep`
    rankdf = rankdf[rankdf['csv_file'].isin(['Cancer_Data', 'balance_scale', 'breast_cancer', 'crx', 'diabetes', 'ecoli_5', 'flare1', 'glass', 'heart_disease', 'heart_failure', 'hepatitis', 'ionosphere', 'iris', 'parkinsons', 'seeds','tic-tac-toe'])]
    
    rankdf = rankdf.dropna()

    #Print out the used csv_files
    if verbose > 0:
        print(list(rankdf["csv_file"]))
        print(len(rankdf["csv_file"]))

    rankdf = rankdf.drop(columns = "csv_file").rank(axis=1)

    # Drop the columns 'Split_A', 'Split_B', 'RFBL1', 'RFBL2' as they're not to be compared
    comparison_columns = rankdf.columns.difference(['Split_A', 'Split_B', 'RFBL1', 'RFBL2'])

    # Dictionary to store the counts of values higher than both baselines
    rf_scores_both = {}
    og_scores_both = {}
    rf_scores_one = {}
    og_scores_one = {}

    # Iterate over each column (besides Split_A, Split_B, and baselines)
    for col in comparison_columns:
        total = np.array(np.isnan(rankdf[col]) == False).sum() / 100 #To get it as a percent
        if total == 0:
            rf_scores_both[col] = np.nan
            rf_scores_one[col] = np.nan
            og_scores_both[col] = np.nan
            og_scores_one[col] = np.nan

        else:
            # Count how many times the value in each column is higher than both RFBL1 and RFBL2
            count = ((rankdf[col] > rankdf['RFBL1']) & (rankdf[col] > rankdf['RFBL2'])).sum()
            rf_scores_both[col] = np.round(count / total, decimals = 0)

            # Count how many times the value in each column is higher than RFBL1 or RFBL2
            count = ((rankdf[col] > rankdf['RFBL1']) | (rankdf[col] > rankdf['RFBL2'])).sum()
            rf_scores_one[col] = np.round(count / total, decimals = 0)

            # Count how many times the value in each column is higher than both Knn baselines
            count = ((rankdf[col] > rankdf['Split_A']) & (rankdf[col] > rankdf['Split_B'])).sum()
            og_scores_both[col] = np.round(count / total, decimals = 0)

            # Count how many times the value in each column is higher than one of the Knn baselines
            count = ((rankdf[col] > rankdf['Split_A']) | (rankdf[col] > rankdf['Split_B'])).sum()
            og_scores_one[col] = np.round(count / total, decimals = 0)

    return pd.DataFrame((rf_scores_one, rf_scores_both, og_scores_one, og_scores_both), 
                        index = ["MA Exceeds a Domain (RF)", "MA Exceeds Both Domains (RF)",  "MA Exceeds a Domain (KNN)", "MA Exceeds Both Domains (KNN)"])

def get_mean_std_df(split = "all", scoring = "Combined_Metric", columns_to_drop = ["MASH_RF", "MALI_RF", "KEMA_RF", "SPUD_RF", "MALI"], **kwargs):

    #Add DF 
    if "df" not in kwargs.keys():
        kwargs["df"] = df

    #Add the values of all the dfs together
    if split == "all":
        #Create the base set
        split_df = plt_methods_by_CSV_max(df = subset_df(split = "turn", **kwargs), metric = scoring, return_df=True)
        split_df = split_df[split_df['csv_file'].isin(['Cancer_Data', 'balance_scale', 'breast_cancer', 'crx', 'diabetes', 'ecoli_5', 'flare1', 'glass', 'heart_disease', 'heart_failure', 'hepatitis', 'ionosphere', 'iris', 'parkinsons', 'seeds','tic-tac-toe'])]


        for s_type in ["distort", "even", "skewed", "random"]:
            #Add each of the sets to the dataframe
            split_df = split_df._append(plt_methods_by_CSV_max(df = subset_df(split = s_type, **kwargs), metric = scoring, return_df=True))
            split_df = split_df[split_df['csv_file'].isin(['Cancer_Data', 'balance_scale', 'breast_cancer', 'crx', 'diabetes', 'ecoli_5', 'flare1', 'glass', 'heart_disease', 'heart_failure', 'hepatitis', 'ionosphere', 'iris', 'parkinsons', 'seeds','tic-tac-toe'])]


    else:
        #Create the df 
        split_df = plt_methods_by_CSV_max(df = subset_df(split = split, **kwargs), metric = scoring, return_df=True)
        split_df = split_df[split_df['csv_file'].isin(['Cancer_Data', 'balance_scale', 'breast_cancer', 'crx', 'diabetes', 'ecoli_5', 'flare1', 'glass', 'heart_disease', 'heart_failure', 'hepatitis', 'ionosphere', 'iris', 'parkinsons', 'seeds','tic-tac-toe'])]


    #Drop unneeded columns
    csv_df = split_df.drop(columns= ["SPUD_D", "csv_file", "Split_A", "Split_B"] + columns_to_drop).dropna()
    
    #Get column size 
    n_cols = len(csv_df.columns)

    #Create csv df
    csv_df_std = pd.DataFrame(np.reshape(np.std(csv_df.to_numpy(), axis = 0), newshape = (1,n_cols)), columns = csv_df.columns)
    csv_df = csv_df_std._append((pd.DataFrame(np.reshape(np.mean(csv_df.to_numpy(), axis = 0), newshape = (1,n_cols)), columns = csv_df.columns)._append(pd.DataFrame(np.reshape(np.mean(csv_df.to_numpy(), axis = 0), newshape = (1,n_cols)), columns = csv_df.columns).rank(ascending=False, method='max', axis = 1))).reset_index().sort_values(by = 0, ascending = False, axis=1).drop(columns = ["index"]))


    #Rename the index
    csv_df.index = ["STD", "Mean", "rankings"]

    return csv_df

def plot_ranks(scoring = "Combined_Metric", **kwargs):
    """Kwargs are for the get_mean_std_df.
    
    Its parameters are: split, scoring, columns_to_drop, and kwargs for subset df"""

    #Create a dataframe to add too:
    agregate_df = get_mean_std_df(scoring = scoring, **kwargs).head(2)

    #Add all the other dfs
    for kind in ["random", "skewed", "even", "distort", "turn"]:
        agregate_df = agregate_df._append(get_mean_std_df(scoring = scoring, split = kind, **kwargs).head(2))
    
    #Reset index
    agregate_df.index = ["STD - all", "MEAN - all", "STD - random", "MEAN - random","STD - skewed", "MEAN - skewed","STD - even", "MEAN - even","STD - distort", "MEAN - distort","STD - turn", "MEAN - turn"]

    # Sort columns based on the 'MEAN - all' row
    sorted_columns = agregate_df.loc['MEAN - all'].sort_values(ascending = False).index
    agregate_df = agregate_df[sorted_columns]
    columns = agregate_df.columns
    x = np.array(range(len(columns))) / 15
    
    # Plotting
    plt.figure(figsize=(14, 8))

    legend = True

    
    # Define the colorblind-friendly colors
    colors = ['#000000', '#e41a1c', '#377eb8', 
              '#4daf4a', '#984ea3', '#ff7f00', 
              '#a65628', '#f781bf', '#999999']
    
    for i in range(0, len(agregate_df), 2):
        std_values = agregate_df.iloc[i].values
        mean_values = agregate_df.iloc[i+1].values

        for pos, mean, std, label, c in zip(x, mean_values, std_values, columns, colors):
            
            #change the format
            if pos < x[int(len(x)/2)]:
                fmt = 'o'
            else:
                fmt = "^"

            #Rename DIG to MAD
            if label == "DIG":
                label = "MASH-"
            elif label == 'CwDIG':
                label = "MASH"
            elif label == 'PCR':
                label = "MAPA"
            elif label == "MASH_RF":
                label = "RF-MASH"
            elif label == "MALI_RF":
                label = "RF-MALI"
            elif label == "SPUD_RF":
                label = "RF-SPUD"
            elif label == "KEMA_RF":
                label = "KEMA"

            plt.errorbar(pos - 0.3, mean, yerr=std, fmt=fmt, label=label, elinewidth= 2, color = c, ms = 10, capsize=5)

        if legend:
            plt.legend(fontsize = 16, loc = (0.635, 0.01))
            legend = False

        #Move the positions over
        x += 1
    
    
    plt.xticks(np.array(range(0,6)) - 0.04, ["all", "random", "skewed", "even", "distort", "rotation"], fontsize = 20, rotation=0)
    plt.yticks(fontsize=16)
    #plt.xlabel('Domain Adaptation Methods', fontsize=15)
    plt.ylabel('Combined Metric', fontsize=20)
    plt.title('Comparison of Results', fontsize=25)
    plt.grid(visible=True, axis = "y")
    plt.tight_layout()
    plt.show()
    
def line_plot_methods(df_subset, argument = "Percent_of_Anchors", metric = "Combined_Metric", plt_legend = False, custom_title = False):
    "subset should equal none or the csv file name"

    #Create figure
    plt.figure(figsize=(14,7))

    # Group by 'csv_file' and 'argument' to get the max 'metric' for each file and argument
    SSMA = df_subset[df_subset["method"] == "SSMA"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]
    DIG = df_subset[df_subset["method"] == "DIG"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]
    CwDIG = df_subset[df_subset["method"] == "CwDIG"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]
    JLMA = df_subset[df_subset["method"] == "JLMA"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]
    MAGAN = df_subset[df_subset["method"] == "MAGAN"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]
    MAPA = df_subset[df_subset["method"] == "PCR"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]
    SPUD = df_subset[df_subset["method"] == "SPUD"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]
    DTA = df_subset[df_subset["method"] == "DTA"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]


    #Keywords arguments
    line_styles = {"linewidth" : 3,
                   "linestyle" : '-'
                   }
    
    scatter_styles = {"s" : 100}



    #Plot the graphs
    plt.plot(SPUD.mean(), label = "SPUD", color = '#000000', **line_styles)
    plt.plot(CwDIG.mean(), label = "MASH", color ='#e41a1c', **line_styles)
    plt.plot(DIG.mean(), label = "MASH-", color =  '#377eb8', **line_styles)    
    plt.plot(DTA.mean(), label = "DTA", color = "#4daf4a", **line_styles)
    plt.plot(MAGAN.mean(), label = "MAGAN", color = "#984ea3", **line_styles)

    if argument != "Percent_of_KNN":
        NAMA = df_subset[df_subset["method"] == "NAMA"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]
        plt.plot(NAMA.mean(), label = "NAMA", color = "#ff7f00", **line_styles)
        
    plt.plot(SSMA.mean(), label = "SSMA", color = "#a65628", **line_styles) #I want the max, and then the mean of the csvs    
    plt.plot(MAPA.mean(), label = "MAPA", color = "#f781bf", **line_styles )
    plt.plot(JLMA.mean(), label = "JLMA", color = "#999999", **line_styles)


    """#Scatter plot
    plt.errorbar(x = [0,1,2,3,4,5], y = SPUD.mean(), label = "SPUD", color = '#000000', yerr = SPUD.std()/2, **scatter_styles)
    plt.errorbar(x = np.array([0,1,2,3,4,5]) + 0.1/2, y =CwDIG.mean(), yerr = CwDIG.std()/2, label = "MASH", color ='#e41a1c', **scatter_styles)
    plt.errorbar(x = np.array([0,1,2,3,4,5]) - 0.1/2, y = DIG.mean(), yerr = DIG.std()/2, label = "MASH-", color =  '#377eb8', **scatter_styles)    
    plt.errorbar(x = np.array([0,1,2,3,4,5]) + 0.2/2, y = DTA.mean(),yerr = DTA.std()/2,  label = "DTA", color = "#4daf4a", **scatter_styles)
    plt.errorbar(x = np.array([0,1,2,3,4,5]) + 0.3/2, y = MAGAN.mean(),yerr = MAGAN.std()/2, label = "MAGAN", color = "#984ea3", **scatter_styles)

    if argument != "Percent_of_KNN":
        NAMA = df_subset[df_subset["method"] == "NAMA"].groupby(['csv_file', argument])[metric].max().reset_index().groupby(argument)[metric]
        plt.errorbar(x = np.array([0,1,2,3,4,5]) -0.2/2, y = NAMA.mean(), yerr = NAMA.std()/2, label = "NAMA", color = "#ff7f00", **scatter_styles)
        
    plt.errorbar(x = np.array([0,1,2,3,4,5]) - 0.3/2, y = SSMA.mean(), yerr = SSMA.std()/2, label = "SSMA", color = "#a65628", **scatter_styles) #I want the max, and then the mean of the csvs    
    plt.errorbar(x = np.array([0,1,2,3,4,5]) + 0.4/2 , y = MAPA.mean(), yerr = MAPA.std()/2, label = "MAPA", color = "#f781bf", **scatter_styles )
    plt.errorbar(x = np.array([0,1,2,3,4,5]) - 0.4/2, y = JLMA.mean(), yerr = JLMA.std()/2, label = "JLMA", color = "#999999", **scatter_styles)

    """

    import seaborn as sns
    sns.scatterplot(CwDIG.mean(), color ='#e41a1c', **scatter_styles)
    sns.scatterplot(DIG.mean(), color =  '#377eb8', **scatter_styles)    
    sns.scatterplot(DTA.mean(),  color = "#4daf4a", **scatter_styles)
    sns.scatterplot(MAGAN.mean(), color = "#984ea3", **scatter_styles)

    if argument != "Percent_of_KNN":
        NAMA = df_subset[df_subset["method"] == "NAMA"].groupby(['csv_file', argument])[metric].max().reset_index()
        sns.scatterplot(NAMA.groupby(argument)[metric].mean(),  color = "#ff7f00", **scatter_styles)
        
    sns.scatterplot(SSMA.mean(), color = "#a65628", **scatter_styles) #I want the max, and then the mean of the csvs    
    sns.scatterplot(MAPA.mean(), color = "#f781bf", **scatter_styles )
    sns.scatterplot(JLMA.mean(), color = "#999999", **scatter_styles)



    #Make it pretty
    plt.xlabel(argument.replace('_', " "), fontsize = 20)
    plt.ylabel(metric.replace('_', " "), fontsize = 20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if plt_legend != False:
        plt.legend(fontsize=16, loc = plt_legend, ncol = 3)

    if custom_title != False:
        plt.title(custom_title, fontsize = 25)

def plot_param_heat_map(df, parameters, method, metric = "Combined_Metric", figsize=(18, 10)):
    """ Parameters formated like ['Percent_of_KNN', 'Percent_of_Anchors']
        Metric formated like "Combined_Metric"
        Method formated like "Spud"
    """

    df["Percent_of_KNN"] = df["Percent_of_KNN"].round(2)

    #Subset the data to only things we want
    df_params = subset_df(df = df, method = method)[parameters + [metric]].dropna()

    # Melt the dataframe to long format for seaborn
    df_melted = df_params.melt(id_vars=parameters, value_vars=[metric]).drop_duplicates()

    # Group by parameters and calculate the mean of 'value'
    df_mean = df_melted.groupby(parameters)['value'].mean().reset_index()

    #Create the figure
    plt.figure(figsize=figsize)

    #Create pivot table we can plot
    if len(parameters) > 3:
        df_pivot = df_mean.pivot_table(index=parameters[:2], columns=parameters[2:], values = ["value"])
        index_levels = len(df_pivot.index.levels[0]) 
    else:
        df_pivot = df_mean.pivot_table(index=parameters[:1], columns=parameters[1:], values = ["value"])
        index_levels = 0.8


    df_pivot.fillna(0, inplace = True)

    # Create the heatmap
    ax = sns.heatmap(df_pivot, annot=True, cmap='viridis')

    #Make it pretty
    plt.title(f'{method}\'s Parameter Heatmap Colored by {metric.replace("_", " ")}', fontsize = 20)
    plt.xlabel("")
    plt.ylabel("")

    if len(parameters) > 2:
        # Special art if we have multiple categories
        num_headings = len(set([index[-1] for index in df_pivot.columns]))

        #Customize x-ticks labels
        ax.set_xticks(np.arange(df_pivot.shape[1]) + 0.5)
        ax.set_xticklabels([label for label in df_pivot.columns.get_level_values(2)], rotation=90, fontsize = 13)

        # Draw a vertical black line every x columns if we have multiple categories
        for i in range(0, df_pivot.shape[1] + 1, num_headings):
            plt.axvline(x=i, color='black', linewidth = 5)

        # Add the second level of labels
        for i in range(0, df_pivot.shape[1], num_headings):
            ax.text(i + (num_headings * 0.5), (figsize[1] * index_levels)+3, df_pivot.columns.get_level_values(1)[i], ha='center', va='center', fontsize=18, rotation=0)


    if len(parameters) > 3:

        # Special art if we have four categories
        num_headings = len(set([index[-1] for index in df_pivot.index]))


        #Customize x-ticks labels
        ax.set_yticks(np.arange(df_pivot.shape[0]) + 0.5)
        ax.set_yticklabels([label for label in df_pivot.index.get_level_values(1)], rotation=0, fontsize = 13)

        # Draw a vertical black line every x columns if we have multiple categories
        for i in range(0, df_pivot.shape[0] + 1, num_headings):
            plt.axhline(y=i, color='black', linewidth = 5)

        # Add the second level of labels
        for i in range(0, df_pivot.shape[0], num_headings):
            ax.text(-4, i + (num_headings * 0.5), df_pivot.index.get_level_values(0)[i], ha='center', va='center', fontsize=18, rotation=90)

    else:
        plt.ylabel(parameters[0].replace("_", " "), fontsize = 20)
        plt.yticks(fontsize = 13)


    plt.show()
    

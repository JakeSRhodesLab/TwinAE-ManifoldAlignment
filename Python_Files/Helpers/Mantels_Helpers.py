#Imports 
from regression_helpers import read_json_files_to_dataframe
import os
import numpy as np
import pandas as pd
import random
from Pipeline_Helpers import method_dict, create_unique_pairs
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from Grae import GRAEBase, BaseDataset

class split_data():
    """Made to spoof the TMA class but is lightweight"""
    def __init__(self, csv_file, split):
        #Get the labels
        try:
            df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Classification_CSV/" + csv_file)
        except:
            df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Regression_CSV/" + csv_file)

        #If categorical strings :)
        if df[df.columns[0]].dtype == 'object':
            df[df.columns[0]] = pd.Categorical(df[df.columns[0]]).codes

        self.labels = np.array(df.pop(df.columns[0]))

        #Upload cached split
        self.split_A, self.split_B = split_features(csv_file, split, 42) #NOTE: We usually use 42 as a seed

        #Anchors are assumed to be .3 of the dataset
        self.anchors = create_unique_pairs(self.split_A.shape[0], int(self.split_A.shape[0] * .3))

#Directory Constant
def split_features(csv_file, split, seed):

        #Step 1. Check if a file exists already
        #Create filename 
        filename = "/yunity/arusty/Graph-Manifold-Alignment/Results/Splits_Data/" + csv_file[:-4] + "/"
        filename += split[0] + str(seed) + ".npz"

        #Step 2b. If so, simply load the files into split A and split B
        if os.path.exists(filename):

            #Load in the file
            data = np.load(filename) 

            #Grab the splits
            return data['split_a'], data["split_b"]
        
        else:
            from Main.test_manifold_algorithms import test_manifold_algorithms as tma
            tma(csv_file, random_state= seed, split = split)
            print(f"Splitting {csv_file} with seed {seed} and split {split} complete.")
            split_features(csv_file, split, seed)

# Create function to Extract best fit information from the results
def extract_all_files():
    #Get the regression results and classification results
    df = read_json_files_to_dataframe("/yunity/arusty/Graph-Manifold-Alignment/Results")

    #Filter the dataframe to only include the valid methods
    filtered_df = df[df["method"].isin(["SPUD", "RF-SPUD", "NAMA", "RF-NAMA", "MASH", "MASH-", "RF-MASH", "RF-MASH-"])]

    return filtered_df

# Create a function to get the parameters for a parrelization loop
def create_tasks_for_parrelization(df):
    #Create the task list
    tasks = []

    #Iterate through the dataframe
    for index, row in df.iterrows():
        #Get the parameters, method, and dataset
        params = row["Best_Params"]
        method = row["method"]
        dataset = row["csv_file"]
        split = row["split"]

        #Create the task
        task = (method, dataset, split, params)

        #Append the task to the tasks list
        tasks.append(task)  

    return tasks

# To keep code cleaner
def create_and_fit_method(method_data, data, params):
    if method_data["Name"] == "MASH" or method_data["Name"] == "RF-MASH":
        method_class = method_data["Model"](knn = params["knn"], page_rank = params["page_rank"], DTM = params["DTM"], density_normalization = params["density_normalization"])
        method_class = method_data["Fit"](method_class, data, data.anchors[len(data.anchors)//2:])
        method_class.optimize_by_creating_connections(threshold = params["threshold"], connection_limit = params["connection_limit"], epochs = params["epochs"])

    else:
        method_class = method_data["Model"](**params)
        method_class = method_data["Fit"](method_class, data, data.anchors)
    
    return method_class

# Create function to create the embeddings (One with excluded test points) from Mash or SPUD
def get_embeddings(method, dataset, split, params, *, return_labels = False):

    #Create a TMA spoof class
    data = split_data(dataset + ".csv", split)

    #Create a custom MDS where we keep only 1 job (Not to have nested parrelization)
    n_comps = max(min(data.split_A.shape[1], data.split_B.shape[1]), 2) #Ensures the min is 2 or the lowest data split dimensions
    mds = MDS(metric=True, dissimilarity = 'precomputed', n_init = 4,
                n_jobs=1, random_state = 42, n_components = n_comps) 

    #Get the method data, fit it and prepare it to extract the block
    method_data = method_dict[method]
    method_class = create_and_fit_method(method_data, data, params)

    #Get the true embedding
    emb_full = mds.fit_transform(method_data["Block"](method_class))
    #print("Full Embedding Complete")

    if return_labels:
        normal_labels = np.vstack([data.labels, data.labels])

    """GET GRAE's EMBEDDING below"""
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(data.split_A, data.labels[:len(data.split_A)], test_size=0.2, random_state=42)
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(data.split_B, data.labels[:len(data.split_B)], test_size=0.2, random_state=42)

    # Reformat data using x train
    data.anchors = create_unique_pairs(len(X_A_train), int(data.split_A.shape[0] * .3)) #NOTE: we choose to keep the same amount of anchors. 
    data.split_A = X_A_train
    data.split_B = X_B_train
    data.labels = y_A_train # y_A_train will equal y_B_train

    method_class = create_and_fit_method(method_data, data, params)

    #Get the partial embedding
    emb_partial = mds.fit_transform(method_data["Block"](method_class))
    #print("Partial Embedding Complete")


    #GRAE on domain A
    myGrae = GRAEBase(n_components = n_comps)
    split_A = BaseDataset(x = X_A_train, y = y_A_train, split_ratio = 0.8, random_state = 42, split = "none")
    myGrae.fit(split_A, emb = emb_partial[:len(X_A_train)])
    testA = BaseDataset(x = X_A_test, y = y_A_test, split_ratio = 0.8, random_state = 42, split = "none")
    pred_A, _ = myGrae.score(testA)

    #Grae on domain B 
    myGrae = GRAEBase(n_components = n_comps)
    split_B = BaseDataset(x = X_B_train, y = y_B_train, split_ratio = 0.8, random_state = 42, split = "none")
    myGrae.fit(split_B, emb = emb_partial[int(len(emb_partial)/2):])
    testB = BaseDataset(x = X_B_test, y = y_B_test, split_ratio = 0.8, random_state = 42, split = "none")
    pred_B, _ = myGrae.score(testB)
    
    #Grab the scores
    A_train = emb_partial[:int(len(emb_partial)/2)]
    B_train = emb_partial[int(len(emb_partial)/2):]
    emb_pred = np.vstack([A_train, pred_A, B_train, pred_B]) #NOTE: Train on just train
    #print("GRAE Embedding Complete")

    if return_labels:
        return emb_partial, emb_pred, emb_full, normal_labels, np.hstack([y_A_train, y_A_test, y_B_train, y_B_test])
    
    return emb_partial, emb_pred, emb_full

def stub_function_for_MARSHALL(method, dataset, split, params, *, return_labels = False): #DON'T Delete any of these parameters - though you can add your own if you want
    #Get the embeddings 

    # #NOTE: I'm assuming you want the labels Marshall. You may not, and you can switch this to be false
    emb_partial, emb_pred, emb_full, full_labels, pred_labels = get_embeddings(method, dataset, split, params, return_labels = return_labels)

    #TODO: Compare the embeddings in the results using Mantels correlation
    #NOTE: The results are a list of tuples. The elements of each tuple are "partial, pred, full, full_labels, pred_labels"
    

    #TODO: Save the result.

    #NOTE: This is set up to do comparision one task at a time so it can be trivially paralylized
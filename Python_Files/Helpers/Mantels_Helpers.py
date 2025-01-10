#Imports 
from regression_helpers import read_json_files_to_dataframe
import os
import numpy as np
import pandas as pd
import random
from Pipeline_Helpers import method_dict, create_unique_pairs
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split

class split_data():
    """Made to spoof the TMA class but is lightweight"""
    def __init__(self, csv_file, split):
        #Get the labels
        try:
            df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Classification_CSV/" + csv_file)
        except:
            df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Regression_CSV/" + csv_file)

        self.labels = df.pop(df.columns[0])

        #Upload cached split
        self.split_a, self.split_b = split_features(csv_file, split, 42) #NOTE: We usually use 42 as a seed

        #Anchors are assumed to be .3 of the dataset
        self.anchors = create_unique_pairs(self.split_a.shape[0], int(self.split_a.shape[0] * .3))

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
def extract_all_fits():
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


# Create function to create the embeddings (One with excluded test points) from Mash or SPUD
def get_embeddings(method, dataset, split, params):

    #Create a TMA spoof class
    data = split_data(dataset, split)

    #Create a custom MDS where we keep only 1 job (Not to have nested parrelization)
    mds = MDS(metric=True, dissimilarity = 'precomputed', n_init = 4,
                n_jobs=1, random_state = 42, n_components = max(min(data.split_a.shape[1], data.split_b.shape[1]),2)) #Ensures the min is 2 or the lowest data split dimensions

    #Get the method data, fit it and prepare it to extract the block
    method_data = method_dict[method]
    
    if method == "MASH" or method == "RF-MASH":
        method_class = method_data["Model"](knn = params["knn"], page_rank = params["page_rank"], DTM = params["DTM"], density_normalization = params["density_normalization"])
        method_class = method_data["Fit"](method_class, data, data.anchors[len(data.anchors)//2:])
        method_class.optimize_by_creating_connections(threshold = params["threshold"], connection_limit = params["connection_limit"], epochs = params["epochs"])

    else:
        method_class = method_data["Model"](**params)
        method_class = method_data["Fit"](method_class, data, data.anchors)

    #Get the true embedding
    emb_full = mds.fit_transform(method_data["Block"](method_class))

    #To avoid using the data on the text and Train, we will need to split it
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(data.split_A, data.labels[:len(data.split_A)], test_size=0.2, random_state=42)
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(data.split_B, data.labels[:len(data.split_B)], test_size=0.2, random_state=42)

    #Because of RF MASH, we need to specialize the initilization so we can call optimize on it later
    if method_data["Name"][-4:] == "MASH":

        optimize_dict = {"hold_out_anchors": data.anchors[len(data.anchors)//2:]}

        #Select half of the anchors for training
        anchors = optimize_dict["hold_out_anchors"][:int(len(tma.anchors) * tma.percent_of_anchors[0]/2)]
        
        for param in ["connection_limit", "threshold", "epochs"]:
            optimize_dict[param] = best_fit[param]
            del best_fit[param]

        #Delete hold out anchors
        if "hold_out_anchors" in best_fit.keys():
            del best_fit["hold_out_anchors"]
        
    else:
        anchors = create_unique_pairs(len(X_A_train), int(len(tma.anchors) * tma.percent_of_anchors[0]))

    #Fit using x train
    tma.split_A = X_A_train
    tma.split_B = X_B_train
    tma.labels = y_A_train # y_A_train will equal y_B_train
    
    #Create model
    rf_method_class = self.method_data["Model"](**self.overide_defaults, **best_fit)
    rf_method_class = self.method_data["Fit"](rf_method_class, tma, anchors)

    #Optimize for RF-MASH
    if self.method_data["Name"][-4:] == "MASH":
        rf_method_class.optimize_by_creating_connections(**optimize_dict)

    ##Create a custom MDS where we can run a higher n_init and n_jobs
    mds = MDS(metric=True, dissimilarity = 'precomputed', n_init = max(self.parallel_factor, 3),
            n_jobs=self.parallel_factor, random_state = seed, n_components = tma.n_comp)
    emb = mds.fit_transform(self.method_data["Block"](rf_method_class)) 

    #GRAE on domain A
    myGrae = GRAEBase(n_components = tma.n_comp)
    split_A = BaseDataset(x = X_A_train, y = y_A_train, split_ratio = 0.8, random_state = seed, split = "none")
    myGrae.fit(split_A, emb=emb[:len(X_A_train)])
    testA = BaseDataset(x = X_A_test, y = y_A_test, split_ratio = 0.8, random_state = seed, split = "none")
    pred_A, _ = myGrae.score(testA)

    #Grae on domain B 
    myGrae = GRAEBase(n_components = tma.n_comp)
    split_B = BaseDataset(x = X_B_train, y = y_B_train, split_ratio = 0.8, random_state = seed, split = "none")
    myGrae.fit(split_B, emb=emb[int(len(emb)/2):])
    testB = BaseDataset(x = X_B_test, y = y_B_test, split_ratio = 0.8, random_state = seed, split = "none")
    pred_B, _ = myGrae.score(testB)
    
    #Grab the scores
    A_train = emb[:int(len(emb)/2)]
    B_train = emb[int(len(emb)/2):]
    emb = np.vstack([A_train, pred_A, B_train, pred_B]) #NOTE: Train on just train
    knn_score, rf_score, knn_metric, rf_metric = get_embedding_scores(emb, (y_A_train, y_A_test, y_B_train, y_B_test), seed)

    #Methods with Andres fit have an enlarged embedding... so we need to concanenate the lables differently
    if self.method_data["Name"] in ["DTA", "SSMA", "MAPA"]:
        rf_oob_score = get_RF_score(emb, (tma.labels, y_A_test, tma.labels, y_B_test), seed)
    else:
        rf_oob_score = get_RF_score(emb, (y_A_train, y_A_test, y_B_train, y_B_test), seed)


    print(f"                GRAE KNN Score {knn_score}")
    print(f"                GRAE RF on embedding Score {rf_score}")
    print(f"                GRAE Random Forest out of bag score {rf_oob_score}")
    print(f"                GRAE KNN's f1 or Root mean square error score {rf_score}")
    print(f"                GRAE Random Forest f1 or Root mean square error score {rf_oob_score}")

    return rf_oob_score, knn_score, rf_score, knn_metric, rf_metric

        

    #Calculate the Grae Embedding
    
    return  emb_full



# Create the GRAE model evaluation
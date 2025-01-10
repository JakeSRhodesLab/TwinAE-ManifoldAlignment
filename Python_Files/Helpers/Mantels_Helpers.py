#Imports 
from regression_helpers import read_json_files_to_dataframe
import os
import numpy as np
import pandas as pd
import random
from Pipeline_Helpers import method_dict
from sklearn.manifold import MDS

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
        self.anchors = create_anchors(self.split_a.shape[0])

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

def create_anchors(dataset_size):
    """Returns an array of anchors equal to the datset size."""
    random.seed(42)

    #Generate anchors that can be subsetted
    rand_ints = random.sample(range(dataset_size), dataset_size)

    return np.vstack([rand_ints, rand_ints]).T

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

    #Get the method data
    method_data = method_dict[method]

    try:
        method_class = method_data["Model"](**params)
        method_class = method_data["Fit"](method_class, tma, anchors) #This usually just returns self, except with MAGAN

        # FOSCTTM Evaluation Metrics
        f_score = method_data["FOSCTTM"](method_class)

        #Create a custom MDS where we keep only 1 job (Not to have nested parrelization and lower n_init)
        mds = MDS(metric=True, dissimilarity = 'precomputed', n_init = 1,
                    n_jobs=1, random_state = self.seed, n_components = tma.n_comp)
        emb = mds.fit_transform(self.method_data["Block"](method_class))

        
        return f_score, c_score, emb

    except Exception as e:
        print(f"<><><>      Tests failed for: {test_parameters}. Why {e}        <><><>")
        logger.warning(f"Name: {self.method_data['Name']}. CSV: {self.csv_file}. Parameters: {test_parameters}. Error: {e}")
        return (np.NaN, np.NaN, np.NaN)


# Create the GRAE model evaluation
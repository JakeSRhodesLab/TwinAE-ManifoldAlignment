#Imports 
from Helpers.regression_helpers import read_json_files_to_dataframe
import os
import numpy as np
import pandas as pd
from Helpers.Pipeline_Helpers import method_dict, create_unique_pairs
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from Helpers.Grae import GRAEBase, BaseDataset
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from scipy.spatial.distance import pdist, squareform

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
            return split_features(csv_file, split, seed)

# Create function to Extract best fit information from the results
def extract_all_files():
    #Get the regression results and classification results
    df = read_json_files_to_dataframe("/yunity/arusty/Graph-Manifold-Alignment/Results")

    files = ["EnergyEfficiency",  
    "AutoMPG", "ComputerHardware", "diabetes", "tic-tac-toe", 'Medicaldataset',
    "hepatitis", "iris", "audiology", "parkinsons", "seeds", 
    "segmentation", "glass", "heart_disease", "heart_failure", "flare1", 
    "ecoli_5", "ionosphere", "Cancer_Data", "hill_valley", "balance_scale", 
    "CommunityCrime", "ConcreteSlumpTest", "Automobile", "ConcreteCompressiveStrength"
    ]

    #Filter the dataframe to only include the valid methods and exclude rf-mali
    filtered_df = df[df["csv_file"].isin(files) & (~df["method"].isin(["RF-MALI", "MALI-RF", "RF-SPUD", "RF-NAMA", 
                                                                                   "RF-MASH", "MALI", "RF-MASH-"]))]

    return filtered_df

# Create a function to get the parameters for a parrelization loop
def create_tasks_for_parrelization(df):
    #Create the task list
    tasks = []

    #Iterate through the dataframe
    for index, row in df.iterrows():

        for grae_build in ["anchor_loss", "original"]:
            #Get the parameters, method, and dataset
            params = row["Best_Params"]
            method = row["method"]
            dataset = row["csv_file"]
            split = row["split"]

            #Create the task
            task = (method, dataset, split, params, grae_build)

            #Append the task to the tasks list
            if dataset not in ["S-c", "b", "blobs", "blob", "S-curve"]:
                tasks.append(task)  

    return tasks

# To keep code cleaner
def create_and_fit_method(method_data, data, params):
    if method_data["Name"] == "MASH" or method_data["Name"] == "RF-MASH":
        method_class = method_data["Model"](knn = params["knn"], page_rank = params["page_rank"], DTM = params["DTM"], density_normalization = params["density_normalization"])
        method_class = method_data["Fit"](method_class, data, data.anchors[len(data.anchors)//2:])
        method_class.optimize_by_creating_connections(threshold = params["threshold"], connection_limit = params["connection_limit"], epochs = params["epochs"],
                                                      hold_out_anchors = data.anchors[:len(data.anchors)//2])

    else:
        method_class = method_data["Model"](**params)
        method_class = method_data["Fit"](method_class, data, data.anchors)
    
    return method_class

# Create function to create the embeddings (One with excluded test points) from Mash or SPUD
def get_embeddings(method, dataset, split, params,  lam = 100, *, return_labels = False):
    """
    Returns embeddings for the full and partial datasets using the specified method.
    Also returns the heatmap.
    """

    #Create a TMA spoof class
    data = split_data(dataset + ".csv", split)

    # Ensure both domains share the same shuffled indices
    indices = np.arange(len(data.split_A))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_size = int(0.8 * len(indices))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_A_train = data.split_A[train_idx]
    X_A_test = data.split_A[test_idx]
    y_A_train = data.labels[train_idx]
    y_A_test = data.labels[test_idx]
    X_B_train = data.split_B[train_idx]
    X_B_test = data.split_B[test_idx]
    y_B_train = data.labels[train_idx]
    y_B_test = data.labels[test_idx]

    #Recreate data to be in the same order
    data.split_A = np.vstack([X_A_train, X_A_test])
    data.split_B = np.vstack([X_B_train, X_B_test])
    data.labels = np.hstack([y_A_train, y_A_test]) #This will equal the same for B

    #Create a custom MDS where we keep only 1 job (Not to have nested parrelization)
    n_comps = max(min(data.split_A.shape[1], data.split_B.shape[1]), 2) #Ensures the min is 2 or the lowest data split dimensions
    mds = MDS(metric=True, dissimilarity = 'precomputed', n_init = 4,
                n_jobs=1, random_state = 42, n_components = n_comps) 

    #Get the method data, fit it and prepare it to extract the block
    method_data = method_dict[method]
    method_class = create_and_fit_method(method_data, data, params)

    #Get the true embedding
    block_full = method_data["Block"](method_class)
    emb_full = mds.fit_transform(method_data["Block"](method_class))
    #print("Full Embedding Complete")

    if return_labels:
        normal_labels = np.hstack([y_A_train, y_A_test, y_B_train, y_B_test])

    """GET GRAE's EMBEDDING below"""
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
    myGrae = GRAEBase(lam = lam, n_components = n_comps)
    split_A = BaseDataset(x = X_A_train, y = y_A_train, split_ratio = 0.8, random_state = 42, split = "none")
    myGrae.fit(split_A, emb = emb_partial[:len(X_A_train)])
    testA = BaseDataset(x = X_A_test, y = y_A_test, split_ratio = 0.8, random_state = 42, split = "none")
    pred_A, _ = myGrae.score(testA)

    #Grae on domain B 
    myGrae = GRAEBase(lam = lam, n_components = n_comps)
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
    
    return emb_pred, emb_full, block_full

def GRAE_tests(method, dataset, split, params, grae_build = "original", *, permutations = 10000, 
                plot = False, repeat_results = False): #DON'T Delete any of these parameters - though you can add your own if you want

    """
    Perform a Mantel test to compute the correlation between two embeddings
    
    Returns:
        r (float): Observed correlation.
        p_value (float): Significance of the observed correlation.
    """
    try:

        #Return null values if file already exsists
        if repeat_results == False:
            if file_already_exists(method, dataset, split, grae_build):
                #print(f"Results already exist for {method}, {dataset}, {split}.")
                
                if plot:
                    print("Plotting is disabled for existing files.")

                return np.nan, np.nan
        
        #Get the embeddings
        emb_pred, emb_full, block_full = get_embeddings(method, dataset, split, params, return_labels = False, lam = lam)

        matrix1 = squareform(pdist(emb_pred))
        matrix2 = squareform(pdist(emb_full))
        
        # Extract the upper triangle of the distance matrices (excluding the diagonal)
        mask = np.triu_indices_from(matrix1, k=1)
        dist1, dist2 = matrix1[mask], matrix2[mask]
        
        # Compute the observed Pearson correlation
        r_obs, _ = pearsonr(dist1, dist2)
        
        # Permutation test to find the null distribution of the correlation
        perm_r = []
        for _ in range(permutations):
            permuted = np.random.permutation(dist2)
            perm_r.append(pearsonr(dist1, permuted)[0])
        
        # Compute the p-value
        perm_r = np.array(perm_r)
        p_value = np.sum(perm_r >= r_obs) / permutations
        
        if plot == True:
            # Plot the smoothed distribution curve of the null distribution of correlations
            plt.figure(figsize=(10, 6))
            sns.kdeplot(perm_r, color='blue', lw=2)  # KDE line
            
            # Add a vertical line to represent the observed correlation
            plt.axvline(x=r_obs, color='red', linestyle='--', lw=2)
            
            # Title and labels
            plt.title('Null Distribution of Correlations Between Embeddings with Observed Correlation', fontsize=14)
            plt.xlabel('Correlation', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            
            # Show plot
            plt.show()

        save_mantel_results(method, dataset, split, r_obs, p_value, perm_r, lam = lam)
        
        return r_obs, p_value #Results are saved above
    
    except:
        return np.nan, np.nan

def save_GRAE_Build_results(method, dataset, split, mse, scores lam = 100, grae_build = "original"):   

    results_dir = "/yunity/arusty/Graph-Manifold-Alignment/Results/Grae_Builds"

    file_name = f"{method}_{dataset}_{str(split)}_graeBuild:{grae_build}_lam_{lam}.json"
    file_path = os.path.join(results_dir, file_name)

    five_point_summary = {
        "min": float(np.min(perm_r)),
        "Q1": float(np.percentile(perm_r, 25)),
        "median": float(np.percentile(perm_r, 50)),
        "Q3": float(np.percentile(perm_r, 75)),
        "max": float(np.max(perm_r))
    }

    results_data = {
        "method": method,
        "dataset": dataset,
        "split": split,
        "r_obs": r_obs,
        "lam": lam,
        "p_value": p_value,
        "five_point_summary": five_point_summary
        }
    
    try:
        with open(file_path, "w") as out_file:
            json.dump(results_data, out_file, indent=4)
    except Exception as e:
        print(f"Error saving Mantel results: {e}")

    print(f"Mantel results saved to: {file_path}")

def read_all_mantel_results():
    results_dir = "/yunity/arusty/Graph-Manifold-Alignment/Results/Mantel"
    all_data = []

    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            file_path = os.path.join(results_dir, file)
            with open(file_path, "r") as f:
                data = json.load(f)
            all_data.append(data)

    return pd.DataFrame(all_data)

def read_all_mantel_results_lam():
    results_dir = "/yunity/arusty/Graph-Manifold-Alignment/Results/Mantel_lam"
    all_data = []

    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            file_path = os.path.join(results_dir, file)
            with open(file_path, "r") as f:
                data = json.load(f)
            all_data.append(data)

    lam_df = pd.DataFrame(all_data)
    normal_df = read_all_mantel_results()

    #Add lambda column to df 
    normal_df["lam"] = 100

    return pd.concat([normal_df, lam_df])

def plot_averaged_mantel_stats(df):
    # Average r_obs and plot distribution with vertical line
    avg_r = df["r_obs"].mean()

    # Compute average of five point summaries and create single boxplot
    vals = df["five_point_summary"].apply(lambda x: [x["min"], x["Q1"], x["median"], x["Q3"], x["max"]])
    plt.figure()
    #Set figure to strecth form -0.3 to 1
    plt.xlim(-0.3, 1)
    plt.boxplot([[vals.apply(lambda v: v[0]).mean(),
                   vals.apply(lambda v: v[1]).mean(),
                   vals.apply(lambda v: v[2]).mean(),
                   vals.apply(lambda v: v[3]).mean(),
                   vals.apply(lambda v: v[4]).mean()]],
                   vert=False)
    plt.title("Averaged Five-Point Summary Boxplot")
    plt.axvline(avg_r, color='red', linestyle='--', label = "Average r_obs")
    plt.legend(), plt.show()

def file_already_exists(method, dataset, split, lam = 100, grae_build = "original"):
    """
    Checks if a results file already exists for the given method, dataset, and split.
    Returns True if it is found, else False.
    """

    file_name = f"{method}_{dataset}_{str(split)}_graeBuild:{grae_build}_lam_{lam}.json"
    return os.path.isfile(os.path.join("/yunity/arusty/Graph-Manifold-Alignment/Results/Grae_Builds", file_name))
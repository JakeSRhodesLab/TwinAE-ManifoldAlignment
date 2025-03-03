#Imports 
from Helpers.regression_helpers import read_json_files_to_dataframe
import os
import numpy as np
import pandas as pd
from Helpers.Pipeline_Helpers import method_dict, create_unique_pairs, get_RF_score, get_embedding_scores
from sklearn.manifold import MDS
from Helpers.Grae import GRAEBase, anchorGRAE, BaseDataset
import json
import os
from sklearn.metrics import mean_squared_error

class split_data():
    """Made to spoof the TMA class but is lightweight"""
    def __init__(self, csv_file, split, anchor_percent):


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
        self.anchors = create_unique_pairs(self.split_A.shape[0], int(self.split_A.shape[0] * anchor_percent))

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

        for grae_build in ["anchor_loss050", "anchor_loss100", "anchor_loss150", "original"]:
            for seed in [42, 4921, 1906]:
                for anchor_percent in [0.1, 0.5, 1]:
                    #Get the parameters, method, and dataset
                    params = row["Best_Params"]
                    method = row["method"]
                    dataset = row["csv_file"]
                    split = row["split"]

                    #Create the task
                    task = (method, dataset, split, params, anchor_percent, grae_build, seed)

                    #Append the task to the tasks list
                    if dataset not in ["S-c", "b", "blobs", "blob", "S-curve"]:
                        tasks.append(task)  

    return tasks

def create_tasks_MSE(df):
    #Create the task list
    tasks = []

    #Iterate through the dataframe
    for index, row in df.iterrows():

        for grae_build in ["just_MSE"]:
            for seed in [42, 4921, 1906]:
                for anchor_percent in [0.1, 0.5, 1]:
                    #Get the parameters, method, and dataset
                    params = row["Best_Params"]
                    method = row["method"]
                    dataset = row["csv_file"]
                    split = row["split"]

                    #Create the task
                    task = (method, dataset, split, params, anchor_percent, grae_build, seed)

                    #Append the task to the tasks list
                    if dataset not in ["S-c", "b", "blobs", "blob", "S-curve"]:
                        tasks.append(task)  

    return tasks

def create_tasks_for_DTA_MAGAN_MASH(df):
    #Create the task list
    tasks = []

    #Iterate through the dataframe
    for index, row in df.iterrows():

        for seed in [42, 4921, 1906]:
            for anchor_percent in [0.1, 0.5, 1]:
                #Get the parameters, method, and dataset
                params = row["Best_Params"]
                method = row["method"]
                dataset = row["csv_file"]
                split = row["split"]

                #Create the task
                task = (method, dataset, split, params, anchor_percent, "alternate", seed)

                #Append the task to the tasks list
                if dataset not in ["S-c", "b", "blobs", "blob", "S-curve"] and method in ["DTA", "MAGAN", "MASH", "RF-MASH"]:
                    tasks.append(task)  

    return tasks

# To keep code cleaner
def create_and_fit_method(method_data, data, params):
    if method_data["Name"] == "MASH" or method_data["Name"] == "RF-MASH":
        method_class = method_data["Model"](knn = params["knn"], page_rank = params["page_rank"], DTM = params["DTM"], density_normalization = params["density_normalization"])
        method_class = method_data["Fit"](method_class, data, data.anchors[len(data.anchors)//2:])
        method_class.optimize_by_creating_connections(threshold = params["threshold"], connection_limit = params["connection_limit"], epochs = params["epochs"],
                                                      hold_out_anchors = data.anchors[:len(data.anchors)//2])

    elif method_data["Name"] == "MAGAN":
        method_class = method_data["Model"](**params)
        method_class, magan = method_data["Fit"](method_class, data, data.anchors, return_MAGAN = True)    
        method_data["magan"] = magan

    else:
        method_class = method_data["Model"](**params)
        method_class = method_data["Fit"](method_class, data, data.anchors)
    
    return method_class

# Create function to create the embeddings (One with excluded test points) from Mash or SPUD
def get_embeddings(method, dataset, split, params, anchor_percent, grae_build = "original", seed = 42, lam = 100):
    """
    Returns embeddings for the full and partial datasets using the specified method.
    Also returns the heatmap.
    """

    #Create a TMA spoof class
    data = split_data(dataset + ".csv", split, anchor_percent=anchor_percent)

    # Ensure both domains share the same shuffled indices
    indices = np.arange(len(data.split_A))
    np.random.seed(seed)
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
    data.anchors = create_unique_pairs(len(X_A_train), int(len(X_A_train) * anchor_percent)) #NOTE: we choose to keep the same amount of anchors. 


    #Create a custom MDS where we keep only 1 job (Not to have nested parrelization)
    n_comps = max(min(data.split_A.shape[1], data.split_B.shape[1]), 2) #Ensures the min is 2 or the lowest data split dimensions
    mds = MDS(metric=True, dissimilarity = 'precomputed', n_init = 4,
                n_jobs=1, random_state = seed, n_components = n_comps) 

    #Get the method data, fit it and prepare it to extract the block
    method_data = method_dict[method]
    method_class = create_and_fit_method(method_data, data, params)

  
    #Get the true embedding
    emb_full = mds.fit_transform(method_data["Block"](method_class))

    """GET GRAE's EMBEDDING below"""
    # Reformat data using x train
    data.split_A = X_A_train
    data.split_B = X_B_train
    data.labels = y_A_train # y_A_train will equal y_B_train

    method_class = create_and_fit_method(method_data, data, params)


    #Get the partial embedding
    emb_partial = mds.fit_transform(method_data["Block"](method_class))
    #print("Partial Embedding Complete")
    
    if grae_build == "alternate":
        
        get_alt_pred_embedding(method_class, dataset, split, method_data, seed, X_A_test,  X_B_test, anchor_percent)
        return None, emb_full, (y_A_train, y_A_test, y_B_train, y_B_test)


    #GRAE on domain A
    split_A = BaseDataset(x = X_A_train, y = y_A_train, split_ratio = 0.8, random_state = 42, split = "none")

    if grae_build != "original":
        if grae_build[-3:] == "050":
            myGrae = anchorGRAE(lam = lam, n_components = n_comps, anchor_lam=50)
        elif grae_build[-3:] == "100":
            myGrae = anchorGRAE(lam = lam, n_components = n_comps, anchor_lam=100)
        else:
            myGrae = anchorGRAE(lam = lam, n_components = n_comps, anchor_lam=150)

        myGrae.fit(split_A, emb = emb_partial[:len(X_A_train)], anchors = data.anchors)

    else:
        myGrae = GRAEBase(lam = lam, n_components = n_comps)
        myGrae.fit(split_A, emb = emb_partial[:len(X_A_train)])

    testA = BaseDataset(x = X_A_test, y = y_A_test, split_ratio = 0.8, random_state = 42, split = "none")
    pred_A, _ = myGrae.score(testA)

    #Grae on domain B 
    split_B = BaseDataset(x = X_B_train, y = y_B_train, split_ratio = 0.8, random_state = 42, split = "none")

    if grae_build != "original":
        if grae_build[-3:] == "050":
            myGraeB = anchorGRAE(lam = lam, n_components = n_comps, anchor_lam=50)
        elif grae_build[-3:] == "100":
            myGraeB = anchorGRAE(lam = lam, n_components = n_comps, anchor_lam=100)
        else:
            myGraeB = anchorGRAE(lam = lam, n_components = n_comps, anchor_lam=150)

        myGraeB.fit(split_B, emb = emb_partial[int(len(emb_partial)/2):], anchors = data.anchors)

    else:
        myGraeB = GRAEBase(lam = lam, n_components = n_comps)
        myGraeB.fit(split_B, emb = emb_partial[int(len(emb_partial)/2):])

    testB = BaseDataset(x = X_B_test, y = y_B_test, split_ratio = 0.8, random_state = 42, split = "none")
    pred_B, _ = myGraeB.score(testB)

    if grae_build == "just_MSE":
        #Transform test points to other domain
        A_to_z = myGrae.transform(testA)
        B_to_z = myGraeB.transform(testB)

        A_to_B = myGraeB.inverse_transform(A_to_z)
        B_to_A = myGrae.inverse_transform(B_to_z)

        #Calculate mse
        mse = (mean_squared_error(A_to_B, X_B_test) + mean_squared_error(B_to_A, X_A_test))/2

        save_GRAE_Build_results(method_data["Name"], dataset, split, mse, [None]*9, [None]*9, grae_build="just_MSE", seed = seed, anchor_percent=anchor_percent)
        return None, emb_full, (y_A_train, y_A_test, y_B_train, y_B_test)


    #Grab the scores
    A_train = emb_partial[:int(len(emb_partial)/2)]
    B_train = emb_partial[int(len(emb_partial)/2):]
    emb_pred = np.vstack([A_train, pred_A, B_train, pred_B]) #NOTE: Train on just train
 
    return emb_pred, emb_full, (y_A_train, y_A_test, y_B_train, y_B_test)

def get_alt_pred_embedding(method_class, dataset, split, method_data, seed, X_A_test,  X_B_test, anchor_percent):

    if method_data["Name"] == "MAGAN":
        # Translate test points
        A_to_B = method_data["magan"].translate_1_to_2(X_A_test)
        B_to_A = method_data["magan"].translate_2_to_1(X_B_test)

        #Calculate mse
        mse = (mean_squared_error(A_to_B, X_B_test) + mean_squared_error(B_to_A, X_A_test))/2
        #Save results
        save_GRAE_Build_results("MAGAN", dataset, split, mse, [None]*9, [None]*9, grae_build="alternate", seed = seed, anchor_percent=anchor_percent)

    elif method_data["Name"] == "DTA":
        #Rescale to work for test data
        # Perform PCA on projectionBA
        from sklearn.decomposition import PCA

        pca = PCA(n_components=X_A_test.shape[1])
        projection = pca.fit_transform(method_class.T.T).T

        pca = PCA(n_components=X_B_test.shape[1])
        projection = pca.fit_transform(projection)

        projection = 2 * (projection - projection.min()) / (projection.max() - projection.min()) - 1

        #Translate test points
        B_to_A =  (projection @ X_B_test.T).T
        A_to_B = X_A_test @ projection

        mse = (mean_squared_error(A_to_B, X_B_test) + mean_squared_error(B_to_A, X_A_test))/2

        #Save results
        save_GRAE_Build_results("DTA", dataset, split, mse, [None]*9, [None]*9, grae_build="alternate", seed = seed, anchor_percent=anchor_percent)

    else:
        #Translate test points
        projection = method_class.get_linear_transformation()
        B_to_A =  (projection @ X_B_test.T).T
        A_to_B = X_A_test @ projection

        mse = (mean_squared_error(A_to_B, X_B_test) + mean_squared_error(B_to_A, X_A_test))/2

        #Save results
        save_GRAE_Build_results(method_data["Name"], dataset, split, mse, [None]*9, [None]*9, grae_build="alternate", seed = seed, anchor_percent=anchor_percent)
        
def GRAE_tests(method, dataset, split, params, anchor_percent, grae_build = "original", seed = 42): #DON'T Delete any of these parameters - though you can add your own if you want

    """
    Perform a Mantel test to compute the correlation between two embeddings
    
    Returns:
        r (float): Observed correlation.
        p_value (float): Significance of the observed correlation.
    """
    try:

        #Return null values if file already exsists
        if file_already_exists(method, dataset, split, grae_build, seed, anchor_percent = anchor_percent):
            print(f"Results already exist for {method}, {dataset}, {split}.")

            return False #indicating already ran
        
        #Get the embeddings
        emb_pred, emb_full, labels = get_embeddings(method, dataset, split, params, anchor_percent =  anchor_percent,  grae_build = grae_build, seed = seed)

        if grae_build == "alternate":
            #Magan results return early
            return True
        
        if grae_build == "just_MSE":
            #Magan results return early
            return True

        # Calculate MSE between embeddings
        train_len = len(labels[0])
        test_len = train_len + len(labels[1])
        mse_emb_pred = np.vstack([emb_pred[train_len:test_len], emb_pred[test_len + train_len:]])
        mse_emb_full = np.vstack([emb_full[train_len:test_len], emb_full[test_len + train_len:]])
        mse = mean_squared_error(mse_emb_pred, mse_emb_full)

        #Calculate scores
        rf_oob_true = get_RF_score(emb_full, labels, seed)
        rf_oob_pred = get_RF_score(emb_pred, labels, seed)

        knn_scoreA, rf_scoreA, knn_metricA, rf_metricA, knn_scoreB, rf_scoreB, knn_metricB, rf_metricB = get_embedding_scores(emb_full, labels, seed)
        emb_full_scores = (rf_oob_true, knn_scoreA, rf_scoreA, knn_metricA, rf_metricA, knn_scoreB, rf_scoreB, knn_metricB, rf_metricB)

        knn_scoreA, rf_scoreA, knn_metricA, rf_metricA, knn_scoreB, rf_scoreB, knn_metricB, rf_metricB = get_embedding_scores(emb_pred, labels, seed)
        emb_pred_scores = (rf_oob_pred, knn_scoreA, rf_scoreA, knn_metricA, rf_metricA, knn_scoreB, rf_scoreB, knn_metricB, rf_metricB)

        #Check to see what functions we can hookly doo upto

        save_GRAE_Build_results(method, dataset, split, mse, emb_full_scores, emb_pred_scores, grae_build=grae_build, seed = seed, anchor_percent=anchor_percent)
        
        return True #Indicating sucessful run    
    except Exception as e:
        print("Hit error in GRAE_tests: ", e)
        return False #Indicating failed run

def save_GRAE_Build_results(method, dataset, split, mse, emb_full_scores, emb_pred_scores, lam = 100, grae_build = "original", anchor_percent = 0.3, seed = 42):   

    results_dir = "/yunity/arusty/Graph-Manifold-Alignment/Results/Grae_Builds"

    file_name = f"{method}_{dataset}_{str(split)}_graeBuild_{grae_build}_lam_{lam}_seed{str(seed)}_an{str(anchor_percent)}.json"
    file_path = os.path.join(results_dir, file_name)
        
    full_rf_oob, full_knn_scoreA, full_rf_scoreA, full_knn_metricA, full_rf_metricA, full_knn_scoreB, full_rf_scoreB, full_knn_metricB, full_rf_metricB = emb_full_scores
    pred_rf_oob, pred_knn_scoreA, pred_rf_scoreA, pred_knn_metricA, pred_rf_metricA, pred_knn_scoreB, pred_rf_scoreB, pred_knn_metricB, pred_rf_metricB = emb_pred_scores

    results_data = {
        "method": method,
        "dataset": dataset,
        "split": split,
        "lam": lam,
        "Anchor_Percent": anchor_percent,
        "grae_build": grae_build,
        "MSE": mse,
        "full_rf_oob": full_rf_oob,
        "full_knn_scoreA": full_knn_scoreA,
        "full_rf_scoreA": full_rf_scoreA,
        "full_knn_metricA": full_knn_metricA,
        "full_rf_metricA": full_rf_metricA,
        "full_knn_scoreB": full_knn_scoreB,
        "full_rf_scoreB": full_rf_scoreB,
        "full_knn_metricB": full_knn_metricB,
        "full_rf_metricB": full_rf_metricB,
        "pred_rf_oob": pred_rf_oob,
        "pred_knn_scoreA": pred_knn_scoreA,
        "pred_rf_scoreA": pred_rf_scoreA,
        "pred_knn_metricA": pred_knn_metricA,
        "pred_rf_metricA": pred_rf_metricA,
        "pred_knn_scoreB": pred_knn_scoreB,
        "pred_rf_scoreB": pred_rf_scoreB,
        "pred_knn_metricB": pred_knn_metricB,
        "pred_rf_metricB": pred_rf_metricB,
    }
    
    try:
        with open(file_path, "w") as out_file:
            json.dump(results_data, out_file, indent=4)
    except Exception as e:
        print(f"Error saving Mantel results: {e}")

    print(f"Mantel results saved to: {file_path}")

def read_all_graeBuild_results():
    results_dir = "/yunity/arusty/Graph-Manifold-Alignment/Results/Grae_Builds"

    all_data = []

    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            file_path = os.path.join(results_dir, file)
            with open(file_path, "r") as f:
                data = json.load(f)

            # Only rename if the key exists in the dictionary
            if "Anchor_percent" in data:
                data["Anchor_Percent"] = data.pop("Anchor_percent")

            all_data.append(data)

    return pd.DataFrame(all_data)

def file_already_exists(method, dataset, split, grae_build = "original", seed = 42, anchor_percent = 0.3, lam = 100):
    """ (method, dataset, split, grae_build, seed)
    Checks if a results file already exists for the given method, dataset, and split.
    Returns True if it is found, else False.
    """

    results_dir = "/yunity/arusty/Graph-Manifold-Alignment/Results/Grae_Builds"

    file_name = f"{method}_{dataset}_{str(split)}_graeBuild_{grae_build}_lam_{lam}_seed{str(seed)}_an{str(anchor_percent)}.json"
    file_path = os.path.join(results_dir, file_name)

    return os.path.isfile(file_path)
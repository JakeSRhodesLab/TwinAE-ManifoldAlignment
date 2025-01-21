"""
This File is to run basic baseline tests. 


To remain accurate against tests and how we score the models we are going to implement the same testing methodology as given in the Pipline file. 

Methodology: 
1. Create Train test splits with 20 % test size with the following random seeds: 1738, 5271, 9209, 1316, 42. 
2. Use get_RF_score from Pipeline_Helpers.py to get the rf scores. 
3. Use get_embedding_scores as same as above. 
4. Save it as a CSV. 
"""

#Imports
from Helpers.Pipeline_Helpers import get_RF_score, get_embedding_scores
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Helpers.utils import dataprep
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from Helpers.Mantels_Helpers import split_features

print("Imports completed.\n-------------------------------------------------\n\n")


#Determine to run classification or regression
run_regression = None

while run_regression is None:
    user_input = input("Type 'True' to run regression or 'False' to do classification: ")
    run_regression = True if user_input == "True" else False if user_input == "False" else None

#Determine run all metrics or KNN train test
test_to_run = None

while test_to_run is None:
    user_input = input("Type 'True' to train test knn or 'False' to do pipeline: ")
    test_to_run = True if user_input == "True" else False if user_input == "False" else None

print("Running...\n")


# Data sets below.
if not run_regression:  
    csv_files = [
        "zoo.csv", "hepatitis.csv", "iris.csv", "audiology.csv", "parkinsons.csv", "seeds.csv", "segmentation.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "flare1.csv", "ecoli_5.csv", "ionosphere.csv",
        "Cancer_Data.csv", "hill_valley.csv", "balance_scale.csv", 'winequality-red.csv', 'car.csv', "crx.csv", "breast_cancer.csv", "titanic.csv", "diabetes.csv", "tic-tac-toe.csv",
        'Medicaldataset.csv', "water_potability.csv", 'treeData.csv', "optdigits.csv", "waveform.csv", "chess.csv", "artificial_tree.csv"
        ]
else: 
    csv_files = [
        "EnergyEfficiency.csv", "Hydrodynamics.csv", "OpticalNetwork.csv","AirfoilSelfNoise.csv","AutoMPG.csv","ComputerHardware.csv","CommunityCrime.csv",
        "ConcreteSlumpTest.csv", "FacebookMetrics.csv", "Parkinsons.csv", "IstanbulStock.csv", "Automobile.csv", "ConcreteCompressiveStrength.csv", "SML2010.csv"
        ]

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def cross_embedding_knn(embedding, Y, knn_args = {'n_neighbors': 4}):
        
        n1 = int(len(Y)/2)

        # Determine if the task is classification or regression
        if not run_regression:
            knn = KNeighborsClassifier(**knn_args)
        else:
            knn = KNeighborsRegressor(**knn_args)
            knn.fit(embedding[:n1], Y[:n1])

            return knn.score(embedding[n1:], Y[n1:])

# Data Prep Function
def prep_data_file(csv_file, seed, split):
    """
    Takes the csv_file (and seed) to a csv file and returns embedding and labels for the get_embedding_scores and
    get_RF_scores functions. 
    """

    #Find whether its a classification or regression problem
    if not run_regression:
        df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Classification_CSV/" + csv_file)
    else:
        df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Regression_CSV/" + csv_file)

    #We need to gt the labels
    _, y = dataprep(df, label_col_idx=0)
    y = np.array(y)

    #get data files
    split_A, split_B = split_features(csv_file, split, seed)

    """For Domain A"""
    # Split into two groups A and B. This should have no effect as its reconstructed later
    X_A, X_B, y_A, y_B = train_test_split(split_A, y, test_size=0.5, random_state=42)

    # Further split each group into train and test sets
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(X_A, y_A, test_size=0.2, random_state=seed)
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(X_B, y_B, test_size=0.2, random_state=seed)

    # Combine embeddings into a single array and labels as a tuple
    embA = np.vstack((X_A_train, X_A_test, X_B_train, X_B_test))
    labelsA = (y_A_train, y_A_test, y_B_train, y_B_test)

    """For Domain B"""
    # Split into two groups A and B. This should have no effect as its reconstructed later
    X_A, X_B, y_A, y_B = train_test_split(split_B, y, test_size=0.5, random_state=42)

    # Further split each group into train and test sets
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(X_A, y_A, test_size=0.2, random_state=seed)
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(X_B, y_B, test_size=0.2, random_state=seed)

    # Combine embeddings into a single array and labels as a tuple
    embB = np.vstack((X_A_train, X_A_test, X_B_train, X_B_test))
    labelsB = (y_A_train, y_A_test, y_B_train, y_B_test)

    #Return the necessary arguments
    return embA, labelsA, embB, labelsB

# Evaluation function - Similar to pipeline
def get_results(csv_file, seed, split):
    embA, labelsA, embB, labelsB = prep_data_file(csv_file, seed, split)

    """Domain A"""
    knn_score, rf_score, knn_rmse, rf_rmse = get_embedding_scores(embA, labelsA, seed)
    rf_oob = get_RF_score(embA, labelsA, seed)
    knn = cross_embedding_knn(embA, np.hstack(labelsA))

    #Return it as a dictionary so we can make a Pandas table easier later
    domain_A_results =  {"csv_file" : csv_file, "Method": "Domain A Pipeline Baseline", "split": split,
            "Random Forest OOB": rf_oob,
            "Random Forest Emb": rf_score,
            "Nearest Neighbor": knn_score,
            "Nearest Neighbor (F1 score or RMSE)": knn_rmse,
            "Random Forest (F1 score or RMSE)": rf_rmse,
            "CE (4 KNN)": knn
            }
    
    """Domain B"""
    knn_score, rf_score, knn_rmse, rf_rmse = get_embedding_scores(embB, labelsB, seed)
    rf_oob = get_RF_score(embB, labelsB, seed)
    knn = cross_embedding_knn(embB, np.hstack(labelsB))


    #Return it as a dictionary so we can make a Pandas table easier later
    domain_B_results =  {"csv_file" : csv_file, "Method": "Domain B Pipeline Baseline", "split": split,
            "Random Forest OOB": rf_oob,
            "Random Forest Emb": rf_score,
            "Nearest Neighbor": knn_score,
            "Nearest Neighbor (F1 score or RMSE)": knn_rmse,
            "Random Forest (F1 score or RMSE)": rf_rmse,
            "CE (4 KNN)": knn
            }
    
    return domain_A_results, domain_B_results

def get_train_test_results(csv_file, seed, split):
    """
    Evaluate the KNN classifier on train-test split of the data file.

    Args:
        csv_file (str): The name of the CSV file containing the dataset.
        seed (int): The random seed for reproducibility.
        split (str): The type of split to be used.

    Returns:
        dict: A dictionary containing the classification score for Domain A.
        dict: A dictionary containing the classification score for Domain B.
    """
    try:

        # Evaluation function - train test split on the data file
        embA, labelsA, embB, labelsB = prep_data_file(csv_file, seed, split)

        labelsA = np.hstack(labelsA)
        labelsB = np.hstack(labelsB)
        
        #Use KNN to get the score of four
        if run_regression:
            knn = KNeighborsRegressor(n_neighbors=4)
        else:
            knn = KNeighborsClassifier(n_neighbors=4)

        #Domain A
        X_train, X_test, y_train, y_test = train_test_split(embA, labelsA, test_size=0.2, random_state=seed)
        knn.fit(X_train, y_train)
        knn_scoreA = knn.score(X_test, y_test)

        #Use KNN to get the score of four
        if run_regression:
            knn = KNeighborsRegressor(n_neighbors=4)
        else:
            knn = KNeighborsClassifier(n_neighbors=4)

        #Domain B
        X_train, X_test, y_train, y_test = train_test_split(embB, labelsB, test_size=0.2, random_state=seed)
        knn.fit(X_train, y_train)
        knn_scoreB = knn.score(X_test, y_test)

        #Return it as a dictionary so we can make a Pandas table easier later
        return {"csv_file" : csv_file, "Method": "Train Test Baseline", "split": split, "A_Classification_Score": knn_scoreA, "B_Classification_Score": knn_scoreB}

    except Exception as e:
        return {"csv_file" : csv_file, "Method": "Train Test Baseline", "split": split, "A_Classification_Score": np.NaN, "B_Classification_Score": np.NaN}




csv_seed_list = []
for seed in [1738, 5271, 9209, 1316, 42]: 
    for csv in csv_files: #CHANGE THIS FOR REGRESSION OR NOT
        for split in ["random", "even", "turn", "skewed", "distort"]:
            csv_seed_list.append((csv, seed, split))

if test_to_run:
    method = get_train_test_results
    file_string = "TrainTestBaselines"
else:
    method = get_results
    file_string = "PipelineBaselines"


# Get the results and show progress
with tqdm_joblib(tqdm(desc="Processing tasks", total=len(csv_seed_list))) as progress_bar:
    results = Parallel(n_jobs=10)(delayed(method)(csv, seed, split) for csv, seed, split in csv_seed_list)

#Unpack and make into dataframe
if test_to_run:
    flattened_results = results
else:
    flattened_results = [item for sublist in results for item in sublist]


new_df = pd.DataFrame(flattened_results)

#Write Pandas dataframe
if not run_regression:
    new_df.to_csv("/yunity/arusty/Graph-Manifold-Alignment/Results/ManifoldData/" + file_string + ".csv")
else:
    new_df.to_csv("/yunity/arusty/Graph-Manifold-Alignment/Results/RegressionData/" + file_string + ".csv")

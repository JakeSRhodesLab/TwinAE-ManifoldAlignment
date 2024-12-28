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
print("Imports completed.\n-------------------------------------------------\n\n")


#Determine to run classification or regression
run_regression = None

while run_regression is None:
    user_input = input("Type 'True' to run regression or 'False' to do classification: ")
    run_regression = True if user_input == "True" else False if user_input == "False" else None

print("Running...\n")


# Data sets below.
if not run_regression:  
    csv_files = [
        "zoo.csv", "hepatitis.csv", "iris.csv", "audiology.csv", "parkinsons.csv", "seeds.csv", "segmentation.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "flare1.csv", "ecoli_5.csv", "ionosphere.csv",
        "Cancer_Data.csv", "hill_valley.csv", "balance_scale.csv", "S-curve", "blobs", 'winequality-red.csv', 'car.csv', "crx.csv", "breast_cancer.csv", "titanic.csv", "diabetes.csv", "tic-tac-toe.csv",
        'Medicaldataset.csv', "water_potability.csv", 'treeData.csv', "optdigits.csv", "waveform.csv", "chess.csv", "artificial_tree.csv"
        ]
else: 
    csv_files = [
        "EnergyEfficiency.csv", "Hydrodynamics.csv", "OpticalNetwork.csv","AirfoilSelfNoise.csv","AutoMPG.csv","ComputerHardware.csv","CommunityCrime.csv",
        "ConcreteSlumpTest.csv", "FacebookMetrics.csv", "Parkinsons.csv", "IstanbulStock.csv", "Automobile.csv", "ConcreteCompressiveStrength.csv", "SML2010.csv"
        ]


# Data Prep Function
def prep_data_file(csv_file, seed):
    """
    Takes the csv_file (and seed) to a csv file and returns embedding and labels for the get_embedding_scores and
    get_RF_scores functions. 
    """

    #Find whether its a classification or regression problem
    if not run_regression:
        df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Classification_CSV/" + csv_file)
    else:
        df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Regression_CSV/" + csv_file)

    #Split data into features and labels. This also normalizes.
    X, y = dataprep(df, label_col_idx=0)

    # Split into two groups A and B. This should have no effect as its reconstructed later
    X_A, X_B, y_A, y_B = train_test_split(X, y, test_size=0.5, random_state=42)

    # Further split each group into train and test sets
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(X_A, y_A, test_size=0.2, random_state=seed)
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(X_B, y_B, test_size=0.2, random_state=seed)

    # Combine embeddings into a single array and labels as a tuple
    emb = np.vstack((X_A_train, X_A_test, X_B_train, X_B_test))
    labels = (y_A_train, y_A_test, y_B_train, y_B_test)

    #Return the necessary arguments
    return emb, labels, seed

# Evaluation function
def get_results(csv_file, seed):
    knn_score, rf_score, knn_rmse, rf_rmse = get_embedding_scores(*prep_data_file(csv_file, seed))
    rf_oob = get_RF_score(*prep_data_file(csv_file, seed))

    #Return it as a dictionary so we can make a Pandas table easier later
    return {"csv_file" : csv_file, "Method": "Pipeline Baseline", 
            "Random Forest OOB": rf_oob,
            "Random Forest Emb": rf_score,
            "Nearest Neighbor": knn_score,
            "Nearest Neighbor (F1 score or RMSE)": knn_rmse,
            "Random Forest (F1 score or RMSE)": rf_rmse
            }


csv_seed_list = []
for seed in [1738, 5271, 9209, 1316, 42]: 
    for csv in csv_files: #CHANGE THIS FOR REGRESSION OR NOT
        csv_seed_list.append((csv, seed))

# Get the results and show progress
with tqdm_joblib(tqdm(desc="Processing tasks", total=len(csv_seed_list))) as progress_bar:
    results = Parallel(n_jobs=10)(delayed(get_results)(csv, seed) for csv, seed in csv_seed_list)

#Write Pandas dataframe
if run_regression:
    pd.DataFrame(results).to_csv("/yunity/arusty/Graph-Manifold-Alignment/Results/ManifoldData/PipelineBasline.csv")
else:
    pd.DataFrame(results).to_csv("/yunity/arusty/Graph-Manifold-Alignment/Results/RegressionData/PipelineBaseline.csv")

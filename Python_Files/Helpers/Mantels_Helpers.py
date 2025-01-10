#Imports 
from regression_helpers import read_json_files_to_dataframe


# Create function to Extract best fit information from the results
def extract_all_fits():
    #Get the regression results and classification results
    df = read_json_files_to_dataframe("/yunity/arusty/Graph-Manifold-Alignment/Results")

    #Filter the dataframe to only include the valid methods
    filtered_df = df[df["method"].isin(["SPUD", "RF-SPUD", "NAMA", "RF-NAMA", "MASH", "MASH-", "RF-MASH", "RF-MASH-"])]

    return filtered_df

# Create a function to get the parameters for a parrelization loop
def create_params_for_parrelization(df):
    #Create the task list
    tasks = []

    #Iterate through the dataframe
    for index, row in df.iterrows():
        #Get the parameters
        params = row["Best_Params"]
        #Get the method
        method = row["method"]
        #Get the dataset
        dataset = row["dataset"]
        #Get the task
        task = (params, method, dataset)
        #Append the task to the tasks list
        tasks.append(task)  


# Create function to create the embeddings (One with excluded test points) from Mash or SPUD

# Create the GRAE model evaluation
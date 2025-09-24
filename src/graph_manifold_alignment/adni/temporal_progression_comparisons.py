"""
This is a function for calculating the difference between time series data using cross correlation
This code is meant to handle a dataframe with a multilevel index where level one is RID and level
two is the visit month
There should be no nans in the input data
"""

import numpy as np
import pandas as pd
from tslearn.metrics import dtw_path
from dtaidistance import dtw

# Create a squareform array of the comparisons

def normalized_euclidean_distance(sequence_1, sequence_2):
    length = len(sequence_1)
    dimensions = sequence_1.shape[1]
    #calculate the distance between the sequences with the eudlidean distance formula
    distance = np.sqrt(np.sum((sequence_1 - sequence_2) ** 2))
    #normalize for the lenth of the sequences and the number of dimesions being compared
    normalized_distance = distance / np.sqrt(length * dimensions)
    return normalized_distance


def normalized_dynamic_time_warping(sequence_1, sequence_2):
    # Compute DTW path and distance
    path, distance = dtw_path(sequence_1, sequence_2)
    # Normalize by path length
    normalized_distance = distance / len(path)
    return normalized_distance

def dynamic_time_warping(sequence_1, sequence_2):
    distance = dtw.distance(sequence_1.flatten(), sequence_2.flatten())
    return distance
    dtw_distance = dtw.distance(sequence_1.flatten(), sequence_2.flatten())
    return dtw_distance

def sequences_distance(df, id_1, id_2, method):
    """feed in the two temporal sequences that you want to compare and specify the method to 
    compare them with, eucidean for just a generalized euclidean distance, dtw for a dynamic time 
    warping, and normdtw for dtw normalized for time sequence length"""
    #print(id_1) #uncomment to see if it's actually running and how fast it's going
    person_1 = df.loc[[id_1], :].to_numpy()
    person_2 = df.loc[[id_2], :].to_numpy()
    #get the possible lag values with an overlap of at least 3
    possible_lag_values = np.arange(-(len(person_2) - 3), (len(person_1) - 2))
    #get the set of positions for each lag and use the intersection between those sets to know how to truncate them
    set_1 = [x for x in range(len(person_1))]
    distance_by_lag = []
    for lag in possible_lag_values:
        set_2 = [x+lag for x in range(len(person_2))]
        intersection = list(set(set_1) & set(set_2)) #the positions of the overlapping values
        person_1_truncated = person_1[intersection]
        person_2_truncated = person_2[intersection - lag]
        if method == "euclidean":
            distance_by_lag.append(normalized_euclidean_distance(person_1_truncated, person_2_truncated))
        elif method == "dtw":
            distance_by_lag.append(dynamic_time_warping(person_1_truncated, person_2_truncated))
        elif method == "normdtw":
            distance_by_lag.append(normalized_dynamic_time_warping(person_1_truncated, person_2_truncated))
        else:
            print("Invalid comparison method option")
    return min(distance_by_lag)


def wrapped_euclidean_distances(df):
    # Normalize the values for each variable so that smaller units don't lend an advantage
    def normalize_column(column):
        normalized_column = (column - np.mean(column)) / np.std(column)
        return normalized_column
    normalized_df = df.apply(normalize_column, axis=0)
    # Make and return the omparisions squareform array
    rids = normalized_df.index.get_level_values("RID").unique()
    comparisons = [[sequences_distance(normalized_df, id_1, id_2, method="euclidean") for id_2 in rids] for id_1 in rids]
    squareform = np.array(comparisons)
    # Normalize to the range of 0 to 1
    min_val = np.min(squareform)
    max_val = np.max(squareform)
    squareform = (squareform - min_val) / (max_val - min_val)
    return squareform

def wrapped_dtw_distances(df):
    # Normalize the values for each variable so that smaller units don't lend an advantage
    def normalize_column(column):
        normalized_column = (column - np.mean(column)) / np.std(column)
        return normalized_column
    normalized_df = df.apply(normalize_column, axis=0)
    # Make and return the omparisions squareform array
    rids = normalized_df.index.get_level_values("RID").unique()
    comparisons = [[sequences_distance(normalized_df, id_1, id_2, method="normdtw") for id_2 in rids] for id_1 in rids]
    squareform = np.array(comparisons)
    # Normalize to the range of 0 to 1
    min_val = np.min(squareform)
    max_val = np.max(squareform)
    squareform = (squareform - min_val) / (max_val - min_val)
    return squareform

if __name__ == "__main__":
    #this code runs if you just wanna run the functions in here and get the squareform
    print("This is running as __main__")
    sample_size = 100
    data = pd.read_excel(r"C:/Users/jcory/Box/ADNI/Datasets/Merged Data Files/Progression Variables 2024-08-02.xlsx", index_col=[0,1])
    first_level_values = data.index.get_level_values(0)
    unique_rids = first_level_values.unique()
    sample_rids = unique_rids[:sample_size]
    sample_data = data.loc[sample_rids]
    squareform = wrapped_euclidean_distances(sample_data)
    print(squareform)
else:
    print("This is not running as __main__")


# comparisons2 = [[sequences_distance(id_1, id_2, method="normdtw") for id_2 in rids[0:10]] for id_1 in rids[0:10]]
# squareform2 = np.array(comparisons2)

#TODO Somehow convert this so that there's a function in there that you can just spit the data into and then it returns
#a squareform for each of the comparison methods, then test it in the plugin tester and make sure that the data being
#fed in is in the same form that it will eventually be in when integrated with SPUD


# person_1 = df.loc[[4], :].to_numpy()
# person_2 = df.loc[[4], :].to_numpy()
# print(normalized_euclidean_distance(person_1, person_2))





"""This is a function for calculating the difference between time series data using cross correlation"""

import numpy as np
import pandas as pd
from tslearn.metrics import dtw_path
from dtaidistance import dtw


df = pd.read_excel("C:/Users/jcory/Box/ADNI/Datasets/Merged Data Files/Visit Variables 2024-07-11.xlsx", index_col=[0,1])
#df.reset_index(inplace=True)

# Get all of the temporal sequences in step

patients = df.index.get_level_values('RID').unique()
max_months = df.index.get_level_values("VISMONTH").max()
months = np.arange(0, (max_months + 1), 6)
# Create a complete index of possible six month visits
multi_index = pd.MultiIndex.from_product([patients, months], names=['RID', 'VISMONTH']) 
# Reindex the DataFrame with all possible six month visits
df = df.reindex(multi_index)

# Interpolate and remove any trailing visits for each person that have no information in them
def fill_and_chop_nans(small_df): 
    something_in_row = small_df.notna().any(axis=1) #returns True or False to say if each index has any info at all
    if sum(something_in_row) >= 4: #must have at least 4 visits worth of information to be compared
        last_valid_index = something_in_row[::-1].idxmax() #reverses the order and gets the index of the first True value
    else:
        last_valid_index = 0
    small_df = small_df.loc[:last_valid_index] #chops off the empty visits
    small_df.interpolate(axis=0, method="linear", inplace=True)
    return small_df

df = df.groupby("RID", group_keys = False).apply(fill_and_chop_nans)

# Normalize the values for each variable so that smaller units don't lend an advantage

def normalize_column(column):
    normalized_column = (column - np.mean(column)) / np.std(column)
    return normalized_column

normalized_df = df.apply(normalize_column, axis=0)

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

def sequences_distance(id_1, id_2, method):
    """feed in the two temporal sequences that you want to compare and specify the method to 
    compare them with, eucidean for just a generalized euclidean distance, dtw for a dynamic time 
    warping, and normdtw for dtw normalized for time sequence length"""
    print(id_1)
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

rids = df.index.get_level_values("RID").unique()
patient_count = df.index.get_level_values("RID").nunique()
comparisons1 = [[sequences_distance(id_1, id_2, method="euclidean") for id_2 in rids[0:10]] for id_1 in rids[0:10]]
squareform1 = np.array(comparisons1)
comparisons2 = [[sequences_distance(id_1, id_2, method="normdtw") for id_2 in rids[0:10]] for id_1 in rids[0:10]]
squareform2 = np.array(comparisons2)

#TODO Make each of the small dataframes into numpy arrays and use to test the normalized eucidean distance formula
#then, create another function between sequences_distance and normalized_eucidean distance that runs NUD for each
#lag and then returns the minimum distance that it was able to find at a particular lag

# person_1 = df.loc[[4], :].to_numpy()
# person_2 = df.loc[[4], :].to_numpy()
# print(normalized_euclidean_distance(person_1, person_2))



#TODO Figure out the dimensionalities here so that it considers the many dimensions of the temporal sequences in order
#TODO Make a part of it that fills in for missing values and check for edge effects, I'm not totally sure that the roll
#thing is what we want

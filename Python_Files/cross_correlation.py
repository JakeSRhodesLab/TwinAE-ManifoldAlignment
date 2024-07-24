"""This is a function for calculating the difference between time series data using cross correlation"""

import numpy as np
import pandas as pd
from scipy.signal import correlate

# Example time series data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
# x = np.array([[0,0,0,2,2,2,0,0,0], [0,0,0,2,2,2,0,0,0]])
# y = np.array([[0,2,2,2,0,0,0,0,0], [0,2,2,2,0,0,0,0,0]])
# x = np.transpose(x)
# y = np.transpose(y)



df = pd.read_excel("C:/Users/jcory/Box/ADNI/Datasets/Merged Data Files/Visit Variables 2024-07-11.xlsx", index_col=[0,1])
person_1 = df.loc[[2], :]
person_2 = df.loc[[3], :]

#TODO Figure out the dimensionalities here so that it considers the many dimensions of the temporal sequences in order
#TODO Make a part of it that fills in for missing values and check for edge effects, I'm not totally sure that the roll
#thing is what we want

# Normalize the data (optional)
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

# Compute cross-correlation
cross_corr = correlate(x, y, mode='full')

# def correlate_1d(x, y):
#     return correlate(x, y, mode='full')

# # Apply correlation along axis 1
# corr_result = np.apply_along_axis(lambda row: correlate_1d(row, y[0]), axis=1, arr=x)

# Determine the lag
lags = np.arange(-(len(y) - 1), len(x))
lag = lags[np.argmax(cross_corr)]

print(f"Best lag: {lag}")

# Shift the time series by the identified lag
if lag > 0:
    y_shifted = np.roll(y, lag)
elif lag < 0:
    x_shifted = np.roll(x, -lag)
else:
    x_shifted, y_shifted = x, y

# Now x_shifted and y_shifted can be compared directly

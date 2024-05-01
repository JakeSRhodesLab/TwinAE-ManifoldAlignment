import scipy
import numpy as np    
import pandas as pd


def row_normalize(x):

    if scipy.sparse.issparse(x):
        row_sums = x.sum(axis = 1)
        n_rows = x.shape[0]
        inverse_row_sums = 1 / np.array(row_sums).flatten()
        diagonal_matrix = scipy.sparse.diags(inverse_row_sums, shape=(n_rows, n_rows), format = "csr")
        normalized_matrix = diagonal_matrix.dot(x)
        normalized_matrix.data[np.isinf(normalized_matrix.data)] = 0
        return normalized_matrix
    else:
        normalized_matrix = x / np.sum(x, axis = 1, keepdims = True)
        normalized_matrix[np.isinf(normalized_matrix)] = 0
        return normalized_matrix


def col_normalize(x):

    if scipy.sparse.issparse(x):
        col_sums = x.sum(axis = 0)
        n_cols = x.shape[1]
        inverse_col_sums = 1 / np.array(col_sums).flatten()
        diagonal_matrix = scipy.sparse.diags(inverse_col_sums, shape=(n_cols, n_cols), format = "csr")
        normalized_matrix = x.dot(diagonal_matrix) # Checkthis
        normalized_matrix.data[np.isinf(normalized_matrix.data)] = 0
        return normalized_matrix
    else:
        normalized_matrix = x / np.sum(x, axis = 0, keepdims = True)
        normalized_matrix[np.isinf(normalized_matrix)] = 0
        return normalized_matrix
        


def label_impute(x, y, model):

    y = y.copy()
    missing_idx = y == -1

    if sum(missing_idx) == 0:
        # model.fit(x, y)
        return y, model

    model.fit(x[~missing_idx, :], y[~missing_idx])
    preds = model.predict(x[missing_idx, :])
    y[missing_idx] = preds


    return y, model

def dataprep(data, label_col_idx = 0, transform = 'normalize', cat_to_numeric = True):

    # TODO: Fix cat_to_numeric; only apply to labels
    """
    This method normalizes (default) or standardizes all non-categorical variables in an array.
    All categorical variables are kept.

    If tranform = "normalize", categorical varibles are scaled from 0 to 1. The highest value
    is assigned the value of 1, the lowest value a will be assigned to 0.

    """

    data = data.copy()
    categorical_cols = []
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'int64':
            categorical_cols.append(col)
            if cat_to_numeric:
                data[col] = pd.Categorical(data[col]).codes


    if label_col_idx is not None:
        label = data.columns[label_col_idx]
        y     = data.pop(label)
        x     = data


    if transform == 'standardize':
        for col in x.columns:
            # if col not in categorical_cols:
            if data[col].std() !=0:
                data[col] = (data[col] - data[col].mean()) / data[col].std()

    elif transform == 'normalize':
        for col in x.columns:
            # if col not in categorical_cols:
            if data[col].max() != data[col].min():
                data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    if label_col_idx is None:
        return np.array(x)
    else:
        return np.array(x), np.array(y) # Updated y to array
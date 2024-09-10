import scipy
import numpy as np    
import pandas as pd
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


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
    
def make_swiss_s(n_samples = 100, noise = 0, random_state = None, n_categories = 3):
    """Create S curve and Swiss Data"""

    x_swiss, t_swiss = make_swiss_roll(n_samples = n_samples,
                                                noise = noise,
                                                random_state = random_state)


    x_s, t_s = make_s_curve(n_samples = n_samples,
                                     noise = noise,
                                     random_state = random_state)


    scaler = MinMaxScaler()
    # t_swiss = scaler.fit_transform(t_swiss.reshape(-1, 1))
    # t_s = scaler.fit_transform(t_s.reshape(-1, 1))


    if n_categories is not None:

        # cuts_swiss = pd.cut(t_swiss.squeeze(), bins = n_categories)
        # mapper_swiss = {cuts_swiss.categories[i]: i for i in range(len(cuts_swiss.categories))}
        # categories_swiss = cuts_swiss.map(mapper_swiss)

        # cuts_s = pd.cut(t_s.squeeze(), bins = n_categories)
        # mapper_s = {cuts_s.categories[i]: i for i in range(len(cuts_s.categories))}
        # categories_s = cuts_s.map(mapper_s)

        categories_swiss = pd.qcut(t_swiss.squeeze(), labels = False, q = n_categories)
        categories_s = pd.qcut(t_s.squeeze(), labels = False, q = n_categories)

        return x_swiss, x_s, categories_swiss, categories_s



    #X_swiss and X_s are the features. t_swiss and t_s are the labels
    return x_swiss, x_s, pd.Series(t_swiss), pd.Series(t_s)

def make_multivariate_data_set(Mean = [0,0], cov = [[0.5,0], [0,0.5]], amount = 500, adjust = 0):
    """Returns the a data set of three multivariate data blobs and the assiociated labels
    adjust is where we want to put the mover data set"""
    #Create the multivariate datasets function
    Mean = np.array(Mean) 

    # Draw 10 samples from the distribution
    dens_samples1 = np.random.multivariate_normal(Mean, cov, amount) #This is shaped (500, 2) where each point has X and Y
    dens_samples2 = np.random.multivariate_normal(Mean + 5, cov, amount) #Add five to the mean
    dens_samples3 = np.random.multivariate_normal(Mean + adjust, cov, amount) #Have this overlap with the first dataset

    #Now, we want to merge all of these dots into a single data set. Not too bad :)
    mv_data = np.array([dens_samples1, dens_samples2, dens_samples3])
    mv_data = mv_data.reshape(amount*3, 2)
    mv_labels = np.concatenate([np.repeat(0, amount), np.repeat(1, amount), np.repeat(2, amount)]) #We use numbers because thats what DTA needs

    return mv_data, mv_labels


def subset_df(df, **kwargs):
    """Each way we want to be subset should be as so 'csv_file' = "glass". It can take any column and any key.

    subset should be a DataFrame
    
    Returns subseted Dataframe"""

    for key in kwargs:
        df = df[df[key] == kwargs[key]]
    
    return df

        
def plot_in_fig(columns, rows, df = "None", plot_labels = False, **kwargs):
    """df should be the dataframe
    
    Columns should be a list of dictionaries that represent the key word arguments for plotting.
    rows should be a list of dictionaries that represent the how you want to subset the DF by
    """
    if type(df) == type("None"):
        df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/All_Data_DataFrame.csv", keep_default_na=False, na_values=['', 'NaN'], index_col= None)

    fig, axes = plt.subplots(len(rows), len(columns), figsize = (len(columns)*6, len(rows)*6))

    #Plot everything in the graphs
    row_count = 0 

    for dictionary in rows:
        df_new = subset_df(df, **dictionary)
        column_count = 0
        for column in columns:
            df_new.plot(ax = axes[row_count, column_count], **column, **kwargs)


            if plot_labels:
                summary = df_new[column.values()].describe().loc[['min', '25%', '50%', '75%', 'max']]

                # Annotate the box plot
                for value in summary.values:
                    value = float(value)
                    axes[row_count, column_count].text(1.13, value, f'{value:.2f}', horizontalalignment='center')

            #Add one to the next axis
            column_count += 1

        #Set the label
        axes[row_count, 0].set_ylabel(f"{dictionary}")

        #Add one to the next row
        row_count += 1

        
    #This gets rid of the annoying text
    plt.show()

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Parameters
    ----------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Returns
    -------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = np.linalg.norm(X0, 'fro')**2 #(X0**2.).sum()
    ssY = np.linalg.norm(Y0, 'fro')**2 #(Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def get_DataFrame_stats():
    """Prints several statements about the size and splits of the Data"""
    #Load DataFrame
    df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/ManifoldData/Data_DataFrame.csv", keep_default_na=False, na_values=['', 'NaN'], index_col= None)

    print("<><<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>      DataFrame Statistics        <><<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print(f"Total Number of instances in DataFrame: {len(df)}\n\n")

    #Print total number of samples
    print(" Method        Lengths")
    print("--------      ----------")
    print(f" JLMA:          {len(subset_df(df, method = 'JLMA'))}")
    print(f" CwDIG:          {len(subset_df(df, method = 'CwDIG'))}")
    print(f" MAGAN:          {len(subset_df(df, method = 'MAGAN'))}")
    print(f" SSMA:          {len(subset_df(df, method = 'SSMA'))}")
    print(f" DTA:          {len(subset_df(df, method = 'DTA'))}")
    print(f" Nama:          {len(subset_df(df, method = 'NAMA'))}")
    print(f" SPUD:          {len(subset_df(df, method = 'SPUD'))}")
    print(f" DIG:          {len(subset_df(df, method = 'DIG'))}\n\n")


    #Print the total number of sample per split
    print('----------------------       Splits      ----------------------')

    for split in ["random", "even", "skewed", "distort", "turn"]:
        print(f"Total data of {split}: {len(subset_df(df, split = split))}\n")
        print(f" {split}        Lengths")
        print("--------      ----------")
        print(f" MAGAN:          {len(subset_df(df, method = 'MAGAN', split = split))}")
        print(f" SSMA:          {len(subset_df(df, method = 'SSMA', split = split))}")
        print(f" CwDIG:          {len(subset_df(df, method = 'CwDIG', split = split))}")
        print(f" JLMA:          {len(subset_df(df, method = 'JLMA', split = split))}")
        print(f" DTA:          {len(subset_df(df, method = 'DTA', split = split))}")
        print(f" Nama:          {len(subset_df(df, method = 'NAMA', split = split))}")
        print(f" SPUD:          {len(subset_df(df, method = 'SPUD', split = split))}")
        print(f" DIG:          {len(subset_df(df, method = 'DIG', split = split))}\n\n")


    #Print the total number of sample per split
    print('----------------------       CSV Files      ----------------------')

    for csv_file in ["zoo", "hepatitis", "iris", "audiology", "parkinsons", "seeds", 
             "segmentation", "glass", "heart_disease", "heart_failure", "flare1", 
             "ecoli_5", "ionosphere", "Cancer_Data", "hill_valley", "balance_scale",
             "S-curve", "blobs", 
             "crx", "breast_cancer", "titanic", "diabetes", "tic-tac-toe", 
             'Medicaldataset', "water_potability",
             'treeData', 'winequality-red', 'car'
             ]:
        print(f"Total data of {csv_file}: {len(subset_df(df, csv_file = csv_file))}\n")
        print(f" {csv_file}        Lengths")
        print("--------      ----------")
        print(f" JLMA:          {len(subset_df(df, method = 'JLMA', csv_file = csv_file))}")
        print(f" MAGAN:          {len(subset_df(df, method = 'MAGAN', csv_file = csv_file))}")
        print(f" CwDIG:          {len(subset_df(df, method = 'CwDIG', csv_file = csv_file))}")
        print(f" SSMA:          {len(subset_df(df, method = 'SSMA', csv_file = csv_file))}")
        print(f" DTA:          {len(subset_df(df, method = 'DTA', csv_file = csv_file))}")
        print(f" Nama:          {len(subset_df(df, method = 'NAMA', csv_file = csv_file))}")
        print(f" SPUD:          {len(subset_df(df, method = 'SPUD', csv_file = csv_file))}")
        print(f" DIG:          {len(subset_df(df, method = 'DIG', csv_file = csv_file))}\n\n")
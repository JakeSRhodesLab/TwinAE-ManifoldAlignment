import numpy as np
from Helpers.rfgap import RFGAP
from AlignmentMethods.MAGAN import run_MAGAN, get_pure_distance, magan #TF import INCOMPATIBLE WITH GRAE
from Main.test_manifold_algorithms import test_manifold_algorithms as tma
import inspect
from mashspud import MASH, SPUD
from AlignmentMethods.jlma import JLMA
from AlignmentMethods.ssma import ssma
from AlignmentMethods.ma_procrustes import MAprocr
from AlignmentMethods.mali import MALI #TF import INCOMPATIBLE WITH GRAE
from AlignmentMethods.DTA_andres import DTA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error


def discretize_labels(regression_labels):

    """
    Transforms regression labels into ten discrete labels.
    
    Parameters:
    regression_labels (list or np.array): Array of regression labels.
    
    Returns:
    np.array: Array of discretized labels.
    """
    # Convert to numpy array if not already
    regression_labels = np.array(regression_labels)
    
    # Calculate the percentiles for discretization
    percentiles = np.percentile(regression_labels, np.arange(0, 101, 10))
    
    # Digitize the regression labels into 10 bins
    discrete_labels = np.digitize(regression_labels, percentiles, right=True) - 1
    
    # Ensure labels are in the range 0-9
    discrete_labels = np.clip(discrete_labels, 0, 9)
    
    return discrete_labels

def get_RF_score(emb, labels, seed):

    if np.issubdtype(labels.dtype, np.integer):
        rf_class = RFGAP(prediction_type="classification", y=labels, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=False, oob_score = True, random_state=seed)
    else:
        rf_class = RFGAP(prediction_type="regression", y=labels, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=False, oob_score = True, random_state=seed)
        
    #Fit it for Data A and get proximities
    rf_class.fit(emb, y = labels)
    return rf_class.oob_score_

def get_embedding_scores(emb, seed, data):
    # Unpack data
    y_A_train, y_A_test, y_B_train, y_B_test = data

    # Create X_train and X_test
    X_train = np.vstack((emb[:len(y_A_train)], emb[len(y_A_train) + len(y_A_test): len(y_A_train) + len(y_A_test) + len(y_B_train)]))
    X_test = np.vstack((emb[len(y_A_train):len(y_A_train) + len(y_A_test)], emb[-len(y_B_test):]))

    labels = np.hstack((y_A_train, y_A_test, y_B_train, y_B_test))

    # Determine knn to be 1/30 dataset size. This way we can be consistent
    knn = max(1, int(len(labels) / 30))  # Ensure knn >= 1

    # Determine if the task is classification or regression
    if np.issubdtype(labels.dtype, np.integer):
        # Classification task
        knn_model = KNeighborsClassifier(n_neighbors=knn)
        rf_model = RandomForestClassifier(random_state=seed)
        classification_task = True
    else:
        # Regression task
        knn_model = KNeighborsRegressor(n_neighbors=knn)
        rf_model = RandomForestRegressor(random_state=seed)
        classification_task = False

    y_train = np.hstack((y_A_train, y_B_train))
    y_test = np.hstack((y_A_test, y_B_test))

    # Fit and score KNN model
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    knn_score = knn_model.score(X_test, y_test)

    # Fit and score Random Forest model
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_score = rf_model.score(X_test, y_test)

    if classification_task:
        # Compute F1-score for classification
        knn_f1 = f1_score(y_test, knn_predictions, average="weighted")
        rf_f1 = f1_score(y_test, rf_predictions, average="weighted")
        return knn_score, rf_score, knn_f1, rf_f1
    else:
        # Compute RMSE for regression
        knn_rmse = np.sqrt(mean_squared_error(y_test, knn_predictions))
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
        return knn_score, rf_score, knn_rmse, rf_rmse

def get_default_parameters(cls):
    signature = inspect.signature(cls.__init__)
    defaults = {
        param.name: param.default
        for param in signature.parameters.values()
        if param.default is not param.empty
    }
    return defaults

def mash_foscttm(self):
    #Get both directions
    return np.mean([self.FOSCTTM(self.int_diff_dist[:self.len_A, self.len_A:]), self.FOSCTTM(self.int_diff_dist[self.len_A:, :self.len_A])])

def spud_foscttm(self):
    #Get both directions
    return np.mean([self.FOSCTTM(self.block[:self.len_A, self.len_A:]), self.FOSCTTM(self.block[self.len_A:, :self.len_A])])

def jlma_foscttm(self):

    len_A = len(self.X1)
    block = self.SquareDist(self.Y)

    return np.mean([tma.FOSCTTM(None, block[:len_A, len_A:]), tma.FOSCTTM(None, block[len_A:, :len_A])])

def get_mash_score_connected(self, tma, **kwargs):
    import copy
    import logging

    #Start Logging:
    logging.basicConfig(filename='/yunity/arusty/Graph-Manifold-Alignment/Resources/Pipeline.log',
                        level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger('Pipe')

    try:
        use_params = {}
        for key in kwargs.keys():
            if key in ["epochs", "threshold", "connection_limit", "hold_out_anchors"]:
                use_params[key] = kwargs[key]

        #We need to copy the class as it get changed and will be parralized
        self = copy.deepcopy(self)
        self.optimize_by_creating_connections(**use_params)

        # Cross Embedding Evaluation Metric
        emb = tma.mds.fit_transform(self.int_diff_dist)
        c_score = tma.cross_embedding_knn(emb, (tma.labels, tma.labels), knn_args={'n_neighbors': 4})
        f_score = np.mean([self.FOSCTTM(self.int_diff_dist[:self.len_A, self.len_A:]), self.FOSCTTM(self.int_diff_dist[self.len_A:, :self.len_A])])
        
        if 'hold_out_anchors' in use_params:
            del use_params['hold_out_anchors']

        print(f"MASH Parameters: {use_params}")
        print(f"                FOSCTTM {f_score}")
        print(f"                CE Score {c_score}")

        #Return FOSCTTM score
        return f_score, c_score, emb
    
    except Exception as e:
        print(f"<><><>      Tests failed for: {kwargs}. Why {e}        <><><>")
        logger.warning(f"Name: {self.method_data['Name']}. CSV: {tma.csv_file}. Parameters: {kwargs}. Error: {e}")
        return (np.NaN, np.NaN, np.NaN, np.NaN)

def Rustad_fit(self, tma, anchors):
    self.fit(tma.split_A, tma.split_B, anchors)
    return self

def get_rf_proximites(self, tuple):
    """Creates RF proximities similarities
    
        tuple should be a tuple with position 0 being the data and position 1 being the labels"""
    
    if np.issubdtype(np.array(tuple[1]).dtype, np.integer):
        rf_class = RFGAP(prediction_type="classification", y=tuple[1], prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=False)
    else:
        rf_class = RFGAP(prediction_type="regression", y=tuple[1], prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=False)
        
    #Fit it for Data A
    rf_class.fit(tuple[0], y = tuple[1])

    #Get promities
    dataA = rf_class.get_proximities()

    #Reset len_A and other varables
    if self.len_A == 2:
        self.len_A = len(tuple[0]) 

        #Change known_anchors to correspond to off diagonal matricies -- We have to change this as its dependent upon A
        self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

    elif self.len_B == 2:
        self.len_B = len(tuple[0])

    #Scale it and check to ensure no devision by 0
    if np.max(dataA[~np.isinf(dataA)]) != 0:

      dataA = (dataA - dataA.min()) / (dataA[~np.isinf(dataA)].max() - dataA.min()) 

    #Reset inf values
    dataA[np.isinf(dataA)] = 1

    return 1 - dataA

def rf_test_proximities(self, data_tuple):
    """Create on Train label and not Test"""

    X_train, X_test, y_train = data_tuple
    
    if np.issubdtype(np.array(y_train).dtype, np.integer):
        rf_class = RFGAP(prediction_type="classification", y=y_train, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=False)
    else:
        rf_class = RFGAP(prediction_type="regression", y=y_train, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=False)
        
    #Fit it for Data A
    rf_class.fit(X_train, y =y_train)

    #Get promities
    data_partial = rf_class.get_proximities()
    data_extend = rf_class.prox_extend(X_test)

    # Create the new  matrix filled with ones (as it will be swapped to 0s later)
    n = len(data_partial)
    m = len(data_extend)

    data = np.zeros((n + m, n + m))

    # Place the n x n matrix in the top-left corner
    data[:n, :n] = data_partial

    # Place the n x m matrix in the top-right corner
    data[:n, n:] = data_extend.T

    # Place the n x m matrix in the bottom-left corner
    data[n:, :n] = data_extend

    #Reset len_A and other varables
    if self.len_A < 6:
        self.len_A = n + m

        #Change known_anchors to correspond to off diagonal matricies -- We have to change this as its dependent upon A
        self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

    elif self.len_B < 6:
        self.len_B = n + m

    #Scale it and check to ensure no devision by 0
    if np.max(data[~np.isinf(data)]) != 0:

      data = (data - data.min()) / (data[~np.isinf(data)].max() - data.min()) 

    #Reset inf values
    data[np.isinf(data)] = 0

    np.fill_diagonal(data, 1)

    return 1 - data

def Rhodes_fit(self, tma, anchors):
    """RF Gap Fit for the methods"""

    #Reset these variables
    self.distance_measure_A = get_rf_proximites
    self.distance_measure_B = get_rf_proximites

    self.fit(dataA = (tma.split_A, tma.labels), dataB = (tma.split_B, tma.labels), known_anchors=anchors)
    return self

def Rhodes_test_fit(self, data_tupleA, data_tupleB, anchors):
    """RF Gap Fit for the methods"""

    #Reset these variables
    self.distance_measure_A = rf_test_proximities
    self.distance_measure_B = rf_test_proximities

    self.fit(dataA = data_tupleA, dataB = data_tupleB, known_anchors=anchors)
    return self

def Andres_fit(self, tma, anchors):
    #Reformat the anchors 
    sharedD1 = tma.split_A[anchors.T[0]] 
    sharedD2 = tma.split_B[anchors.T[1]]
    labelsh1 = tma.labels[anchors.T[0]]
    
    #We only need to overide the labels once otherwise it will mess up the CE score
    if len(tma.labels) != (len(labelsh1) + len(tma.split_A)):
        tma.labels = np.concatenate((tma.labels, labelsh1))
        tma.labels_doubled = np.concatenate((tma.labels, tma.labels))


    self.fit(tma.split_A, tma.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2)

    return self

def MAGAN_fit(self, tma, anchors):

    #Fit, and initilize model
    domain_a, domain_b, domain_ab, domain_ba = run_MAGAN(tma.split_A, tma.split_B, anchors, self.learning_rate)

    #Reshape the domains
    domain_a, domain_b = get_pure_distance(domain_a, domain_b)
    domain_ab, domain_ba = get_pure_distance(domain_ab, domain_ba)
    
    #Return a different thing back to calculate FOSCTTM and CE
    return [domain_a, domain_b, domain_ab, domain_ba]

def get_MAGAN_block(block_pieces):
    #Return the block
    return np.block([[block_pieces[0], block_pieces[3]], [block_pieces[3], block_pieces[1]]])

def magan_foscttm(block_pieces):
    return np.mean((tma.FOSCTTM(None, block_pieces[2]), tma.FOSCTTM(None, block_pieces[3])))

def pcr_foscttm(self):
    len_A = int(self.W.shape[0]/2)
                    
    return np.mean([tma.FOSCTTM(None, 1 - self.W[len_A:, :len_A]), 
                       tma.FOSCTTM(None, 1 - self.W[:len_A, len_A:])])

def fit_with_labels(self, tma, anchors):
    labels = discretize_labels(tma.labels)

    self.fit((tma.split_A, tma.split_B), (labels, labels))

    return self

def create_unique_pairs(max_num, num_pairs):
    import random 
    
    # Ensure there are enough numbers for unique selection
    if num_pairs * 2 > max_num:
        raise ValueError("Not enough unique numbers to create the specified number of pairs.")
    
    # Generate a pool of unique numbers
    random.seed(42)
    numbers = random.sample(range(max_num), num_pairs)
    
    # Create pairs from the list of unique numbers
    pairs = [[numbers[i], numbers[i]] for i in range(0, len(numbers))]
    
    return pairs

#Create dictionaries for the different classes
method_dict = {
     #Default
     "MASH-" : {"Name": "MASH-", "Model": MASH, "KNN" : True,   "Block" : lambda mash: mash.int_diff_dist, "FOSCTTM" : mash_foscttm, "Fit" : Rustad_fit},
     "MASH" : {"Name": "MASH", "Model": MASH, "KNN" : True,   "Block" : lambda mash: mash.int_diff_dist, "FOSCTTM" : mash_foscttm, "Fit" : Rustad_fit},
     "SPUD" : {"Name": "SPUD", "Model": SPUD, "KNN" : True,   "Block" : lambda spud: spud.block, "FOSCTTM" : spud_foscttm, "Fit" : Rustad_fit},
     "NAMA" : {"Name": "NAMA", "Model": SPUD, "KNN" : False,   "Block" : lambda spud: spud.block, "FOSCTTM" : spud_foscttm, "Fit" : Rustad_fit},
     
     #RFGAP
     "RF-MASH-" : {"Name": "RF-MASH-", "Model": MASH, "KNN" : True,   "Block" : lambda mash: mash.int_diff_dist, "FOSCTTM" : mash_foscttm, "Fit" : Rhodes_fit},
     "RF-MASH" : {"Name": "RF-MASH", "Model": MASH, "KNN" : True,   "Block" : lambda mash: mash.int_diff_dist, "FOSCTTM" : mash_foscttm, "Fit" : Rhodes_fit},
     "RF-SPUD" : {"Name": "RF-SPUD", "Model": SPUD, "KNN" : True,   "Block" : lambda spud: spud.block, "FOSCTTM" : spud_foscttm, "Fit" : Rhodes_fit},
     "RF-NAMA" : {"Name": "RF-NAMA", "Model": SPUD, "KNN" : False,   "Block" : lambda spud: spud.block, "FOSCTTM" : spud_foscttm, "Fit" : Rhodes_fit},
     "RF-MALI" : {"Name": "MALI-RF", "Model": MALI, "KNN" : True,  "Block" : lambda mali: ((1 - mali.W.toarray()) + (1 - mali.W.toarray()).T) /2, "FOSCTTM" : lambda mali: tma.FOSCTTM(None, 1 - mali.W_cross.toarray()), "Fit": fit_with_labels},

     

     #NOTE: adopted fit below
     "DTA" : {"Name": "DTA", "Model": DTA, "KNN" : True,   "Block" : lambda dta: 1 - tma.normalize_0_to_1(None, dta.W), "FOSCTTM" : lambda dta : tma.FOSCTTM(None, 1 - tma.normalize_0_to_1(None, dta.W12)), "Fit": Andres_fit},
     "SSMA" : {"Name": "SSMA", "Model": ssma, "KNN" : True,   "Block" : lambda ssma: 1 - tma.normalize_0_to_1(None, ssma.W), "FOSCTTM" : lambda ssma : tma.FOSCTTM(None, 1 - ssma.W[len(ssma.domain1):, :len(ssma.domain1)]), "Fit": Andres_fit},
     "MAPA" : {"Name": "MAPA", "Model": MAprocr, "KNN" : True,   "Block" : lambda pcr: 1 - tma.normalize_0_to_1(None, pcr.W), "FOSCTTM" : pcr_foscttm, "Fit": Andres_fit},

     "MAGAN" : {"Name": "MAGAN", "Model": magan, "KNN" : False,   "Block" : get_MAGAN_block, "FOSCTTM" : magan_foscttm, "Fit": MAGAN_fit},
     "JLMA" : {"Name": "JLMA", "Model": JLMA, "KNN" : True,   "Block" : lambda jlma: jlma.SquareDist(jlma.Y), "FOSCTTM" : jlma_foscttm, "Fit": Rustad_fit},
     
     "MALI" : {"Name": "MALI", "Model": MALI, "KNN" : True,  "Block" : lambda mali: ((1 - mali.W.toarray()) + (1 - mali.W.toarray()).T) /2, "FOSCTTM" : lambda mali: tma.FOSCTTM(None, 1 - mali.W_cross.toarray()), "Fit": fit_with_labels},
     #Not worth doing KEMA. It only uses knn, and so we can just use the old testing way. :) "KEMA" : {"Name": "KEMA", "Model": MALI, "KNN" : True,  "Block" : lambda mali: ((1 - mali.W.toarray()) + (1 - mali.W.toarray()).T) /2, "FOSCTTM" : lambda mali: tma.FOSCTTM(None, 1 - mali.W_cross.toarray()), "Fit": fit_with_labels}
}

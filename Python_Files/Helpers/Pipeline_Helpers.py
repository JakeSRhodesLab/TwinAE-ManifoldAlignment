import numpy as np
from Helpers.rfgap import RFGAP
from AlignmentMethods.MAGAN import run_MAGAN, get_pure_distance, magan
from Main.test_manifold_algorithms import test_manifold_algorithms as tma
import inspect
from mashspud import MASH, SPUD
from AlignmentMethods.jlma import JLMA
from AlignmentMethods.ssma import ssma
from AlignmentMethods.ma_procrustes import MAprocr
from AlignmentMethods.mali import MALI
from AlignmentMethods.DTA_andres import DTA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


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

def get_RF_score(emb, labels):

    if np.issubdtype(labels.dtype, np.integer):
        rf_class = RFGAP(prediction_type="classification", y=labels, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=True, oob_score = True)
    else:
        rf_class = RFGAP(prediction_type="regression", y=labels, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=True, oob_score = True)
        
    #Fit it for Data A and get proximities
    rf_class.fit(emb, y = labels)
    return rf_class.oob_score_

def get_KNN_score(emb, labels):

    #Determine knn to be 1/30 dataset size. This way we can be consistent
    knn = int(len(labels)/30)

    # Determine if the task is classification or regression
    if np.issubdtype(labels.dtype, np.integer):
        model = KNeighborsClassifier(n_neighbors = knn)
    else:
        model = KNeighborsRegressor(n_neighbors = knn)

    model.fit(emb, labels)
    return model.score(emb, labels)

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
        rf_score = get_RF_score(emb, tma.labels_doubled)
        knn_score = get_KNN_score(emb, tma.labels_doubled)


        if 'hold_out_anchors' in use_params:
            del use_params['hold_out_anchors']

        print(f"MASH Parameters: {use_params}")
        print(f"                FOSCTTM {f_score}")
        print(f"                CE Score {c_score}")
        print(f"                RF Score {rf_score}")
        print(f"                KNN Score {knn_score}")

        #Return FOSCTTM score
        return c_score, f_score, rf_score, knn_score
    
    except Exception as e:
        print(f"<><><>      Tests failed for: {kwargs}. Why {e}        <><><>")
        logger.warning(f"Name: {self.method_data['Name']}. CSV: {tma.csv_file}. Parameters: {kwargs}. Error: {e}")
        return (np.NaN, np.NaN, np.NaN, np.NaN)


def Rustad_fit(self, tma, anchor_amount):
    self.fit(tma.split_A, tma.split_B, tma.anchors[:anchor_amount])
    return self

def Andres_fit(self, tma, anchor_amount):
    #Reformat the anchors 
    sharedD1 = tma.split_A[tma.anchors[:anchor_amount].T[0]] 
    sharedD2 = tma.split_B[tma.anchors[:anchor_amount].T[1]]
    labelsh1 = tma.labels[tma.anchors[:anchor_amount].T[0]]
    
    #We only need to overide the labels once otherwise it will mess up the CE score
    if len(tma.labels) != (len(labelsh1) + len(tma.split_A)):
        tma.labels = np.concatenate((tma.labels, labelsh1))

    self.fit(tma.split_A, tma.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2)

    return self

def MAGAN_fit(self, tma, anchor_amount):

    #Fit, and initilize model
    domain_a, domain_b, domain_ab, domain_ba = run_MAGAN(tma.split_A, tma.split_B, tma.anchors[:anchor_amount], self.learning_rate)

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

def fit_with_labels(self, tma, anchor_amount):
    labels = discretize_labels(tma.labels)

    self.fit((tma.split_A, tma.split_B), (labels, labels))

    return self

#Create dictionaries for the different classes
method_dict = {
     "MASH-" : {"Name": "MASH-", "Model": MASH, "KNN" : True,   "Block" : lambda mash: mash.int_diff_dist, "FOSCTTM" : mash_foscttm, "Fit" : Rustad_fit},
     "MASH" : {"Name": "MASH", "Model": MASH, "KNN" : True,   "Block" : lambda mash: mash.int_diff_dist, "FOSCTTM" : mash_foscttm, "Fit" : Rustad_fit},
     "SPUD" : {"Name": "SPUD", "Model": SPUD, "KNN" : True,   "Block" : lambda spud: spud.block, "FOSCTTM" : spud_foscttm, "Fit" : Rustad_fit},
     "NAMA" : {"Name": "NAMA", "Model": SPUD, "KNN" : False,   "Block" : lambda spud: spud.block, "FOSCTTM" : spud_foscttm, "Fit" : Rustad_fit},
     
     #NOTE: adopted fit below
     "DTA" : {"Name": "DTA", "Model": DTA, "KNN" : True,   "Block" : lambda dta: 1 - tma.normalize_0_to_1(None, dta.W), "FOSCTTM" : lambda dta : tma.FOSCTTM(None, 1 - tma.normalize_0_to_1(None, dta.W12)), "Fit": Andres_fit},
     "SSMA" : {"Name": "SSMA", "Model": ssma, "KNN" : True,   "Block" : lambda ssma: 1 - tma.normalize_0_to_1(None, ssma.W), "FOSCTTM" : lambda ssma : tma.FOSCTTM(None, 1 - ssma.W[len(ssma.domain1):, :len(ssma.domain1)]), "Fit": Andres_fit},
     "PCR" : {"Name": "PCR", "Model": MAprocr, "KNN" : True,   "Block" : lambda pcr: 1 - tma.normalize_0_to_1(None, pcr.W), "FOSCTTM" : pcr_foscttm, "Fit": Andres_fit},

     "MAGAN" : {"Name": "MAGAN", "Model": magan, "KNN" : False,   "Block" : get_MAGAN_block, "FOSCTTM" : magan_foscttm, "Fit": MAGAN_fit},
     "JLMA" : {"Name": "JLMA", "Model": JLMA, "KNN" : True,   "Block" : lambda jlma: jlma.SquareDist(jlma.Y), "FOSCTTM" : jlma_foscttm, "Fit": Rustad_fit},
     
     "MALI" : {"Name": "MALI", "Model": MALI, "KNN" : True,  "Block" : lambda mali: ((1 - mali.W.toarray()) + (1 - mali.W.toarray()).T) /2, "FOSCTTM" : lambda mali: tma.FOSCTTM(None, 1 - mali.W_cross.toarray()), "Fit": fit_with_labels}
}

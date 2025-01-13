#This tests all of our models against each other

"""
----------------------------------------------------------     Helpful Information      ----------------------------------------------------------
Supercomputers Access: carter, collings, cox, hilton, rencher, and tukey
Resource Monitor Websitee: http://statrm.byu.edu/
KanbanFlow: https://kanbanflow.com/board/g25iW5x
"""
#I should fix these imports to be more localized
#Import libraries
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, KNeighborsRegressor
from sklearn.manifold import MDS
import os
from joblib import Parallel, delayed
from Helpers.rfgap import RFGAP
import Helpers.utils as utils
from scipy.spatial.distance import pdist, squareform

#Simply, for my sanity
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

CURR_DIR = "/yunity/arusty/Graph-Manifold-Alignment/Results"

#Directory Constant
MANIFOLD_DATA_DIR = CURR_DIR + "/ManifoldData/"

#Needed function
#Create an RF Proximities function
def use_rf_proximities(self, tuple):
    """Creates RF proximities similarities
    
        tuple should be a tuple with position 0 being the data and position 1 being the labels"""
    #Initilize Class
    rf_class = RFGAP(prediction_type="classification", y=tuple[1], prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=False)

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

#Create function to do everything
class test_manifold_algorithms():
    def __init__(self, csv_file, split = "random", percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5],  verbose = 0, random_state = 42):
        """csv_file should be the name of the csv file. If set to 'S-curve' or "blobs", it will create a toy data set. 
        
        split can be 'skewed' (for the features to be split by more important and less important),
        or 'random' for the split to be completely random, or 'even' for each split to have both 
        important or unimportant features. If split = "distort", then it will create a second dataset, 
        with the features distorted in the second one. 
        
        Verbose has different levels. 0 includes no additional prints. 1 prints a little bit more, and
        2 everything."""

        self.verbose = verbose


        self.random_state = random_state
        random.seed(self.random_state)

        global MANIFOLD_DATA_DIR
        self.base_directory = MANIFOLD_DATA_DIR + csv_file[:-4] + "/"

        #Since Blobs does not require a csv file, if it is chosen we make the dataset in house
        if csv_file == "blobs":
            self.create_blobs()

            #Just so all of the file naming conventions remain the same
            csv_file = "blobs.csv"
            
        #Since Scurve does not require reading a csv file, we have a different process
        elif csv_file != "S-curve":
            self.split = split
            self.prep_data(csv_file)

        else:
            #Create labels, and Data
            self.create_Scurve()

            #Just so all of the file naming conventions remain the same
            csv_file = "S-curve.csv"

        if self.verbose > 0:
            print(f"\n \n \n---------------------------       Initalizing class with {csv_file} data       ---------------------------\n")

        #Create anchors
        self.anchors = self.create_anchors()

        #Testing the amount of anchors
        self.percent_of_anchors = percent_of_anchors

        #Set our KNN range dependent on the amount of values in the dataset
        self.knn_range = tuple(self.find_knn_range())
        if verbose > 1:
            print(f"The knn values are: {self.knn_range}")

    """EVALUATION FUNCTIONS"""
    def cross_embedding_knn(self, embedding, Y, knn_args = {'n_neighbors': 4}, other_side = True):
        (y1, y2) = Y

        n1, n2 = len(y1), len(y2)

        # Determine if the task is classification or regression
        if np.issubdtype(y1.dtype, np.integer):
            knn = KNeighborsClassifier(**knn_args)
            print("Using a classifier")
        else:
            knn = KNeighborsRegressor(**knn_args)
            print("Using a regression model")

        if other_side:
            knn.fit(embedding[:n1, :], y1)

            return knn.score(embedding[n1:, :], y2)

        else:
            #Train on other domain, predict on other domain ---- TODO
            knn.fit(embedding[n1:, :], y2)

            return knn.score(embedding[:n1, :], y1)
        
    def FOSCTTM(self, Wxy): #Wxy should be just the parrallel matrix
        n1, n2 = np.shape(Wxy)
        if n1 != n2:
            raise AssertionError('FOSCTTM only works with a one-to-one correspondence. ')

        dists = Wxy

        nn = NearestNeighbors(n_neighbors = n1, metric = 'precomputed')
        nn.fit(dists)

        _, kneighbors = nn.kneighbors(dists)

        return np.mean([np.where(kneighbors[i, :] == i)[0] / n1 for i in range(n1)])
    
    """HELPER FUNCTIONS"""
    def split_features(self, features, labels):

        #Step 1. Check if a file exists already
        #Create filename 
        filename = CURR_DIR + "/Splits_Data/" + self.base_directory[len(MANIFOLD_DATA_DIR):]

        if not os.path.exists(filename):
            os.makedirs(filename) 

        filename += self.split[0] + str(self.random_state) + ".npz"

        #Step 2b. If so, simply load the files into split A and split B
        if os.path.exists(filename):

            #Load in the file
            data = np.load(filename) 

            #Grab the splits
            return data['split_a'], data["split_b"]

        #Step 2a. if not, Do the methodology we have before
        else:

            if self.split == "random":
                if self.verbose > 0:
                    print("Splitting the data randomly")

                # Generate column indices and shuffle them
                column_indices = np.arange(features.shape[1])
                np.random.shuffle(column_indices)

                # Choose a random index to split the shuffled column indices
                split_index = random.randint(1, len(column_indices) - 1)

                # Use the shuffled indices to split the features array into two parts
                split_a = features[:, column_indices[:split_index]]
                split_b = features[:, column_indices[split_index:]]

            elif self.split == "turn":
                rng = np.random.default_rng(self.random_state)
                n, d = np.shape(features)
                random_matrix = rng.random((d, d))
                q, _ = np.linalg.qr(random_matrix)

                split_a = features
                split_b = features @ q
            
            elif self.split == "distort":
                if self.verbose > 0:
                    print("Creating a mirror dataset and distorting the features in the second Domain")

                #Split A remains the same
                split_a = features

                #Add noise to split B
                split_b = features + np.random.normal(scale = 0.05, size = np.shape(features))

            else: 
        
                # Splitting the dataset into training and testing sets
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

                # Training the RandomForest Classifier
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                if np.issubdtype(labels.dtype, np.integer):
                    clf = RandomForestClassifier(random_state=self.random_state) #NOTE: this might take forever based on this algorithm 
                else:
                    clf = RandomForestRegressor(random_state=self.random_state)

                clf.fit(X_train, y_train)

                from sklearn.inspection import permutation_importance
                result = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=self.random_state)
                # Get the indices that would sort the importances
                sorted_idx = result.importances_mean.argsort()

                if self.split == "skewed":
                    if self.verbose > 0:
                        print("Splitting the data in a skewed fashion")
                    #Split the data at the half point
                    half_index = int(len(sorted_idx)/2)

                    #Since its sorted by importance, the more important features will be the first half. We return all the important, and then the less important features
                    split_a = features[:, (sorted_idx[:half_index])]
                    split_b = features[:, (sorted_idx[half_index:])]
                
                elif self.split == "even":
                    if self.verbose > 0:
                        print("Spliting the data evenly")

                    #Get a list of indexes to include
                    indexes = np.array(range(0, len(sorted_idx), 2))

                    #Use the above indexes to retrieve the balanced indexes
                    split_a_indexes = sorted_idx[indexes]
                    try: #Because if feature count is odd this will fail
                        split_b_indexes = sorted_idx[indexes + 1]
                    except:
                        split_b_indexes = sorted_idx[indexes[:-1] + 1]
                    
                    #Now retrieve the values
                    split_a = features[:, split_a_indexes]
                    split_b = features[:, split_b_indexes]
                
                else:
                    raise NameError("Split type not recognized. Please type 'even', 'skewed', 'distort', or 'random'.")

            #Reshape if they only have one sample
            if split_a.shape[1] == 1:
                split_a = split_a.reshape(-1, 1)
            if split_b.shape[1] == 1:
                split_b = split_b.reshape(-1, 1)

            #Print how they are shaped if verbose is 2 or more
            if self.verbose > 1:
                    print(f"Split A features shape: {split_a.shape}")
                    print(f"Split B Features shape {split_b.shape}")

            #Step 3. If not, upload to file. CSV file name and split. 
            np.savez(filename, split_a=split_a, split_b=split_b)

            return split_a, split_b

    def create_blobs(self): #TODO: FINISH DAKINE
        """Creates 3 blobs for each split"""
        self.split_A, self.labels = utils.make_multivariate_data_set(amount=100)
        self.split_B, labels2 = utils.make_multivariate_data_set(amount = 100, adjust=5)

        #Use both labels
        self.labels_doubled = np.concatenate((self.labels, labels2))

        #Create the mds
        self.n_comp = 2
        self.mds = MDS(metric=True, dissimilarity = 'precomputed', n_init = 1, random_state = self.random_state, n_components = 2)

        self.split = "None"

    # TODO: May want to add path as an argument, with the current data path as default
    def prep_data(self, csv_file):

        #Create the base directory
        #Modify Directory Constant
        global MANIFOLD_DATA_DIR

        #Read in file and seperate feautres and labels
        try: #Will fail if not there
            df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Classification_CSV/" + csv_file)
            regression = False
        except:
            
            regression = True
            MANIFOLD_DATA_DIR = CURR_DIR + "/RegressionData/"
            df = pd.read_csv("/yunity/arusty/Graph-Manifold-Alignment/Resources/Regression_CSV/" + csv_file)

        features, self.labels = utils.dataprep(df, label_col_idx=0)
        
        self.base_directory = MANIFOLD_DATA_DIR + csv_file[:-4] + "/"


        #Ensure that labels are continuous
        if not regression:
            unique_values, inverse = np.unique(self.labels, return_inverse=True)
            self.labels = inverse + 1

        #Create file directory to store the information
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory) 

        #Split the features, and prep data
        self.split_A, self.split_B = self.split_features(features, self.labels)
        self.labels_doubled = np.concatenate((self.labels, self.labels))

        #We just assume they want the scomponents to be the number of features
        self.n_comp = max(min(len(self.split_B[1]), len(self.split_A[1])), 2)
        self.mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = self.random_state, n_components = self.n_comp)

        if self.verbose > 1:
            print(f"MDS initialized with {self.n_comp} components")

    def create_Scurve(self):
        """Create Scurve Data"""
        if self.verbose > 1:
                print(f"Creating swiss rolls and S curve data")

        self.split = "None"

        #Create all the data
        self.split_A, self.split_B, self.labels, labels2  = utils.make_swiss_s(n_samples = 200, noise = 0, random_state = self.random_state, n_categories = 3)

        #Double the labels so we can evaluate the effectiveness of the methods
        self.labels_doubled = np.concatenate((self.labels, labels2))

        #Set and train the mds
        self.n_comp = 2
        self.mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = self.random_state, n_components = 2)

    def create_anchors(self):
        #Generate anchors that can be subsetted
        rand_ints = random.sample(range(len(self.labels)), int(len(self.labels)))
        return np.vstack([rand_ints, rand_ints]).T

    def find_knn_range(self):
        """Returns a list of probable KNN values. Each list will have ten values"""
        small_data_size = np.min((len(self.split_A), len(self.split_B)))

        #We want a bigger increment of the data is larger
        if small_data_size < 101:
            return range(2, small_data_size, 2)[:10] #This makes us error proof so we wont have a knn value larger than the set, and only gives us ten. It is 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
        else:
            #Set an increment that gets larger based on the dataset
            increment = (small_data_size // 50) + 1

        #Manufacture a stopping point that will give us ten values
        stop = (increment * 10) + 2

        return range(2, stop, increment) #2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32 | 2, 7, 12, 17, 22, 27, 32, 37, 42, 47 | (9 times increment minus 2)

    def normalize_0_to_1(self, value):
        return (value - value.min()) / (value.max() - value.min())
    
    def create_filename(self, method, **kwargs):
        """Creates the filename to save the file"""
        method = str(method)

        #Create filename 
        filename = self.base_directory + method + '(' + self.split[0] + str(self.random_state) + ')'

        #Loop though the Different opperations
        for key in kwargs:
            filename += "_" + str(key)[:3] + "(" + str(kwargs[key]) + ')'

        #Add in the Anchors
        filename += "_AP(" #Short for Anchor Percent

        #Now check to see if we have run these tests with some anchor percents already
        import glob
        matching_files = glob.glob(filename + "*")

        #Loop through and see what anchor percents we have already used
        AP_values = set([])
        for file in matching_files:
            #Get the KNN increment, and then shrink the file to right size
            knn_upper_bound = file.split("_")[-1]
            file = file[:-(len(knn_upper_bound)+2)]

            #Get the Anchor Percents
            AP_index = file.find('_AP(')
            AP_values = AP_values.union(set([float(num) for num in file[AP_index + 4:].split('-')]))

        #Get the unused AP_values
        AP_values = (set(self.percent_of_anchors) - AP_values)

        #Add in the percent of anchors
        for name in AP_values:
            filename += str(name) + "-"

        #Add the last index for knn range so we know what knn values were used
        filename = filename[:-1] + ")_" + str(self.knn_range[-1])

        #Finish file name
        filename = filename + ".npy"

        return filename, AP_values

    """RUN TESTS FUNCTIONS"""
    def run_RF_SPUD_tests(self, agg_methods = ["log"], OD_methods = ["default"]): 
        """Operations should be a tuple of the different operations wanted to run. All are included by default. """

        from mashspud import SPUD

        #We are going to run test with every variation
        print(f"\n-------------------------------------    SPUD RF Tests " + self.base_directory[53:-1] + "   -------------------------------------\n")
        for agg_method in agg_methods:
            print(f"Aggregation method {agg_method}")

            for OD_method in OD_methods:
                print(f"    Off-diagonal method {OD_method}")


                #Create file directory to store the information
                original_directory = self.base_directory
                self.base_directory = CURR_DIR + "/ManifoldData_RF/" + self.base_directory[len(MANIFOLD_DATA_DIR):]
                

                #Create files and store data
                filename, AP_values = self.create_filename("SPUD_RF", agg_method = agg_method, OD_method = OD_method) 

                #Reset the directory
                self.base_directory = original_directory 
                

                #If file aready exists, then we are done :)
                if os.path.exists(filename) or len(AP_values) < 1:
                    print(f"        <><><><><>    File {filename} already exists   <><><><><>")
                    continue

                #Store the data in a numpy array
                spud_scores = np.zeros((len(self.knn_range), len(AP_values), 2))

                for k, knn in enumerate(self.knn_range):
                    print(f"        KNN {knn}")
                    for l, anchor_percent in enumerate(AP_values):
                        print(f"            Percent of Anchors {anchor_percent}")

                        try:
                            #If KNN is the highest value, use NAMA approach instead
                            if k == 9 and OD_method != "default":
                                print("            Using NAMA approach.")
                                similarity_measure = "NAMA"
                            else:
                                similarity_measure = "distances"

                            #Create the class with all the arguments
                            spud_class = SPUD(knn = knn, agg_method = agg_method, OD_method = OD_method, distance_measure_A = use_rf_proximities, distance_measure_B = use_rf_proximities, overide_method = similarity_measure, n_pca = 100) #self.split_A, self.split_B, known_anchors=self.anchors[:int(len(self.anchors) * anchor_percent)]
                            spud_class.fit(dataA = (self.split_A, self.labels), dataB = (self.split_B, self.labels), known_anchors=self.anchors[:int(len(self.anchors) * anchor_percent)])

                        except Exception as e:
                            print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e} TEST FAILED   <><><><><><>")
                            spud_scores[k, l, 0] = np.NaN
                            spud_scores[k, l, 1] = np.NaN
                            continue

                        #FOSCTTM METRICS
                        try:
                            spud_FOSCTTM = self.FOSCTTM(spud_class.block[:spud_class.len_A, spud_class.len_A:])
                            print(f"                FOSCTTM Score: {spud_FOSCTTM}")
                        except Exception as e:
                            print(f"                FOSCTTM exception occured: {e}")
                            spud_FOSCTTM = np.NaN
                        
                        spud_scores[k, l, 0] = spud_FOSCTTM

                        #Cross Embedding Metrics
                        try:
                            emb = self.mds.fit_transform(spud_class.block)
                            spud_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                            print(f"                CE Score: {spud_CE}")
                        except Exception as e:
                            print(f"                Cross Embedding exception occured: {e}")
                            spud_CE = np.NaN
                        
                        spud_scores[k, l, 1] = spud_CE
                
                #Save the numpy array
                np.save(filename, spud_scores)
        
        #Run successful
        return True
    
    def run_KEMA_tests(self, kernelts = ["lin"]): 
        """Operations should be a tuple of the different operations wanted to run. All are included by default. """

        #We are going to run test with every variation
        print(f"\n-------------------------------------    KEMA Tests " + self.base_directory[53:-1] + "   -------------------------------------\n")

        for kernelt in kernelts:
            print(f"Kenelt: {kernelt}")

            #Create file directory to store the information
            original_directory = self.base_directory
            self.base_directory = CURR_DIR + "/ManifoldData_RF/" + self.base_directory[len(MANIFOLD_DATA_DIR):]

            #Create files and store data
            filename, AP_values = self.create_filename("KEMA_RF", kernelt = kernelt) 

            #Reset the directory
            self.base_directory = original_directory 
            
            #If file aready exists, then we are done :)
            if os.path.exists(filename) or len(AP_values) < 1:
                print(f"        <><><><><>    File {filename} already exists   <><><><><>")
                return True

            #Store the data in a numpy array
            kema_scores = np.zeros((len(self.knn_range), 2))

            #Start MATLAB Engine  #This is a long import -> 4 seconds. Starting the engine takes 18 seconds
            import matlab.engine
            eng = matlab.engine.start_matlab()
            eng.cd(r'/yunity/arusty/Graph-Manifold-Alignment/KEMA/general_routine', nargout=0)

            labeled = [
                {'X': matlab.double((self.split_A.T).tolist()), 'Y': matlab.double(self.labels.reshape(-1, 1).tolist())}, # It needs this shape to be transposed....
                {'X': matlab.double((self.split_B.T).tolist()), 'Y': matlab.double(self.labels.reshape(-1, 1).tolist())}
            ]

            unlabeled = [
                {'X': matlab.double([])},
                {'X': matlab.double([])}
            ]

            for k, knn in enumerate(self.knn_range):
                print(f"  KNN {knn}")

                try:
                    #Create the class with all the arguments
                    ALPHA, LAMBDA, options = eng.KMA(labeled, unlabeled,  {"kernelt" : kernelt, "debug" : 0, "nn" : knn}, nargout = 3, background = False) #We can change the kernelt too

                    # 3) project test data

                    # Convert to MATLAB types
                    test = [
                        {'X': matlab.double((self.split_A.T).tolist())},
                        {'X': matlab.double((self.split_B.T).tolist())}
                    ]

                    save_files = 0
                    emb = eng.Adam_MATLAB(labeled, unlabeled, test, ALPHA, options, save_files, self.n_comp, nargout = 1, background = False)

                    #Convert to array
                    emb = np.array(emb)

                    # Get the middle dimension
                    middle_dim = emb.shape[1]

                    # Flatten the array while preserving the structure
                    emb = emb.transpose(1, 0, 2).reshape(middle_dim, -1).T


                except Exception as e:
                    print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e} TEST FAILED   <><><><><><>")
                    kema_scores[k, 0] = np.NaN
                    kema_scores[k, 1] = np.NaN
                    continue

                #FOSCTTM METRICS
                try:
                    #Create a FOSCTTM Like Field
                    x_dists = squareform(pdist(emb))

                    #normalize it
                    block = x_dists / np.max(x_dists, axis = None)
                    
                    len_A = len(self.split_A)
                    kema_FOSCTTM = self.FOSCTTM(block[:len_A, len_A:])
                    print(f"                FOSCTTM Score: {kema_FOSCTTM}")
                except Exception as e:
                    print(f"                FOSCTTM exception occured: {e}")
                    kema_FOSCTTM = np.NaN
                
                kema_scores[k, 0] = kema_FOSCTTM

                #Cross Embedding Metrics
                try:
                
                    kema_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})

                    print(f"                CE Score: {kema_CE}")

                except Exception as e:
                    print(f"                Cross Embedding exception occured: {e}")
                    kema_CE = np.NaN
                
                kema_scores[k, 1] = kema_CE
        
            #Save the numpy array
            np.save(filename, kema_scores)

        #Run successful
        return True
    
    def run_MALI_tests(self, graph_distances = ["rf_gap", "default"]): #interclass distance set to be RF-Gap
        """Operations should be a tuple of the different operations wanted to run. All are included by default. """
        from AlignmentMethods.mali import MALI

        #We are going to run test with every variation
        print(f"\n-------------------------------------    MALI Tests " + self.base_directory[53:-1] + "   -------------------------------------\n")

        for graph_distance in graph_distances:
            print(f"Using Graph Distance: {graph_distance}")

            #Create file directory to store the information
            original_directory = self.base_directory
            self.base_directory = CURR_DIR + "/ManifoldData_RF/" + self.base_directory[len(MANIFOLD_DATA_DIR):]

            #Create files and store data
            if graph_distance == "default":
                filename, AP_values = self.create_filename("MALI_RF") 
            else:
                filename, AP_values = self.create_filename("MALI") 

            #Reset the directory
            self.base_directory = original_directory 

            #If file aready exists, then we are done :)
            if os.path.exists(filename) or len(AP_values) < 1:
                print(f"        <><><><><>    File {filename} already exists   <><><><><>")    
                return True
            

            #Store the data in a numpy array
            mali_scores = np.zeros((len(self.knn_range), 2))

            for k, knn in enumerate(self.knn_range):
                print(f"  KNN {knn}")

                try:
                    #Create the class with all the arguments
                    if graph_distance == "default":
                        mali_class = MALI(knn = knn, graph_decay=40, graph_knn = knn, random_state = self.random_state)
                    else:
                        mali_class = MALI(knn = knn, graph_decay=40, graph_knn = knn, random_state = self.random_state, interclass_distance = "rfgap") #graph_distance = "rfgap",interclass_distance = "rfgap"

                    mali_class.fit((self.split_A, self.split_B), (self.labels, self.labels))

                except Exception as e:
                    print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e} TEST FAILED   <><><><><><>")
                    mali_scores[k, 0] = np.NaN
                    mali_scores[k, 1] = np.NaN
                    continue

                #FOSCTTM METRICS
                try:
                    mali_FOSCTTM = self.FOSCTTM(1 - mali_class.W_cross.toarray())
                    print(f"                FOSCTTM Score: {mali_FOSCTTM}")
                except Exception as e:
                    print(f"                FOSCTTM exception occured: {e}")
                    mali_FOSCTTM = np.NaN
                
                mali_scores[k, 0] = mali_FOSCTTM

                #Cross Embedding Metrics
                try:
                    #Ensure its symmetric
                    sym_W = ((1 - mali_class.W.toarray()) + (1 - mali_class.W.toarray()).T) /2

                    emb = self.mds.fit_transform(sym_W)
                    mali_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                    print(f"                CE Score: {mali_CE}")
                except Exception as e:
                    print(f"                Cross Embedding exception occured: {e}")
                    mali_CE = np.NaN
                
                mali_scores[k, 1] = mali_CE
        
            #Save the numpy array
            np.save(filename, mali_scores)

        #Run successful
        return True
    
    def run_CSPUD_tests(self, operations = ["log"]): 
        """Operations should be a tuple of the different operations wanted to run. All are included by default. """

        from mashspud import SPUD

        #We are going to run test with every variation
        print(f"\n-------------------------------------    SPUD Tests " + self.base_directory[52:-1] + "   -------------------------------------\n")
        for operation in operations:
            print(f"Operation {operation}")

            #Create files and store data
            filename, AP_values = self.create_filename("SPUD", Operation = operation, Kind = "merge") 

            #If file aready exists, then we are done :)
            if os.path.exists(filename) or len(AP_values) < 1:
                print(f"        <><><><><>    File {filename} already exists   <><><><><>")
                continue

            #Store the data in a numpy array
            spud_scores = np.zeros((len(self.knn_range), len(AP_values), 2))

            for k, knn in enumerate(self.knn_range):
                print(f"        KNN {knn}")
                for l, anchor_percent in enumerate(AP_values):
                    print(f"            Percent of Anchors {anchor_percent}")

                    try:
                        #Create the class with all the arguments
                        spud_class = SPUD(knn = knn, agg_method = operation)
                        spud_class.fit(self.split_A, self.split_B, known_anchors=self.anchors[:int(len(self.anchors) * anchor_percent)])
                    except Exception as e:
                        print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e} TEST FAILED   <><><><><><>")
                        spud_scores[k, l, 0] = np.NaN
                        spud_scores[k, l, 1] = np.NaN
                        continue

                    #FOSCTTM METRICS
                    try:
                        spud_FOSCTTM = self.FOSCTTM(spud_class.block[:spud_class.len_A, spud_class.len_A:])
                        print(f"                FOSCTTM Score: {spud_FOSCTTM}")
                    except Exception as e:
                        print(f"                FOSCTTM exception occured: {e}")
                        spud_FOSCTTM = np.NaN
                    
                    spud_scores[k, l, 0] = spud_FOSCTTM

                    #Cross Embedding Metrics
                    try:
                        emb = self.mds.fit_transform(spud_class.block)
                        spud_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                        print(f"                CE Score: {spud_CE}")
                    except Exception as e:
                        print(f"                Cross Embedding exception occured: {e}")
                        spud_CE = np.NaN
                    
                    spud_scores[k, l, 1] = spud_CE
            
            #Save the numpy array
            np.save(filename, spud_scores)

        #Run successful
        return True
   
    #We can add t as a parameter, and run tests on that as well, but I feel like the auto is good enough for now
    def run_DIG_tests(self, page_ranks = ["None"], connection_limit = [None], predict = False):  #TODO: Add a predict features evaluation 
        """page_ranks should be whether or not we want to test the page_ranks
        
        predict should be a Boolean value and decide whether we want to test the amputation features. 
        NOTE: This assumes a 1 to 1 correspondance with the variables. ThE MAE doesn't make sense if they aren't the same
        
        t is the percent of values you want covered by that many steps"""

        from mashspud import MASH

        #Run through the tests with every variatioin
        print("\n-------------------------------------   DIG TESTS " + self.base_directory[52:-1] + "   -------------------------------------\n")
        for link in page_ranks:
            print(f"Page rank applied: {link}")

            for t in [-1]:#np.append(np.array(self.knn_range)[[1,3,5,7,9]], -1):
                print(f"    T value {t}")

                #Create the filename
                if t == -1:
                    filename_minus, AP_values = self.create_filename("DIG", PageRanks = link) #T is not included it is assumed to be -1
                else:
                    filename_minus, AP_values = self.create_filename("DIG", PageRanks = link, t = np.round(t/len(self.split_A), decimals=2))

                #set a varaible to Save Mash
                saveMASH = True

                #If file aready exists, then we are done :)
                if os.path.exists(filename_minus) or len(AP_values) < 1:
                    print(f"    <><><><><>    File {filename_minus} already exists for MASH-. Will not save again   <><><><><>")
                    saveMASH = False

                #Loop through the connections
                for connection in connection_limit:
                    print(f"        Connection Limit: {connection}")

                    #Create the filename
                    if t == -1:
                        filename, AP_values = self.create_filename("CwDIG", PageRanks = link, Connection_limit = str(connection)) #T is not included it is assumed to be -1
                    else:
                        filename, AP_values = self.create_filename("CwDIG", PageRanks = link, t = np.round(t/len(self.split_A), decimals=2), Connection_limit = str(connection))

                    save_connections = True
                    #If file aready exists, then we are done :)
                    if os.path.exists(filename) or len(AP_values) < 1:
                        print(f"        <><><><><>    File {filename} already exists   <><><><><>")
                        
                        #set a varaible to Save Mash
                        save_connections = False

                        if not saveMASH:
                            continue

                    #Store the data in a numpy array
                    DIG_scores = np.zeros((len(self.knn_range), len(AP_values), 2 + predict))
                    CwDIG_scores = np.zeros((len(self.knn_range), len(AP_values), 2 + predict))


                    for j, knn in enumerate(self.knn_range):
                        print(f"            KNN {knn}")
                        for k, anchor_percent in enumerate(AP_values):
                            print(f"                Percent of Anchors {anchor_percent}")

                            #Cache this information so it is faster
                            anchor_amount = int((len(self.anchors) * anchor_percent)/2)
                            
                            try:
                                #Create our class to run the tests
                                DIG_class = MASH(t = t, knn = knn, page_rank = link)
                                DIG_class.fit(self.split_A, self.split_B, known_anchors= self.anchors[:anchor_amount])

                            except Exception as e:
                                print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e}  <><><><><><>")
                                DIG_scores[j, k, 0] = np.NaN
                                DIG_scores[j, k, 1] = np.NaN

                                #If we are using predict, this must also be NaN
                                if predict:
                                    DIG_scores[j, k, 2] = np.NaN

                                CwDIG_scores[j, k, 0] = np.NaN
                                CwDIG_scores[j, k, 1] = np.NaN

                                #If we are using predict, this must also be NaN
                                if predict:
                                    CwDIG_scores[j, k, 2] = np.NaN

                                continue

                            if saveMASH:
                                #FOSCTTM Evaluation Metrics
                                try:
                                    DIG_FOSCTTM = np.mean([self.FOSCTTM(DIG_class.int_diff_dist[DIG_class.len_A:, :DIG_class.len_A]), self.FOSCTTM(DIG_class.int_diff_dist[:DIG_class.len_A, DIG_class.len_A:])]) 
                                    print(f"                    FOSCTTM: {DIG_FOSCTTM}")

                                except Exception as e:
                                    print(f"                    FOSCTTM exception occured: {e}")
                                    DIG_FOSCTTM = np.NaN

                                DIG_scores[j, k, 0] = DIG_FOSCTTM

                                #Cross Embedding Evaluation Metric
                                try:
                                    emb = self.mds.fit_transform(DIG_class.int_diff_dist)

                                    DIG_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                                    print(f"                    CE Score: {DIG_CE}")

                                except Exception as e:
                                    print(f"                    Cross Embedding exception occured: {e}")
                                    DIG_CE = np.NaN

                                DIG_scores[j, k, 1] = DIG_CE

                                #Predict features test
                                if predict: #NOTE: This assumes a 1 to 1 correspondance with the variables. ThE MAE doesn't make sense if they aren't the same
                                    #Testing the PREDICT labels features
                                    features_pred_B = DIG_class.predict_feature(predict="B")
                                    features_pred_A = DIG_class.predict_feature(predict="A")

                                    #Get the MAE for each set and average them
                                    DIG_MAE = (abs(self.split_B - features_pred_B).mean() + abs(self.split_A - features_pred_A).mean())/2
                                    DIG_scores[j, k, 2] = DIG_MAE
                                    print(f"                    Predicted MAE {DIG_MAE}") #NOTE: this is all scaled 0-1

                            #<><><><><><><><><><><><><><><><><>     Now Repeat all of that!      <><><><><><><><><><><><><><><><><>
                            #Make the connection limit value
                            if save_connections:
                                try:
                                    #Make the connection limit value
                                    if connection != None:
                                        connection = int(DIG_class.len_A * connection)

                                    #Boost the Algorithm
                                    DIG_class.optimize_by_creating_connections(epochs = 10000, connection_limit = connection, threshold = "auto", hold_out_anchors=self.anchors[anchor_amount:int(anchor_amount*2)])

                                except Exception as e:
                                    print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e}  <><><><><><>")
                                    CwDIG_scores[j, k, 0] = np.NaN
                                    CwDIG_scores[j, k, 1] = np.NaN

                                    #If we are using predict, this must also be NaN
                                    if predict:
                                        CwDIG_scores[j, k, 2] = np.NaN
                                    continue

                                #FOSCTTM Evaluation Metrics
                                try:
                                    DIG_FOSCTTM = np.mean([self.FOSCTTM(DIG_class.int_diff_dist[DIG_class.len_A:, :DIG_class.len_A]), self.FOSCTTM(DIG_class.int_diff_dist[:DIG_class.len_A, DIG_class.len_A:])]) 
                                    print(f"                    FOSCTTM: {DIG_FOSCTTM}")

                                except Exception as e:
                                    print(f"                FOSCTTM exception occured: {e}")
                                    DIG_FOSCTTM = np.NaN

                                CwDIG_scores[j, k, 0] = DIG_FOSCTTM

                                #Cross Embedding Evaluation Metric
                                try:
                                    emb = self.mds.fit_transform(DIG_class.int_diff_dist)

                                    DIG_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                                    print(f"                    CE Score: {DIG_CE}")

                                except Exception as e:
                                    print(f"                    Cross Embedding exception occured: {e}")
                                    DIG_CE = np.NaN

                                CwDIG_scores[j, k, 1] = DIG_CE

                                #Predict features test
                                if predict: #NOTE: This assumes a 1 to 1 correspondance with the variables. ThE MAE doesn't make sense if they aren't the same
                                    #Testing the PREDICT labels features
                                    features_pred_B = DIG_class.predict_feature(predict="B")
                                    features_pred_A = DIG_class.predict_feature(predict="A")

                                    #Get the MAE for each set and average them
                                    DIG_MAE = (abs(self.split_B - features_pred_B).mean() + abs(self.split_A - features_pred_A).mean())/2
                                    CwDIG_scores[j, k, 2] = DIG_MAE
                                    print(f"            Predicted MAE {DIG_MAE}") #NOTE: this is all scaled 0-1


                    #Save the numpy array
                    if saveMASH:
                        np.save(filename_minus, DIG_scores)

                    #Save all connections
                    if save_connections:
                        np.save(filename, CwDIG_scores)

        #Run successful
        return True

    def run_RF_MASH_tests(self, DTM = ["log"]):  #TODO: Add a predict features evaluation 
        """page_ranks should be whether or not we want to test the page_ranks
        
        predict should be a Boolean value and decide whether we want to test the amputation features. 
        NOTE: This assumes a 1 to 1 correspondance with the variables. ThE MAE doesn't make sense if they aren't the same
        
        t is the percent of values you want covered by that many steps"""

        from mashspud import MASH


        #Run through the tests with every variatioin
        print("\n-------------------------------------   MASH TESTS " + self.base_directory[52:-1] + "   -------------------------------------\n")
        for link in DTM:
            print(f"Diffusion to matrix method applied: {link}")

            # for t in [-1]: #np.append(np.array(self.knn_range)[[1,3,5,7,9]], -1): #This takes .... FOREVER
            #     print(f"    T value {t}")

            #Create file directory to store the information
            original_directory = self.base_directory
            self.base_directory = CURR_DIR + "/ManifoldData_RF/" + self.base_directory[len(MANIFOLD_DATA_DIR):]
            

            #Create the filename
            filename, AP_values = self.create_filename("MASH_RF", DTM = link)

            #Reset the directory
            self.base_directory = original_directory 

            #If file aready exists, then we are done :)
            if os.path.exists(filename) or len(AP_values) < 1:
                print(f"    <><><><><>    File {filename} already exists for MASH_RF. Will not save again   <><><><><>")
                continue

            #Loop through the connections
            for i, connection in enumerate(["default"]):
                print(f"        Connection Limit: {connection}")

                #Store the data in a numpy array
                DIG_scores = np.zeros((2, len(self.knn_range), len(AP_values), 2)) #the first two if for the connection limit

                for j, knn in enumerate(self.knn_range):
                    print(f"            KNN {knn}")

                    for k, anchor_percent in enumerate(AP_values):
                        print(f"                Percent of Anchors {anchor_percent}")

                        #Cache this information so it is faster
                        anchor_amount = int((len(self.anchors) * anchor_percent))
                        
                        try:
                            #Create our class to run the tests
                            DIG_class = MASH(t = -1, knn = knn, DTM = link, distance_measure_A = use_rf_proximities, distance_measure_B= use_rf_proximities, n_pca = 100)
                            
                            if connection == "default":
                                DIG_class.fit(dataA = (self.split_A, self.labels), dataB = (self.split_B, self.labels), known_anchors=self.anchors[:anchor_amount])

                            else: #Add the optimization
                                DIG_class.fit(dataA = (self.split_A, self.labels), dataB = (self.split_B, self.labels), known_anchors=self.anchors[:int(anchor_amount/2)])
                                DIG_class.optimize_by_creating_connections(epochs = 10000, connection_limit=None, hold_out_anchors = self.anchors[int(anchor_amount/2):anchor_amount])

                        except Exception as e:
                            print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e}  <><><><><><>")
                            DIG_scores[i, j, k, 0] = np.NaN
                            DIG_scores[i, j, k, 1] = np.NaN


                        #FOSCTTM Evaluation Metrics
                        try:
                            DIG_FOSCTTM = np.mean([self.FOSCTTM(DIG_class.int_diff_dist[DIG_class.len_A:, :DIG_class.len_A]), self.FOSCTTM(DIG_class.int_diff_dist[:DIG_class.len_A, DIG_class.len_A:])]) 
                            print(f"                    FOSCTTM: {DIG_FOSCTTM}")

                        except Exception as e:
                            print(f"                    FOSCTTM exception occured: {e}")
                            DIG_FOSCTTM = np.NaN

                        DIG_scores[i, j, k, 0] = DIG_FOSCTTM

                        #Cross Embedding Evaluation Metric
                        try:
                            emb = self.mds.fit_transform(DIG_class.int_diff_dist)

                            DIG_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                            print(f"                    CE Score: {DIG_CE}")

                        except Exception as e:
                            print(f"                    Cross Embedding exception occured: {e}")
                            DIG_CE = np.NaN

                        DIG_scores[i, j, k, 1] = DIG_CE


            np.save(filename, DIG_scores)

        #Run successful
        return True
 
    def run_NAMA_tests(self):
        """Needs no additional parameters"""

        #Create file name
        filename, AP_values = self.create_filename("NAMA")

        #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True

        #Store the results in an array
        NAMA_scores = np.zeros((len(AP_values), 2))

        #Create the Nama object on the dataset
        nama = NAMA(ot_reg = 0.001)

        print("\n-------------------------------------   NAMA TESTS  " + self.base_directory[52:-1] + "  -------------------------------------\n")
        
        for i, anchor_percent in enumerate(AP_values):
            print(f"Percent of Anchors {anchor_percent}")

            #Fit NAMA
            try:
                nama.fit(self.anchors[:int(len(self.anchors)*anchor_percent)], self.split_A, self.split_B)
            except Exception as e:
                print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e}  <><><><><><>")
                NAMA_scores[i, 0] = np.NaN
                NAMA_scores[i, 1] = np.NaN
                continue

            #Test FOSCTTM
            try:
                nama_FOSCTTM = self.FOSCTTM(nama.cross_domain_dists)
                print(f"    FOSCTTM: {nama_FOSCTTM}")
            except Exception as e:
                print(f"    FOSCTTM exception occured: {e}")
                nama_FOSCTTM = np.NaN
            NAMA_scores[i, 0] = nama_FOSCTTM

            #Get embedding for CE
            try:
                emb = self.mds.fit_transform(nama.block)
                nama_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                print(f"    Cross Embedding: {nama_CE}")
            except Exception as e:
                print(f"    Cross Embedding exception occured: {e}")
                nama_CE = np.NaN
            NAMA_scores[i, 1] = nama_CE

        #Save the numpy array
        np.save(filename, NAMA_scores)

        #Run successful
        return True

    def run_DTA_tests(self):
        """Needs no additional parameters"""
        from AlignmentMethods.DTA_andres import DTA


        #Create file name
        filename, AP_values = self.create_filename("DTA")

        #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Store the results in an array
        DTA_scores = np.zeros((len(self.knn_range), len(AP_values), 2))

        print("\n--------------------------------------   DTA TESTS " + self.base_directory[52:-1] + "   --------------------------------------\n")

        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")

            #Initialize the class with the correct KNN
            DTA_class = DTA(knn = knn, entR=0.001, verbose = 0)

            #Loop through each anchor. 
            for j, anchor_percent in enumerate(AP_values):
                print(f"    Percent of Anchors {anchor_percent}")

                #In case the class initialization fails
                try:
                    #Reformat the anchors 
                    sharedD1 = self.split_A[self.anchors[:int(len(self.anchors)*anchor_percent)].T[0]] 
                    sharedD2 = self.split_B[self.anchors[:int(len(self.anchors)*anchor_percent)].T[1]]
                    labelsh1 = self.labels[self.anchors[:int(len(self.anchors)*anchor_percent)].T[0]] #NOTE: Can use these if we want to compare labels
                    #labelsh2 = self.labels[self.anchors.T[1]]
                    labels_extended = np.concatenate((self.labels, labelsh1))

                    #Fit it
                    DTA_class.fit(self.split_A, self.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2)
                except Exception as e:
                    print(f"<><><><><><>   UNABLE TO CREATE CLASS BECUASE {e}   <><><><><><>")
                    DTA_scores[i, j, 0] = np.NaN
                    DTA_scores[i, j, 1] = np.NaN
                    continue

                #FOSCTTM scores
                try:
                    DTA_FOSCTTM = self.FOSCTTM(1 - self.normalize_0_to_1(DTA_class.W12)) #Off Diagonal Block. NOTE: it has to be normalized because it returns values 0-2. We subtract one because it is in similarities
                    print(f"        FOSCTTM {DTA_FOSCTTM}")
                except Exception as e:
                    print(f"        FOSCTTM exception occured: {e}")
                    DTA_FOSCTTM = np.NaN
                DTA_scores[i, j, 0] = DTA_FOSCTTM

                #Cross Embedding Scores
                try:
                    emb = self.mds.fit_transform(1 - self.normalize_0_to_1(DTA_class.W))
                    DTA_CE = self.cross_embedding_knn(emb, (labels_extended, labels_extended), knn_args = {'n_neighbors': 4}) #NOTE: This has a slight advantage because the anchors are counted twice
                    print(f"        Cross Embedding: {DTA_CE}")
                except Exception as e:
                    print(f"        Cross Embedding exception occured: {e}")
                    DTA_CE = np.NaN
                DTA_scores[i, j, 1] = DTA_CE

        #Save the numpy array
        np.save(filename, DTA_scores)

        #Run successful
        return True
    
    def run_PCR_tests(self):
        """Procrutees Manifold Alignment
        
        Needs no additional parameters"""

        #Create file name
        filename, AP_values = self.create_filename("PCR")

        #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Store the results in an array
        scores = np.zeros((len(self.knn_range), len(AP_values), 2))

        print("\n--------------------------------------   PCR TESTS " + self.base_directory[52:-1] + "   --------------------------------------\n")

        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")

            #Initialize the class with the correct KNN
            from AlignmentMethods.ma_procrustes import MAprocr
            PCR_class = MAprocr(knn = knn, random_state = self.random_state, n_jobs = 1)

            #Loop through each anchor. 
            for j, anchor_percent in enumerate(AP_values):
                print(f"    Percent of Anchors {anchor_percent}")

                #In case the class initialization fails
                try:
                    #Get anchor count
                    anchor_count = int(len(self.anchors)*anchor_percent)

                    #Reformat the anchors 
                    sharedD1 = self.split_A[self.anchors[:anchor_count].T[0]] 
                    sharedD2 = self.split_B[self.anchors[:anchor_count].T[1]]
                    labelsh1 = self.labels[self.anchors[:anchor_count].T[0]] #NOTE: Can use these if we want to compare labels
                    #labelsh2 = self.labels[self.anchors.T[1]]
                    labels_extended = np.concatenate((self.labels, labelsh1))

                    #Fit it
                    PCR_class.fit(self.split_A, self.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2)
                except Exception as e:
                    print(f"<><><><><><>   UNABLE TO CREATE CLASS BECUASE {e}   <><><><><><>")
                    scores[i, j, 0] = np.NaN
                    scores[i, j, 1] = np.NaN
                    continue

                #FOSCTTM scores TODO: AVG the different FOCSTTMS
                try:
                    len_A = len(self.split_A) + anchor_count
                    FOSCTTM = np.mean([self.FOSCTTM(1 - PCR_class.W[len_A:, :len_A]), 
                                       self.FOSCTTM(1 - PCR_class.W[:len_A, len_A:])])
                    
                    print(f"        FOSCTTM {FOSCTTM}")
                except Exception as e:
                    print(f"        FOSCTTM exception occured: {e}")
                    FOSCTTM = np.NaN
                scores[i, j, 0] = FOSCTTM

                #Cross Embedding Scores
                try:
                    emb = self.mds.fit_transform(1 - PCR_class.W)
                    CE = self.cross_embedding_knn(emb, (labels_extended, labels_extended), knn_args = {'n_neighbors': 4}) #NOTE: This has a slight advantage because the anchors are counted twice
                    print(f"        Cross Embedding: {CE}")
                except Exception as e:
                    print(f"        Cross Embedding exception occured: {e}")
                    CE = np.NaN
                scores[i, j, 1] = CE

        #Save the numpy array
        np.save(filename, scores)

        #Run successful
        return True

    def run_SSMA_tests(self):
        """ No Additional arguments needed"""
        from AlignmentMethods.ssma import ssma

        #Add the last index for knn range so we know what knn values were used
        filename, AP_values = self.create_filename("SSMA")

        #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Create an array to store the important data in 
        SSMA_scores = np.zeros((len(self.knn_range), len(AP_values), 2))

        print("\n--------------------------------------   SSMA TESTS " + self.base_directory[52:-1] + "   --------------------------------------\n")
        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")
            #Initialize the class with the correct KNN
            SSMA_class = ssma(knn = knn, verbose = 0, r = 2) #R can also be = to this: (self.split_A.shape[1] + self.split_B.shape[1])

            #Loop through each anchor. 
            for j, anchor_percent in enumerate(AP_values):
                print(f"    Percent of Anchors {anchor_percent}")

                #Test to see if class initialization fails
                try:
                    #Reformat the anchors 
                    sharedD1 = self.split_A[self.anchors[:int(len(self.anchors)*anchor_percent)].T[0]] 
                    sharedD2 = self.split_B[self.anchors[:int(len(self.anchors)*anchor_percent)].T[1]]
                    labelsh1 = self.labels[self.anchors[:int(len(self.anchors)*anchor_percent)].T[0]] #NOTE: Can use these if we want to compare labels
                    #labelsh2 = self.labels[self.anchors.T[1]]
                    labels_extended = np.concatenate((self.labels, labelsh1))

                    #Fit it
                    SSMA_class.fit(self.split_A, self.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2)
                except Exception as e:
                    print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e}   <><><><><><>")
                    SSMA_scores[i, j, 0] = np.NaN
                    SSMA_scores[i, j, 1] = np.NaN
                    continue

                #FOSCTTM scores
                try:
                    SSMA_FOSCTTM = self.FOSCTTM(1 - SSMA_class.W[len(SSMA_class.domain1):, :len(SSMA_class.domain1)]) #Off Diagonal Block. NOTE: it has to be normalized because it returns values 0-2. We subtract one because it is in similarities
                    print(f"        FOSCTTM {SSMA_FOSCTTM}")
                except Exception as e:
                    print(f"        FOSCTTM exception occured: {e}")
                    SSMA_FOSCTTM = np.NaN
                SSMA_scores[i, j, 0] = SSMA_FOSCTTM

                #Cross Embedding Scores
                try:
                    emb = self.mds.fit_transform(1 - SSMA_class.W)
                    SSMA_CE = self.cross_embedding_knn(emb, (labels_extended, labels_extended), knn_args = {'n_neighbors': 4}) #NOTE: This has a slight advantage because the anchors are counted twice
                    print(f"        Cross Embedding: {SSMA_CE}")
                except Exception as e:
                    print(f"        Cross Embedding exception occured: {e}")
                    SSMA_CE = np.NaN
                SSMA_scores[i, j, 1] = SSMA_CE

        #Save the numpy array
        np.save(filename, SSMA_scores)

        #Run successful
        return True

    def run_MAGAN_tests(self):
        """Needs no additional parameters"""
        import AlignmentMethods.MAGAN as MAGAN

        #Create file name
        filename, AP_values = self.create_filename("MAGAN")

        #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True

        #Store the results in an array
        MAGAN_scores = np.zeros((len(AP_values), 2))

        print("\n-------------------------------------   MAGAN TESTS  " + self.base_directory[52:-1] + "  -------------------------------------\n")

        #Loop through each anchor. 
        for j, anchor_percent in enumerate(AP_values):
            print(f"    Percent of Anchors {anchor_percent}")

            #Run Magan and tests
            domain_a, domain_b, domain_ab, domain_ba = MAGAN.run_MAGAN(self.split_A, self.split_B, self.anchors[:int(len(self.anchors)*anchor_percent)])

            #Reshape the domains and then create the block
            domain_a, domain_b = MAGAN.get_pure_distance(domain_a, domain_b)
            domain_ab, domain_ba = MAGAN.get_pure_distance(domain_ab, domain_ba)
            MAGAN_block = np.block([[domain_a, domain_ba],
                                    [domain_ba, domain_b]])
            
            #Get FOSCTTM SCORES
            try:
                MAGAN_FOSCTTM = np.mean((self.FOSCTTM(domain_ab), self.FOSCTTM(domain_ba))) #NOTE: Do we choose the best one? Currently chose to average them instead
                print(f"FOSCTTM: {MAGAN_FOSCTTM}")
            except Exception as e:
                print(f"FOSCTTM exception occured: {e}")
                MAGAN_FOSCTTM = np.NaN
            MAGAN_scores[j, 0] = MAGAN_FOSCTTM
            
            #Get embedding for CE
            try:
                emb = self.mds.fit_transform(MAGAN_block)
                MAGAN_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                print(f"Cross Embedding: {MAGAN_CE}")
            except Exception as e:
                print(f"Cross Embedding exception occured: {e}")
                MAGAN_CE = np.NaN
            MAGAN_scores[j, 1] = MAGAN_CE

        #Save the numpy array
        np.save(filename, MAGAN_scores)

        #Run successful
        return True

    def run_JLMA_tests(self):
        """Needs no additional parameters"""
        from AlignmentMethods.jlma import JLMA

        #Create file name
        filename, AP_values = self.create_filename("JLMA")

        #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Store the results in an array
        scores = np.zeros((len(self.knn_range), len(AP_values), 2))

        print("\n--------------------------------------   JLMA TESTS " + self.base_directory[52:-1] + "   --------------------------------------\n")

        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")

            #Initialize the class with the correct KNN, and d is the same way we calculate dimensions for MDS
            JLMA_class = JLMA(k = knn, d = min(len(self.split_B[1]), len(self.split_A[1])))

            #Loop through each anchor. 
            for j, anchor_percent in enumerate(AP_values):
                print(f"    Percent of Anchors {anchor_percent}")

                #In case the class initialization fails
                try:
                    #Fit it
                    JLMA_class.fit(self.split_A, self.split_B, self.anchors[:int(len(self.anchors)*anchor_percent)])

                except Exception as e:
                    print(f"<><><><><><>   UNABLE TO CREATE CLASS BECUASE {e}   <><><><><><>")
                    scores[i, j, 0] = np.NaN
                    scores[i, j, 1] = np.NaN
                    continue

                #FOSCTTM scores
                try:
                    #Prep the block
                    block = JLMA_class.SquareDist(JLMA_class.Y)
                    len_A = len(self.split_A)

                    #Calculate FOSCTTM by averaging the two domains
                    FOSCTTM = self.FOSCTTM(block[:len_A, len_A:])
                    print(f"        FOSCTTM {FOSCTTM}")
                except Exception as e:
                    print(f"        FOSCTTM exception occured: {e}")
                    FOSCTTM = np.NaN
                scores[i, j, 0] = FOSCTTM

                #Cross Embedding Scores
                try:
                    CE = self.cross_embedding_knn(JLMA_class.Y, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                    print(f"        Cross Embedding: {CE}")
                except Exception as e:
                    print(f"        Cross Embedding exception occured: {e}")
                    CE = np.NaN
                scores[i, j, 1] = CE

        #Save the numpy array
        np.save(filename, scores)

        #Run successful
        return True

    def run_KNN_tests(self):
        """Needs no additional paramenters.
        
        Gets the baseline classification without doing any alignment for each data set"""

        #Create file name
        filename, AP_values = self.create_filename("KNN_BL")
        from sklearn.decomposition import PCA

        #Prepare Baseline Data
        pca = PCA(n_components=min(len(self.split_A[1]), len(self.split_B[1])))
        A_emb = pca.fit_transform(self.split_A)
        B_emb = pca.fit_transform(self.split_B)

        #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Create an array to store the important data in 
        scores = np.zeros((len(self.knn_range), 2)) #Now both of these are classification scores -- one for Split A and one for Split B

        print("\n--------------------------------------   Base Line Tests " + self.base_directory[53:-1] + "   --------------------------------------\n")
        #Repeat through each knn value
        from sklearn.model_selection import train_test_split

        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")

            #Initilize model
            # Determine if the task is classification or regression
            if np.issubdtype(self.labels.dtype, np.integer):
                model = KNeighborsClassifier(n_neighbors = knn)
            else:
                model = KNeighborsRegressor(n_neighbors = knn)

            #Split data and train for split A
            try:
                # Split the data into training and testing sets
                A_emb_train, A_emb_test, labels_train, labels_test = train_test_split(A_emb, self.labels, test_size=0.2, random_state=42)

                # Train the model on the training set
                model.fit(A_emb_train, labels_train)

                # Evaluate the model on the testing set
                scores[i, 0] = model.score(A_emb_test, labels_test)
                print(f"    Classification Score trained on A {scores[i, 0]}")
            except:
                scores[i, 0] = np.NaN
                print(f"    Classification Score trained on A Failed")

            #Split data and train for split B
            try:
                # Split the data into training and testing sets
                B_emb_train, B_emb_test, labels_train, labels_test = train_test_split(B_emb, self.labels, test_size=0.2, random_state=42)

                # Train the model on the training set
                model.fit(B_emb_train, labels_train)

                # Evaluate the model on the testing set
                scores[i, 1] = model.score(B_emb_test, labels_test)
                print(f"    Classification Score trained on b {scores[i, 1]}")
            except:
                scores[i, 1] = np.NaN
                print(f"    Classification Score trained on B Failed")

        #Save the numpy array
        np.save(filename, scores)

        self.run_BL_CE()

        #Run successful
        return True
    

    def run_BL_CE(self):
        """Needs no additional paramenters.
        
        Gets the baseline classification by training on one domain to predict on other domain"""

        #Create file name
        filename, AP_values = self.create_filename("BL_CE")
        from sklearn.decomposition import PCA

        #Prepare Baseline Data
        pca = PCA(n_components=min(len(self.split_A[1]), len(self.split_B[1])))
        A_emb = pca.fit_transform(self.split_A)
        B_emb = pca.fit_transform(self.split_B)

        #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Create an array to store the important data in 
        scores = np.zeros((len(self.knn_range), 2)) #Now both of these are classification scores -- one for Split A and one for Split B

        print("\n--------------------------------------   Base Line Tests " + self.base_directory[53:-1] + "   --------------------------------------\n")
        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")

            #Initilize model
            # Determine if the task is classification or regression
            if np.issubdtype(self.labels.dtype, np.integer):
                model = KNeighborsClassifier(n_neighbors = knn)
            else:
                model = KNeighborsRegressor(n_neighbors = knn)

            #Split data and train for split A
            try:
                model.fit(A_emb, self.labels)
                scores[i, 0] = model.score(B_emb, self.labels)
                print(f"    Classification Score trained on A {scores[i, 0]}")
            except:
                scores[i, 0] = np.NaN
                print(f"    Classification Score trained on A Failed")

            #Split data and train for split B
            try:
                model.fit(B_emb, self.labels)
                scores[i, 1] = model.score(A_emb, self.labels)
                print(f"    Classification Score trained on B {scores[i, 1]}")
            except:
                scores[1, 1] = np.NaN
                print(f"    Classification Score trained on B Failed")

        #Save the numpy array
        np.save(filename, scores)

        #Run successful
        return True


    def run_RF_BL_tests(self):
        """Needs no additional paramenters.
        
        Gets the baseline classification without doing any alignment for each data set"""

        #Create file directory to store the information
        original_directory = self.base_directory
        self.base_directory = CURR_DIR + "/ManifoldData_RF/" + self.base_directory[len(MANIFOLD_DATA_DIR):]  

        #Create file name
        filename, AP_values = self.create_filename("RF_BL")

        #Reset the directory
        self.base_directory = original_directory 

        #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Create an array to store the important data in 
        scores = np.zeros(2) #Now both of these are classification scores -- one for Split A and one for Split B

        print("\n--------------------------------------   RF Gap Baseline Tests " + self.base_directory[53:-1] + "   --------------------------------------\n")
        
        
        #Initilize Class
        rf_class = RFGAP(prediction_type="classification", y=self.labels, prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=True, oob_score = True)

        #Fit it for Data A and get proximities
        rf_class.fit(self.split_A, y = self.labels)

        #GET THE SCORE
        scores[0] = rf_class.oob_score_

        #Fit it for Data B and get proximities
        rf_class.fit(self.split_B, y = self.labels)
        
        #GET THE SCORE
        scores[1] = rf_class.oob_score_

        print(f"SCORE A: {scores[0]}")
        print(f"SCORE B: {scores[1]}")

        #Save the numpy array
        np.save(filename, scores)

        #Run successful
        return True

"""
------------------------------------------------------------------------------------------------------------------------------
                                                    OUTSIDE OF THE CLASS
------------------------------------------------------------------------------------------------------------------------------
"""

""" HELPER FUNCTIONS"""
def find_words_order(text, words):
    """Text should be the string in which words appear. 
    
    This is a helper function for us to use to understand the values of the numpy arrays"""

    # Create a list to hold the positions and words
    positions = []

    # Check for each word in the list
    for word in words:
        pos = text.find(word)
        if pos != -1:
            positions.append((pos, word))

    # Sort the list by positions
    positions.sort()

    # Return only the words, sorted by their appearance
    return [word for _, word in positions]

# Function to check if a string contains any of the substrings
def contains_any_substring(s, substrings):
    for sub in substrings:
        if sub in s:
            return True
    return False

def clear_directory(text_curater = "all", not_text = None, directory = "default"):
    """CAREFUL. THIS WIPES THE MANIFOLD DATA DIRECTORY CLEAN"""

    #Use all of our files
    file_names = ["artificial_tree", "audiology", "balance_scale", "breast_cancer", "Cancer_Data", "car", "chess", 
                "crx", "diabetes", "ecoli_5", "flare1", "glass", "heart_disease", "heart_failure", "hepatitis",
                "hill_valley", "ionosphere", "iris", "Medicaldataset", "mnist_test", "optdigits", "parkinsons",
                "seeds", "segmentation", "tic-tac-toe", "titanic", "treeData", "water_potability", "waveform",
                "blobs", "S-curve",
                "winequality-red", "zoo", "AirfoilSelfNoise",  "AutoMPG",
                "ComputerHardware",  "ConcreteSlumpTest",  "FacebookMetrics",
                "IstanbulStock",   "Parkinsons",
                "Automobile",       "CommunityCrime",
                "ConcreteCompressiveStrength",  "EnergyEfficiency",   "Hydrodynamics",
                "OpticalNetwork",  "SML2010"
                ]

    #Modify the file names to become directory names
    if directory == "default":
        directories = [MANIFOLD_DATA_DIR + file_name for file_name in file_names]
    elif directory.lower() == "regression":
        directories = [CURR_DIR + "/RegressionData/" + file_name for file_name in file_names]
    elif directory.lower() == "mantel":
        directories = [CURR_DIR + "/Mantel"]
    else:
        directories = [CURR_DIR + "/ManifoldData_RF/" + file_name for file_name in file_names]

    #Loop through each directory and file and get the file paths
    files = []
    for directory in directories:
        if os.path.isdir(directory): #Check to make sure directory exists
            files += [os.path.join(directory, file) for file in os.listdir(directory)]

    if text_curater != "all":
        selected_files = []
        selected_files += [file for file in files if text_curater in file]
    else:
        selected_files = files

    #Filter out with text in it
    if not_text == None:
        curated_files = selected_files
    
    #if its a list we need to repeat the thing
    elif type(not_text) == list:
        curated_files = []
        curated_files = [file for file in selected_files if not contains_any_substring(file, not_text)]
    
    else:
        curated_files = []
        curated_files = [file for file in selected_files if not_text not in file]
        

    #Add user confirmation
    print(f"Preparing to delete {len(curated_files)} files")
    print(f"First 10 file names to be deleted\n-------------------------------------------------\n{curated_files[:10]}")
    proceed = input("Proceed? [y, n]")

    if proceed == "y":
        #Finally delete all files
        for file in curated_files:
            os.remove(file)

        print("Files Deleted.")
        return True
    
    else: 
        print("<><> Cancelling Process <><>")
        return False
    
def is_even(num):
    return num & 1 == 0

def _upload_file(file, directory = "default"):
    #Simply for error finding
    if directory == "default":
        original_file = MANIFOLD_DATA_DIR + file
    elif directory == "regression":
        original_file = CURR_DIR + "/RegressionData/" + file  
    else:
        original_file = CURR_DIR + "/ManifoldData_RF/" + file
 

    #Create DataFrame
    df = pd.DataFrame(columns= ["csv_file", "method", "seed", "split", "KNN",
                                "Percent_of_KNN", "Percent_of_Anchors", 
                                "Page_Rank", "t_value", "Predicted_Feature_MAE",
                                "Operation", "algorithm", 
                                "FOSCTTM", "Cross_Embedding_KNN"])
    
    #Create Base Line Data Frame
    base_df = pd.DataFrame(columns= ["csv_file", "method", "seed", "split", "KNN", #Shared headers
                                    "A_Classification_Score", "B_Classification_Score"])
    
    if file[-4:] == "json":
        return (df, base_df)

    #Load in the numpy array
    try:
        data = np.load(original_file, allow_pickle=True) #allow_pickle=True
    except Exception as e:
        print(f"-------------------------------------------------------------------------------------------------------\nUnable to load {file}. \nError Caught: {e} \nContinuing without uploading file\n-------------------------------------------------------------------------------------------------------")
        
        #It will be empty, but helps future code
        return (df, base_df)

    #Create a dictionary to use to add rows to our DataFame
    data_dict = {}

    #Drop the .npy
    file = file[:-4]

    #Get the KNN increment, and then shrink the file to right size
    knn_upper_bound = file.split("_")[-1]
    knn_increment = int(knn_upper_bound) // 9
    file = file[:-(len(knn_upper_bound)+2)]

    #Get the name of the csv_file and then cut the csv file out of the name
    name_index = file.find('/')
    data_dict["csv_file"] = file[:name_index]
    file = file[name_index + 1:]

    #Correct csv_files
    if data_dict["csv_file"] == "b":
        data_dict["csv_file"] = "blobs"
    elif data_dict["csv_file"] == "S-c":
        data_dict["csv_file"] = "S-curve"

    #Get the method of the file, then cut the method out of the name
    method_index = file.find('(')
    data_dict["method"] = file[:method_index]
    file = file[method_index + 1:]

    #Get the seed, and then add its name back
    if file[0] == 'r':
        data_dict["split"] = "random"
    elif file[0] == 's':
        data_dict["split"] = 'skewed'
    elif file[0] == 'd':
        data_dict["split"] = "distort"
    elif file[0] == 't':
        data_dict["split"] = "turn"
    else: #File split was even
        data_dict["split"] = "even"

    #Get the seed, and then cut it out of the filename (and the split letter too)
    seed_index = file.find(')')
    data_dict["seed"] = file[1:seed_index]
    file = file[seed_index+1:]

    #Get the Anchor Percents
    AP_index = file.find('_AP(')
    AP_values = file[AP_index + 4:].split('-')
    
    #Overarching error catching system
    try:

        #Correct the bad naming for MALI
        if data_dict["method"] == "MALI":
            data_dict["method"] = "MALI_RF"
            data_dict["Operation"] = "RF_GAP"
        elif data_dict["method"] == "MALI_RF":
            data_dict["method"] = "MALI"
            data_dict["Operation"] = "default"

        if data_dict["method"] == "MASH_RF":

            #Set Page Rank to None
            data_dict["Page_Rank"] = "None"

            #Add the DTM if applicable:
            if "hellinger" in file:
                data_dict["algorithm"] = "hellinger"
            elif "kl" in file:
                data_dict["algorithm"] = "kl"
            elif "log" in file:
                data_dict["algorithm"] = "log"
            else:
                data_dict["algorithm"] = "None"

            #Add the right t value
            if "_t(" in file:
                t_index = file.find("_t(")
                even = is_even(int(file[t_index+3 : t_index + file[t_index:].find(")")][-1]))

                if even:
                    data_dict["t_value"] = float(file[t_index+3 : t_index + file[t_index:].find(")")])
                else:
                    data_dict["t_value"] = np.round(float(file[t_index+3 : t_index + file[t_index:].find(")")]) + 0.01, decimals=2)
            else:
                data_dict["t_value"] = -1.0


            #Loop through both connections
            for i in range(2):
                if i == 0:
                    data_dict["Operation"] = "Not Optimized"
                else:
                    data_dict["Operation"] = "Optimized"

                #Loop through each Knn
                for j in range(0, 10):
                    knn = (j*knn_increment) + 2
                    data_dict["KNN"] = knn

                    #These percents are rough, and not exact. This is so we can have similar estimates to compare
                    data_dict["Percent_of_KNN"] = (j * 0.02) + 0.01

                    #Loop through each Anchor percentage
                    for k in range(len(AP_values)):
                        data_dict["Percent_of_Anchors"] = AP_values[k]

                        #Now use are data array to grab the FOSCTTM and CE scores
                        data_dict["FOSCTTM"] = data[i, j, k, 0]
                        data_dict["Cross_Embedding_KNN"] = data[i, j, k, 1]

                        df = df._append(data_dict, ignore_index=True)

        #Split based on method
        elif data_dict["method"] == "DIG":

            #Add the right Page Rank Argument
            if "full" in file:
                data_dict["Page_Rank"] = "full"
            elif "off-diagonal" in file:
                data_dict["Page_Rank"] = "off-diagonal"
            else: #Then it is full
                data_dict["Page_Rank"] = "None"

            #Add the DTM if applicable:
            if "hellinger" in file:
                data_dict["algorithm"] = "hellinger"
            elif "kl" in file:
                data_dict["algorithm"] = "kl"
            elif "log" in file:
                data_dict["algorithm"] = "log"
            else:
                data_dict["algorithm"] = "None"

            #Add the right t value
            if "_t(" in file:
                t_index = file.find("_t(")
                even = is_even(int(file[t_index+3 : t_index + file[t_index:].find(")")][-1]))

                if even:
                    data_dict["t_value"] = float(file[t_index+3 : t_index + file[t_index:].find(")")])
                else:
                    data_dict["t_value"] = np.round(float(file[t_index+3 : t_index + file[t_index:].find(")")]) + 0.01, decimals=2)
            else:
                data_dict["t_value"] = -1.0

            #Loop through each Knn
            for j in range(0, 10):
                knn = (j*knn_increment) + 2
                data_dict["KNN"] = knn

                #These percents are rough, and not exact. This is so we can have similar estimates to compare
                data_dict["Percent_of_KNN"] = (j * 0.02) + 0.01

                #Loop through each Anchor percentage
                for k in range(len(AP_values)):
                    data_dict["Percent_of_Anchors"] = AP_values[k]

                    #Now use are data array to grab the FOSCTTM and CE scores
                    data_dict["FOSCTTM"] = data[j, k, 0]
                    data_dict["Cross_Embedding_KNN"] = data[j, k, 1]

                    #We failsafe this in a try because there might not be a finally loop
                    try:
                        data_dict["Predicted_Feature_MAE"] = data[j, k, 2]
                    except:
                        data_dict["Predicted_Feature_MAE"] = np.NaN
                    finally:
                        #Create a new Data frame instance with all the asociated values
                        df = df._append(data_dict, ignore_index=True)

        #Split based on method
        elif data_dict["method"] == "CwDIG":

            #Add the right Page Rank Argument
            if "full" in file:
                data_dict["Page_Rank"] = "full"
            elif "off-diagonal" in file:
                data_dict["Page_Rank"] = "off-diagonal"
            else: #Then it is None
                data_dict["Page_Rank"] = "None"

            #Add the DTM if applicable:
            if "hellinger" in file:
                data_dict["Operation"] = "hellinger"
            elif "kl" in file:
                data_dict["Operation"] = "kl"
            elif "log" in file:
                data_dict["Operation"] = "log"
            else:
                data_dict["Operation"] = "None"

            #Add the right t value
            if "_t(" in file:
                t_index = file.find("_t(")
                even = is_even(int(file[t_index+3 : t_index + file[t_index:].find(")")][-1]))

                if even:
                    data_dict["t_value"] = float(file[t_index+3 : t_index + file[t_index:].find(")")])
                else:
                    data_dict["t_value"] = np.round(float(file[t_index+3 : t_index + file[t_index:].find(")")]) + 0.01, decimals=2)
            else:
                data_dict["t_value"] = -1.0

            #Get the Connection value
            con_index = file.find('_Con(')
            data_dict["Operation"] = str(file[con_index + 5:AP_index-1])

            #Loop through each Knn
            for j in range(0, 10):
                knn = (j*knn_increment) + 2
                data_dict["KNN"] = knn

                #These percents are rough, and not exact. This is so we can have similar estimates to compare
                data_dict["Percent_of_KNN"] = (j * 0.02) + 0.01

                #Loop through each Anchor percentage
                for k in range(len(AP_values)):
                    data_dict["Percent_of_Anchors"] = AP_values[k]

                    #Now use are data array to grab the FOSCTTM and CE scores
                    data_dict["FOSCTTM"] = data[j, k, 0]
                    data_dict["Cross_Embedding_KNN"] = data[j, k, 1]

                    #We failsafe this in a try because there might not be a finally loop
                    try:
                        data_dict["Predicted_Feature_MAE"] = data[j, k, 2]
                    except:
                        data_dict["Predicted_Feature_MAE"] = np.NaN
                    finally:
                        #Create a new Data frame instance with all the asociated values
                        df = df._append(data_dict, ignore_index=True)
                    
        #Method SPUD
        elif data_dict["method"] == "SPUD":

            #Assign the operation
            if "abs" in file:
                data_dict["Operation"] = "abs"
            elif "sqrt" in file:
                data_dict["Operation"] = "sqrt"
            elif "average" in file:
                data_dict["Operation"] = "average"
            elif "log" in file:
                data_dict["Operation"] = "log"
            else: 
                data_dict["Operation"] = "normalize"

            #Assign its Kind
            if "distance" in file:
                data_dict["algorithm"] = "distance"
            elif "merge" in file:
                data_dict["algorithm"] = "merge"
            elif "pure" in file:
                data_dict["algorithm"] = "pure"
            else:
                data_dict["algorithm"] = "similarity"
            
            #Loop through each Knn
            for k in range(0, 10):
                knn = (k*knn_increment) + 2
                data_dict["KNN"] = knn

                #These percents are rough, and not exact. This is so we can have similar estimates to compare
                data_dict["Percent_of_KNN"] = (k * 0.02) + 0.01

                #Loop through each Anchor percentage
                for l in range(len(AP_values)):
                    data_dict["Percent_of_Anchors"] = AP_values[l]

                    #Now use are data array to grab the FOSCTTM and CE scores
                    data_dict["FOSCTTM"] = data[k, l, 0]
                    data_dict["Cross_Embedding_KNN"] = data[k, l, 1]

                    #Create a new Data frame instance with all the asociated values
                    df = df._append(data_dict, ignore_index=True)

        #Method SPUD
        elif data_dict["method"] == "SPUD_RF":

            #Assign the operation
            if "None" in file:
                data_dict["Operation"] = "None"
            elif "sqrt" in file:
                data_dict["Operation"] = "sqrt"
            elif "log" in file:
                data_dict["Operation"] = "log"
            else: 
                data_dict["Operation"] = "float" #We can update this later to tell us which float

            #Assign its Kind
            if "abs" in file:
                data_dict["algorithm"] = "abs"
            elif "mean" in file:
                data_dict["algorithm"] = "mean"
            else:
                data_dict["algorithm"] = "default"
            
            #Loop through each Knn
            for k in range(0, 10):

                if k == 9 and data_dict["algorithm"] != "default":
                    data_dict["KNN"] = "NAMA"
                else:
                    knn = (k*knn_increment) + 2
                    data_dict["KNN"] = knn

                #These percents are rough, and not exact. This is so we can have similar estimates to compare
                data_dict["Percent_of_KNN"] = (k * 0.02) + 0.01

                #Loop through each Anchor percentage
                for l in range(len(AP_values)):
                    data_dict["Percent_of_Anchors"] = AP_values[l]

                    #Now use are data array to grab the FOSCTTM and CE scores
                    data_dict["FOSCTTM"] = data[k, l, 0]
                    data_dict["Cross_Embedding_KNN"] = data[k, l, 1]

                    #Create a new Data frame instance with all the asociated values
                    df = df._append(data_dict, ignore_index=True)

        #METHOD MALI
        elif data_dict["method"] == "MALI_RF" or data_dict["method"] == "KEMA_RF" or data_dict["method"] == "MALI":

            if "lin" in file:
                data_dict["Operation"] = "lin"
            elif "rbf" in file:
                data_dict["Operation"] = "rbf"

            #Loop through each Knn
            for k in range(0, 10):
                knn = (k*knn_increment) + 2
                data_dict["KNN"] = knn

                #These percents are rough, and not exact. This is so we can have similar estimates to compare
                data_dict["Percent_of_KNN"] = (k * 0.02) + 0.01
            
                #Now use are data array to grab the FOSCTTM and CE scores
                data_dict["FOSCTTM"] = data[k, 0]
                data_dict["Cross_Embedding_KNN"] = data[k, 1]

                #Create a new Data frame instance with all the asociated values
                df = df._append(data_dict, ignore_index=True)
        
        #METHOD NAMA
        elif data_dict["method"] == "NAMA":
            #Loop through each Anchor percentage
            for j in range(len(AP_values)):
                data_dict["Percent_of_Anchors"] = AP_values[j]

                #Now use are data array to grab the FOSCTTM and CE scores
                data_dict["FOSCTTM"] = data[j, 0]
                data_dict["Cross_Embedding_KNN"] = data[j, 1]

                #Create a new Data frame instance with all the asociated values
                df = df._append(data_dict, ignore_index=True)
        
        #METHOD MAGAN
        elif data_dict["method"] == "MAGAN":
            #Loop through each Anchor percentage
            for j in range(len(AP_values)):
                data_dict["Percent_of_Anchors"] = AP_values[j]

                #Now use are data array to grab the FOSCTTM and CE scores
                data_dict["FOSCTTM"] = data[j, 0]
                data_dict["Cross_Embedding_KNN"] = data[j, 1]

                #Create a new Data frame instance with all the asociated values
                df = df._append(data_dict, ignore_index=True)

        elif data_dict["method"] == "Base_Line_Scores" or data_dict["method"] == "KNN_BL" or data_dict["method"] == "BL_CE":
            #Loop through each Knn
            for k in range(0, 10):
                knn = (k*knn_increment) + 2
                data_dict["KNN"] = knn

                #Now use are data array to grab the FOSCTTM and CE scores
                data_dict[data_dict["method"] + "_A"] = data[k, 0]
                data_dict[data_dict["method"] + "_B"] = data[k, 1]

                #Create a new Data frame instance with all the asociated values -- Attach to base_df instead of df
                base_df = base_df._append(data_dict, ignore_index=True)

            return (df, base_df)
            
        elif data_dict["method"] == "RF_BL":
            
            #Now use are data array to grab the FOSCTTM and CE scores
            data_dict["RF_BL_A"] = data[0]
            data_dict["RF_BL_B"] = data[1]

            #Create a new Data frame instance with all the asociated values -- Attach to base_df instead of df
            base_df = base_df._append(data_dict, ignore_index=True)

            return (df, base_df)

        #METHOD DTA and JLMA and PCR
        #METHOD SSMA NOTE: This is literally the same code as DTA's method. We have it seperate for readability (and clarity writing the code the first time), although doesn't need to be. Maybe we can functionalize the process a little bit
        else: 
            #Loop through each Knn
            for i in range(0, 10):
                knn = (i*knn_increment) + 2
                data_dict["KNN"] = knn

                #These percents are rough, and not exact. This is so we can have similar estimates to compare
                data_dict["Percent_of_KNN"] = (i * 0.02) + 0.01

                #Loop through each Anchor percentage
                for j in range(len(AP_values)):
                    data_dict["Percent_of_Anchors"] = AP_values[j]

                    #Now use are data array to grab the FOSCTTM and CE scores
                    data_dict["FOSCTTM"] = data[i, j, 0]
                    data_dict["Cross_Embedding_KNN"] = data[i, j, 1]

                    #Create a new Data frame instance with all the asociated values
                    df = df._append(data_dict, ignore_index=True)

        #Check to make sure there is data in the file
        if np.isnan(data_dict["FOSCTTM"]) or np.isnan(data_dict["Cross_Embedding_KNN"]):
            print(f"File {original_file} is missing FOSCTTM or Cross_Embedding Score")
            #os.remove(original_file)

        if data_dict["FOSCTTM"] == 0 and data_dict["Cross_Embedding_KNN"] == 0:
            print(f"File {original_file} scores are uncalculated.")
            #os.remove(original_file)

        return (df, base_df)
    
    #If there was an error anywhere in processing the data
    except Exception as e:
        print(f"Error occured with {original_file}, and it will not be fully uploaded. It was {e}.")
        #os.remove(original_file)
        
        #It will be empty
        return (df, base_df)

"""IMPORTANT FUNCTIONS"""
def run_all_tests(csv_files = "all", test_random = 1, run_RF_BL_tests = False, run_RF_MASH = False, run_KEMA = False, run_DIG = True, run_CSPUD = False, run_CwDIG = False, run_MALI = False, run_NAMA = True, run_DTA = True, run_SSMA = True, run_MAGAN = False, run_JLMA = False, run_PCR = False, run_KNN_Tests = False, run_RF_SPUD = False, **kwargs):
    """Loops through the tests and files specified. If all csv_files want to be used, let it equal all. Else, 
    specify the csv file names in a list.

    test_random should be a positive integer greater than 1, and is the amount of random tests we want to do. It can also be a list of seeds. TODO: Make it so each random split only occurs once
    
    Returns a dictionary of test_manifold_algorithms class instances."""
        
    """Convert csv_files to class instances"""
    from Helpers.Visualization_helpers import plt_methods_by_CSV_max, subset_df, df

    #Create the dictionary
    manifold_instances = {}

    #Filter out the necessary Key word arguments - NOTE: This will need to be updated based on the KW wanted to be passed
    filtered_kwargs = {}
    if "split" in kwargs:
        filtered_kwargs["split"] = kwargs["split"]
    if "percent_of_anchors" in kwargs:
        filtered_kwargs["percent_of_anchors"] = kwargs["percent_of_anchors"]
    if "verbose" in kwargs:
        filtered_kwargs["verbose"] = kwargs["verbose"]

    #We can add another check in here
    results_df = plt_methods_by_CSV_max(df = subset_df(df, split = kwargs["split"]), metric = "Cross_Embedding_KNN", return_df=True)

    def create_manifold_instances(method, results_df = results_df):

        csv_files = [#"audiology.csv", "balance_scale.csv", "breast_cancer.csv", "Cancer_Data.csv", "car.csv",
                    # "crx.csv", "diabetes.csv", "ecoli_5.csv", "flare1.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "hepatitis.csv",
                    # "hill_valley.csv", "ionosphere.csv", "iris.csv", "Medicaldataset.csv", "mnist_test.csv", "parkinsons.csv",
                    # "seeds.csv", "segmentation.csv", "tic-tac-toe.csv", "titanic.csv", "treeData.csv", "water_potability.csv",
                    # "winequality-red.csv", "zoo.csv", 
                    # "S-curve.csv", "blobs.csv",
                    "EnergyEfficiency.csv", 
                    "Hydrodynamics.csv",
                    "OpticalNetwork.csv",
                    "AirfoilSelfNoise.csv",  
                    "AutoMPG.csv",
                    "ComputerHardware.csv",
                    "CommunityCrime.csv",
                    "ConcreteSlumpTest.csv", 
                        "FacebookMetrics.csv",
                        "Parkinsons.csv",
                    "IstanbulStock.csv",
                    "Automobile.csv",
                    "ConcreteCompressiveStrength.csv",
                    "SML2010.csv"]

        # Create an instance of TestManifoldAlgorithms for each CSV file.
        for csv_file in csv_files:


            #If already run, we don't care
            if np.isnan(results_df[results_df["csv_file"] == csv_file[:-4]][method].values):

                if csv_file == "S-curve.csv":
                    csv_file = "S-curve"
                elif csv_file == "blobs.csv":
                    csv_file = "blobs"

                #Create Pseudo-Random numbers to test the randomness accorind to the test_random parameter
                random.seed(42) #This is to ensure we get the same random numbers each time

                if type(test_random) == list:
                    seeds = test_random
                else:
                    seeds = []
                    for i in range(0, test_random):
                        seeds.append(random.randint(1, 10000))

                for random_seed in seeds:

                    #Create the class and then store it in our dictionary
                    manifold_instance = test_manifold_algorithms(csv_file, random_state=random_seed, **filtered_kwargs)
                    manifold_instances[csv_file + str(random_seed)] = manifold_instance


        # #TODO: Return to getting baseline results
        # csv_files = [ #REGRESSION 
        #         "EnergyEfficiency.csv", "Hydrodynamics.csv",
        #         "CommunityCrime.csv",
        #         "AirfoilSelfNoise.csv",  "AutoMPG.csv",
        #         "ComputerHardware.csv",
        #         "ConcreteSlumpTest.csv",  "FacebookMetrics.csv",
        #         "IstanbulStock.csv", "Parkinsons.csv",
        #         "Automobile.csv", "CommunityCrime.csv",
        #         "ConcreteCompressiveStrength.csv",   "Hydrodynamics.csv",
        #         "OpticalNetwork.csv",
        #         "SML2010.csv"
        # ]

        # for csv_file in csv_files:

        #     #Create the class and then store it in our dictionary
        #     manifold_instance = test_manifold_algorithms(csv_file, random_state=42, **filtered_kwargs)
        #     manifold_instances[csv_file + str(42)] = manifold_instance

        return manifold_instances

    
    """Preform parralell processing and run the tests"""
    if run_DIG:
        #Filter out the necessary Key word arguments for DIG - NOTE: This will need to be updated based on the KW wanted to be passed
        filtered_kwargs = {}
        if "page_ranks" in kwargs:
            filtered_kwargs["page_ranks"] = kwargs["page_ranks"]
        if "predict" in kwargs:
            filtered_kwargs["predict"] = kwargs["predict"]
    
        #Loop through each file (Using Parralel Processing) for DIG
        Parallel(n_jobs=10)(delayed(instance.run_DIG_tests)(**filtered_kwargs) for instance in create_manifold_instances("DIG").values())

    if run_RF_SPUD:
        #Filter out the necessary Key word arguments for SPUD
        filtered_kwargs = {}
        if "OD_methods" in kwargs:
            filtered_kwargs["OD_methods"] = kwargs["OD_methods"]
        if "agg_methods" in kwargs:
            filtered_kwargs["agg_methods"] = kwargs["agg_methods"]

        #Loop through each file (Using Parralel Processing) for SPUD
        Parallel(n_jobs=10)(delayed(instance.run_RF_SPUD_tests)(**filtered_kwargs) for instance in create_manifold_instances("SPUD_RF").values())

    if run_CSPUD:
        #Filter out the necessary Key word arguments for SPUD - NOTE: This will need to be updated based on the KW wanted to be passed
        filtered_kwargs = {}
        if "operations" in kwargs:
            filtered_kwargs["operations"] = kwargs["operations"]
        if "kind" in kwargs:
            filtered_kwargs["kind"] = kwargs["kind"]

        #Loop through each file (Using Parralel Processing) for SPUD
        Parallel(n_jobs=10)(delayed(instance.run_CSPUD_tests)(**filtered_kwargs) for instance in create_manifold_instances("SPUD").values())

    if run_NAMA:
        #Loop through each file (Using Parralel Processing) for NAMA
        Parallel(n_jobs=10)(delayed(instance.run_NAMA_tests)() for instance in create_manifold_instances("NAMA").values())

    if run_MALI:
        
        filtered_kwargs = {}
        if "graph_distances" in kwargs:
            filtered_kwargs["graph_distances"] = kwargs["graph_distances"]

        #Loop through each file (Using Parralel Processing) for NAMA
        Parallel(n_jobs=10)(delayed(instance.run_MALI_tests)(**filtered_kwargs) for instance in create_manifold_instances("MALI").values())

    if run_KEMA:
        #Loop through each file (Using Parralel Processing) for NAMA
        Parallel(n_jobs=10)(delayed(instance.run_KEMA_tests)() for instance in create_manifold_instances("KEMA_RF").values())
    
    if run_DTA:
        #Loop through each file (Using Parralel Processing) for DTA
        Parallel(n_jobs=10)(delayed(instance.run_DTA_tests)() for instance in create_manifold_instances("DTA").values())

    if run_SSMA:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=10)(delayed(instance.run_SSMA_tests)() for instance in create_manifold_instances("SSMA").values())

    if run_JLMA:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=10)(delayed(instance.run_JLMA_tests)() for instance in create_manifold_instances("JLMA").values())

    if run_MAGAN:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=10)(delayed(instance.run_MAGAN_tests)() for instance in create_manifold_instances("MAGAN").values())

    if run_PCR:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=10)(delayed(instance.run_PCR_tests)() for instance in create_manifold_instances("PCR").values())

    #Now run Knn tests
    if run_KNN_Tests:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=10)(delayed(instance.run_KNN_tests)() for instance in create_manifold_instances("Split_A").values())

    #Now run Knn tests
    if run_RF_BL_tests:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=10)(delayed(instance.run_RF_BL_tests)() for instance in create_manifold_instances("RFBL1").values())

    if run_RF_MASH:
        #Filter out the necessary Key word arguments for DIG - NOTE: This will need to be updated based on the KW wanted to be passed
        filtered_kwargs = {}
        if "DTM" in kwargs:
            filtered_kwargs["DTM"] = kwargs["DTM"]
        if "connection_limit" in kwargs:
            filtered_kwargs["connection_limit"] = kwargs["connection_limit"]
    
        #Loop through each file (Using Parralel Processing) for DIG
        Parallel(n_jobs=-10)(delayed(instance.run_RF_MASH_tests)(**filtered_kwargs) for instance in create_manifold_instances("MASH_RF").values())


    return manifold_instances

def upload_to_DataFrame(directory = "default"):
    """Returns a Panda's DataFrame from all the test data"""

    #Loop through each directory to get all the file names
    files = []

    if directory == "default":
        for directory in os.listdir(MANIFOLD_DATA_DIR):
            if os.path.isdir(MANIFOLD_DATA_DIR + directory): #Check to make sure its a directory
                files += [os.path.join(directory, file) for file in os.listdir(MANIFOLD_DATA_DIR + directory)]

        directory = "default"

    elif directory == "regression": 
        for directory in os.listdir(CURR_DIR + "/RegressionData/"):
            if os.path.isdir(CURR_DIR + "/RegressionData/" + directory): #Check to make sure its a directory
                files += [os.path.join(directory, file) for file in os.listdir(CURR_DIR + "/RegressionData/" + directory)]

        directory = "regression"

    else: 
        for directory in os.listdir(CURR_DIR + "/ManifoldData_RF/"):
            if os.path.isdir(CURR_DIR + "/ManifoldData_RF/" + directory): #Check to make sure its a directory
                files += [os.path.join(directory, file) for file in os.listdir(CURR_DIR + "/ManifoldData_RF/" + directory)]
        

    #Use Parralel processing to upload lines to dataframe
    processed_files = Parallel(n_jobs=-5)(delayed(_upload_file)(file, directory) for file in files)

    # Separate the DataFrames from the list of tuples
    dataframes, base_dataframes = zip(*processed_files)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.concat(dataframes, ignore_index=True)
    base_df = pd.concat(base_dataframes, ignore_index = True)

    #Prep base_df for merging
    base_df = base_df.drop(columns=["method", "seed"])

    #Merge the DataFrames together
    if directory == "default":
        merged_df = pd.merge(df, base_df, on=["csv_file",  "split", "KNN"], how = "left")
        #Add the values for methodsthat don't have KNN by first finding the rows with missing 'KNN' values
        missing_knn_df = merged_df[merged_df['KNN'].isna()]

        # Find the best scores within each 'csv_file'
        for csv_file in missing_knn_df['csv_file'].unique():
            best_scores = merged_df[merged_df['csv_file'] == csv_file][['A_Classification_Score', 'B_Classification_Score']].max()
            merged_df.loc[(merged_df['csv_file'] == csv_file) & (merged_df['KNN'].isna()), ['A_Classification_Score', 'B_Classification_Score']] = best_scores.values
        
    else:
        #Prep base_df for merging
        base_df = base_df.drop(columns=["KNN"])
        merged_df = pd.merge(df, base_df, on=["csv_file",  "split"], how = "left")



    return merged_df#.drop_duplicates()

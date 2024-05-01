#This tests all of our models against each other

"""NOTE:
1. Check with Professor to make sure SSMA is implemented right
2. Distance with SPUD seems to be arbitrarly better than the other arguments"""


""" BEST RESULTS FOR IRIS --- FOR EXAMPLE RANDOM
SPUD    
    KNN 8
        FOSCTTM Score: 0.1694222222222222
        CE Score: 0.94

DIG
    KNN 6
        FOSCTTM Score: 0.14026666666666665
        CE Score: 0.96
        Predicted MAE 0.07722901819486289

NAMA
    FOSCTTM: 0.14417777777777777
    Cross Embedding: 0.94

DTA
    KNN 2
        FOSCTTM 0.3175390266299357
        Cross Embedding: 0.3333333333333333

SSMA
    KNN 6
    FOSCTTM 0.4137006427915519
    Cross Embedding: 0.36363636363636365
"""

#Import libraries
from DIG import DIG
from SPUD import SPUD
from ssma import ssma
from nama import NAMA
from DTA_andres import DTA
import numpy as np
import pandas as pd
import utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import random
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.manifold import MDS
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

#Simply, for my sanity
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Create function to do everything
class test_manifold_algorithms():
    def __init__(self, csv_file, split = "skewed", percent_of_anchors = 0.1, verbose = 0):
        """csv_file should be the name of the csv file
        
        split can be 'skewed' (for the features to be split by more important and less important),
        or 'random' for the split to be completely random, or 'even' for each split to have both 
        important or unimportant features
        
        Verbose has different levels. 0 includes no additional prints. 1 prints a little bit more, and
        2 everything."""

        self.verbose = verbose

        self.split = split
        self.prep_data(csv_file)

        self.anchors = self.create_anchors(percent_of_anchors)

        #Set our KNN range dependent on the amount of values in the dataset
        self.knn_range = tuple(self.find_knn_range())
        if verbose > 1:
            print(f"The knn values are: {self.knn_range}")

        #Create file directory to store the information NOTE: THIS WILL NEED TO BE UPDATED IF USED ELSEWHERE.
        self.base_directory = "/Users/user/Desktop/Work/ManifoldData/" + csv_file[:-4] + "/"
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory) 

    """EVALUATION FUNCTIONS"""
    def cross_embedding_knn(self, embedding, Y, knn_args = {'n_neighbors': 1}):
        (y1, y2) = Y

        n1, n2 = len(y1), len(y2)

        knn = KNeighborsClassifier(**knn_args)
        knn.fit(embedding[:n1, :], y1)

        return knn.score(embedding[n1:, :], y2)
    
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

        else:
            # Splitting the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            # Training the RandomForest Classifier
            clf = RandomForestClassifier(random_state=42) #NOTE: this might take forever based on this algorithm 
            clf.fit(X_train, y_train)

            result = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=42)
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
            
            else: #This is an even split
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

        #Reshape if they only have one sample
        if split_a.shape[1] == 1:
            split_a = split_a.reshape(-1, 1)
        if split_b.shape[1] == 1:
            split_b = split_b.reshape(-1, 1)

        return split_a, split_b

    def prep_data(self, csv_file):
        df = pd.read_csv(csv_file)
        features, self.labels = utils.dataprep(df, label_col_idx=0)
        self.split_A, self.split_B = self.split_features(features, self.labels)
        self.labels_doubled = np.concatenate((self.labels, self.labels))

        #We just assume they want the components to be the number of features
        self.mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = 42, n_components = len(features[1]))

    def create_anchors(self, percent_of_anchors):
        #Generate anchors
        rand_ints = random.sample(range(len(self.labels)), int(len(self.labels) * percent_of_anchors))
        return np.vstack([rand_ints, rand_ints]).T

    def find_knn_range(self):
        """Returns a list of probable KNN values. Each list will have ten values"""
        small_data_size = np.min((len(self.split_A), len(self.split_B)))

        #We want a bigger increment of the data is larger
        if small_data_size < 100:
            return range(2, small_data_size, 2)[:10] #This makes us error proof so we wont have a knn value larger than the set, and only gives us ten
        else:
            #Set an increment that gets larger based on the dataset
            increment = (small_data_size // 100) + 1

        #Manufacture a stopping point that will give us ten values
        stop = (increment * 10) + 2

        return range(2, stop, increment)

    def normalize_0_to_1(self, value):
        return (value - value.min()) / (value.max() - value.min())
    
    """RUN TESTS FUNCTIONS"""
    def run_SPUD_tests(self, operations = ("average", "abs"), kind = ("distance", "pure", "similarity")): #NOTE: After lots of tests, Distance seems to always be the best in every scenario
        """Operations should be a tuple of the different operations wanted to run. All are included by default. 
        
        Kind should be a tuple of the different opperations wanted to run. All are included by default."""

        #Store the data in a numpy array
        spud_scores = np.zeros((len(operations), len(kind), len(self.knn_range), 2))


        #We are going to run test with every variation
        print(f"-------------------------------------    SPUD Tests " + self.base_directory[38:-1] + "   -------------------------------------")
        for i, operation in enumerate(operations):
            print(f"Operation {operation}")
            for j, type in enumerate(kind):
                print(f"    Kind {type}")
                for k, knn in enumerate(self.knn_range):
                    print(f"        KNN {knn}")
                    
                    #Create the class with all the arguments
                    spud_class = SPUD(self.split_A, self.split_B, known_anchors=self.anchors, knn = knn, operation = operation, kind = type)

                    #Get the metrics
                    spud_FOSCTTM = self.FOSCTTM(spud_class.matrix_AB)
                    spud_scores[i, j, k, 0] = spud_FOSCTTM
                    print(f"            FOSCTTM Score: {spud_FOSCTTM}")


                    emb = self.mds.fit_transform(spud_class.block)
                    spud_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                    spud_scores[i, j, k, 1] = spud_CE
                    print(f"            CE Score: {spud_CE}")

        #Create filename
        filename = 'SPUD-OPER_'
        for name in operations:
            filename += name + "_"
        
        filename += "KIND_"
        for name in kind: 
            filename += name + "_"

        #Add the last index for knn range so we know what knn values were used
        filename += str(self.knn_range[-1])

        #Finish file name
        filename = self.base_directory + filename + ".npy"

        #Save the numpy array
        np.save(filename, spud_scores)

        #Run successful
        return True

    #We can add t as a parameter, and run tests on that as well, but I feel like the auto is good enough for now
    def run_DIG_tests(self, page_ranks = ("None", "off-diagonal", "full"), predict = False):  #TODO: Add a predict features evaluation 
        """page_ranks should be whether or not we want to test the page_ranks
        
        predict should be a Boolean value and decide whether we want to test the amputation features. 
        NOTE: This assumes a 1 to 1 correspondance with the variables. ThE MAE doesn't make sense if they aren't the same"""

        #Store the data in a numpy array
        DIG_scores = np.zeros((len(page_ranks), len(self.knn_range), 2 + predict))

        #Run through the tests with every variatioin
        print("-------------------------------------   DIG TESTS " + self.base_directory[38:-1] + "   -------------------------------------")
        for i, link in enumerate(page_ranks):
            print(f"Page rank applied: {link}")
            for j, knn in enumerate(self.knn_range):
                print(f"    KNN {knn}")

                #Create our class to run the tests
                DIG_class = DIG(self.split_A, self.split_B, known_anchors=self.anchors, t = -1, knn = knn, link = link)

                #Get Evaluation metrics
                DIG_FOSCTTM = self.FOSCTTM(DIG_class.sim_diffusion_matrix[DIG_class.len_A:, :DIG_class.len_A])
                DIG_scores[i, j, 0] = DIG_FOSCTTM
                print(f"        FOSCTTM Score: {DIG_FOSCTTM}")

                emb = self.mds.fit_transform(DIG_class.sim_diffusion_matrix)
                DIG_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                DIG_scores[i, j, 1] = DIG_CE
                print(f"        CE Score: {DIG_CE}")

                #Predict features test
                if predict: #NOTE: This assumes a 1 to 1 correspondance with the variables. ThE MAE doesn't make sense if they aren't the same
                    #Testing the PREDICT labels features
                    features_pred_B = DIG_class.predict_feature(predict="B")
                    features_pred_A = DIG_class.predict_feature(predict="A")

                    #Get the MAE for each set and average them
                    DIG_MAE = (abs(self.split_B - features_pred_B).mean() + abs(self.split_A - features_pred_A).mean())/2
                    DIG_scores[i, j, 2] = DIG_MAE
                    print(f"        Predicted MAE {DIG_MAE}") #NOTE: this is all scaled 0-1



        #Create filename
        filename = 'DIG-PageRank_'
        for name in page_ranks:
            filename += name + "_"

        #Add the last index for knn range so we know what knn values were used
        filename += str(self.knn_range[-1])

        #Finish file name
        filename = self.base_directory + filename + ".npy"

        #Save the numpy array
        np.save(filename, DIG_scores)

        #Run successful
        return True

    def run_NAMA_tests(self):
        """Needs no additional parameters"""

        #Store the results in an array
        NAMA_scores = np.zeros(2)

        #Create the Nama object on the dataset
        nama = NAMA(ot_reg = 0.001)

        #Fit the value
        nama.fit(self.anchors, self.split_A, self.split_B)

        print("-------------------------------------   NAMA TESTS  " + self.base_directory[38:-1] + "  -------------------------------------")
        #Test FOSCTTM
        nama_FOSCTTM = self.FOSCTTM(nama.cross_domain_dists)
        NAMA_scores[0] = nama_FOSCTTM
        print(f"FOSCTTM: {nama_FOSCTTM}")

        #Get embedding for CE
        emb = self.mds.fit_transform(nama.block)
        nama_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
        NAMA_scores[1] = nama_CE
        print(f"Cross Embedding: {nama_CE}")

         #Finish file name
        filename = self.base_directory + "_NAMA.npy"

        #Save the numpy array
        np.save(filename, NAMA_scores)

        #Run successful
        return True

    def run_DTA_tests(self):
        """Needs no additional parameters"""

        #Store the results in an array
        DTA_scores = np.zeros((len(self.knn_range), 2))

        #Reformat the anchors 
        sharedD1 = self.split_A[self.anchors.T[0]] 
        sharedD2 = self.split_B[self.anchors.T[1]]
        labelsh1 = self.labels[self.anchors.T[0]] #NOTE: Can use these if we want to compare labels
        #labelsh2 = self.labels[self.anchors.T[1]]

        labels_extended = np.concatenate((self.labels, labelsh1))

        print("--------------------------------------   DTA TESTS " + self.base_directory[38:-1] + "   --------------------------------------")
        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")
            #Initialize the class with the correct KNN
            DTA_class = DTA(knn = knn, entR=0.001, verbose = 0)

            #Fit it
            DTA_class.fit(self.split_A, self.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2) 

            #FOSCTTM scores
            DTA_FOSCTTM = self.FOSCTTM(1 - self.normalize_0_to_1(DTA_class.W12)) #Off Diagonal Block. NOTE: it has to be normalized because it returns values 0-2. We subtract one because it is in similarities
            DTA_scores[i, 0] = DTA_FOSCTTM
            print(f"    FOSCTTM {DTA_FOSCTTM}")

            #Cross Embedding Scores
            emb = self.mds.fit_transform(1 - self.normalize_0_to_1(DTA_class.W))
            DTA_CE = self.cross_embedding_knn(emb, (labels_extended, labels_extended), knn_args = {'n_neighbors': 4}) #NOTE: This has a slight advantage because the anchors are counted twice
            DTA_scores[i, 1] = DTA_CE
            print(f"    Cross Embedding: {DTA_CE}")

        #Add the last index for knn range so we know what knn values were used
        filename = "DTA_" + str(self.knn_range[-1])

        #Finish file name
        filename = self.base_directory + filename + ".npy"

        #Save the numpy array
        np.save(filename, DTA_scores)

        #Run successful
        return True
    
    def run_SSMA_tests(self):
        """ No Additional arguments needed"""

        #Create an array to store the important data in 
        SSMA_scores = np.zeros((len(self.knn_range), 2))

        #Reformat the anchors 
        sharedD1 = self.split_A[self.anchors.T[0]] 
        sharedD2 = self.split_B[self.anchors.T[1]]
        labelsh1 = self.labels[self.anchors.T[0]] #NOTE: Can use these if we want to compare labels
        #labelsh2 = self.labels[self.anchors.T[1]]

        labels_extended = np.concatenate((self.labels, labelsh1))

        print("--------------------------------------   SSMA TESTS " + self.base_directory[38:-1] + "   --------------------------------------")
        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")
            #Initialize the class with the correct KNN
            SSMA_class = ssma(knn = knn, verbose = 0, r = 2) #R can also be = to this: (self.split_A.shape[1] + self.split_B.shape[1])

            #Fit it
            SSMA_class.fit(self.split_A, self.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2) 

            #Normalize the W block
            #normalized_W = self.normalize_0_to_1(SSMA_class.W)

            #FOSCTTM scores
            len(self.split_A) + len(self.anchors)
            SSMA_FOSCTTM = self.FOSCTTM(1 - SSMA_class.W[len(SSMA_class.domain1):, :len(SSMA_class.domain1)]) #Off Diagonal Block. NOTE: it has to be normalized because it returns values 0-2. We subtract one because it is in similarities
            SSMA_scores[i, 0] = SSMA_FOSCTTM
            print(f"    FOSCTTM {SSMA_FOSCTTM}")

            #Cross Embedding Scores
            emb = self.mds.fit_transform(1 - SSMA_class.W)
            SSMA_CE = self.cross_embedding_knn(emb, (labels_extended, labels_extended), knn_args = {'n_neighbors': 4}) #NOTE: This has a slight advantage because the anchors are counted twice
            SSMA_scores[i, 1] = SSMA_CE
            print(f"    Cross Embedding: {SSMA_CE}")

        #Add the last index for knn range so we know what knn values were used
        filename = "SSMA_" + str(self.knn_range[-1])

        #Finish file name
        filename = self.base_directory + filename + ".npy"

        #Save the numpy array
        np.save(filename, SSMA_scores)

        #Run successful
        return True

#OUTSIDE OF THE CLASS
def run_all_tests(csv_files = "all", run_DIG = True, run_SPUD = True, run_NAMA = True, run_DTA = True, run_SSMA = True, **kwargs):
    """Loops through the tests and files specified. If all csv_files want to be used, let it equal all. Else, 
    specify the csv file names in a list.
    
    Returns a dictionary of test_manifold_algorithms class instances"""

    #Use all of our files
    if csv_files == "all":
        csv_files = ["artifical_tree.csv", "audiology.csv", "balance_scale.csv", "breast_cancer.csv", "Cancer_Data.csv", "car.csv", "chess.csv", 
                    "crx.csv", "diabetes.csv", "ecoli_5.csv", "flare1.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "hepatitis.csv",
                    "hill_valley.csv", "ionosphere.csv", "iris.csv", "Medicaldataset.csv", "mnist_test.csv", "optdigits.csv", "parkinsons.csv",
                    "seeds.csv", "segmentation.csv", "tic-tac-toe.csv", "titanic.csv", "treeData.csv", "water_potability.csv", "waveform.csv",
                    "winequality-red.csv", "zoo.csv"]
        
    """Convert csv_files to class instances"""
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

    # Create an instance of TestManifoldAlgorithms for each CSV file. Store it in our dictionary
    for csv_file in csv_files:
        manifold_instance = test_manifold_algorithms(csv_file, **filtered_kwargs)
        manifold_instances[csv_file] = manifold_instance

    
    """Preform parralell processing and run the tests"""
    if run_DIG:
        #Filter out the necessary Key word arguments for DIG - NOTE: This will need to be updated based on the KW wanted to be passed
        filtered_kwargs = {}
        if "page_ranks" in kwargs:
            filtered_kwargs["page_ranks"] = kwargs["page_ranks"]
        if "predict" in kwargs:
            filtered_kwargs["predict"] = kwargs["predict"]
    
        #Loop through each file (Using Parralel Processing) for DIG
        Parallel(n_jobs=-1)(delayed(instance.run_DIG_tests)(**filtered_kwargs) for instance in manifold_instances.values())


    if run_SPUD:
        #Filter out the necessary Key word arguments for SPUD - NOTE: This will need to be updated based on the KW wanted to be passed
        filtered_kwargs = {}
        if "operations" in kwargs:
            filtered_kwargs["operations"] = kwargs["operations"]
        if "kind" in kwargs:
            filtered_kwargs["kind"] = kwargs["kind"]

        #Loop through each file (Using Parralel Processing) for SPUD
        Parallel(n_jobs=1)(delayed(instance.run_SPUD_tests)(**filtered_kwargs) for instance in manifold_instances.values())

    if run_NAMA:
        #Loop through each file (Using Parralel Processing) for NAMA
        Parallel(n_jobs=1)(delayed(instance.run_NAMA_tests)() for instance in manifold_instances.values())
    
    if run_DTA:
        #Loop through each file (Using Parralel Processing) for DTA
        Parallel(n_jobs=1)(delayed(instance.run_DTA_tests)() for instance in manifold_instances.values())

    if run_SSMA:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=1)(delayed(instance.run_SSMA_tests)() for instance in manifold_instances.values())

    


    return manifold_instances

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

def clear_directory():
    """CAREFUL. THIS WIPES THE DIRECTORY CLEAN"""

    #Use all of our files
    file_names = ["artifical_tree", "audiology", "balance_scale", "breast_cancer", "Cancer_Data", "car", "chess", 
                "crx", "diabetes", "ecoli_5", "flare1", "glass", "heart_disease", "heart_failure", "hepatitis",
                "hill_valley", "ionosphere", "iris", "Medicaldataset", "mnist_test", "optdigits", "parkinsons",
                "seeds", "segmentation", "tic-tac-toe", "titanic", "treeData", "water_potability", "waveform",
                "winequality-red", "zoo"]

    #Modify the file names to become directory names
    directories = ["/Users/user/Desktop/Work/ManifoldData/" + file_name for file_name in file_names]

    #Loop through each directory and file and get the file paths
    files = []
    for directory in directories:
        if os.path.isdir(directory): #Check to make sure directory exists
            files += [os.path.join(directory, file) for file in os.listdir(directory)]
    
    #Finally delete all files
    for file in files:
        os.remove(file)

    return True

def visualize_results(file_names = "all"):
    """Creates and shows four plots that compare the scores of each method to each other
    
    csv_files tells us which data set we want to compare are data against. It can be a list of files. If
    set to all, all the data sets will be included."""

    #Use all of our files
    if file_names == "all":
        file_names = ["artifical_tree", "audiology", "balance_scale", "breast_cancer", "Cancer_Data", "car", "chess", 
                    "crx", "diabetes", "ecoli_5", "flare1", "glass", "heart_disease", "heart_failure", "hepatitis",
                    "hill_valley", "ionosphere", "iris", "Medicaldataset", "mnist_test", "optdigits", "parkinsons",
                    "seeds", "segmentation", "tic-tac-toe", "titanic", "treeData", "water_potability", "waveform",
                    "winequality-red", "zoo"]

    #Modify the file names to become directory names
    directories = ["/Users/user/Desktop/Work/ManifoldData/" + file_name for file_name in file_names]

    #Loop through each directory and file and get the file paths
    files = []
    for directory in directories:
        if os.path.isdir(directory): #Check to make sure directory exists
            files += [os.path.join(directory, file) for file in os.listdir(directory)]

    # Load each file and sort them 
    NAMA_data = []
    DTA_data = []
    SSMA_data = []
    DIG_full_data = []
    DIG_none_data = []
    DIG_off_diagonal_data = []
    SPUD_avg_data = []
    SPUD_abs_data = []
    for file in files:
        #Sort if data belongs to Nama
        if "_NAMA" in file:
            NAMA_data.append(np.load(file))
        elif "DTA_" in file:
            DTA_data.append(np.load(file))
        elif "SSMA_" in file:
            SSMA_data.append(np.load(file))
        elif "DIG" in file:
            #Sort these based off of page rank values
            data = np.load(file)

            #Loop through each word in order and add the values to corresponding list
            for i, link in enumerate(find_words_order(file, ("None", "off-diagonal", "full"))):
                if link == "None":
                    DIG_none_data.append(data[i])
                elif link == "off-diagonal":
                    DIG_off_diagonal_data.append(data[i])
                else:
                    DIG_full_data.append(data[i])
        else: #Its SPUD data
            #Sort these based off of operation values
            data = np.load(file)

            #Loop through each word in order and add the values to corresponding list
            for i, operation in enumerate(find_words_order(file, ("average", "abs"))):
                if operation == "average":
                    SPUD_avg_data.append(data[i][0]) #NOTE This zero is currently because im not yet testing kind arguments, and am assuming its all distance atm. 
                else: #Thus it contains abs
                    SPUD_abs_data.append(data[i][0])
        
    #Create graphs. X = KNN values based by percentage (each tick goes up by 1%), and Y = Score
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    x_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    #Top left graph will be FOSCTTM scores
    axs[0,0].set_title("FOSCTTM")
    axs[0,0].axhline(y = np.mean(np.array(NAMA_data)[..., 0]), label = "NAMA", c = "black")
    axs[0,0].plot(x_values, np.mean(np.array(DTA_data)[..., 0], axis = 0), label = "DTA")
    axs[0,0].plot(x_values, np.mean(np.array(SSMA_data)[..., 0], axis = 0), label = "SSMA")
    axs[0,0].plot(x_values, np.mean(np.array(DIG_full_data)[..., 0], axis = 0), label = "DIG: Full PageRank")
    axs[0,0].plot(x_values, np.mean(np.array(DIG_none_data)[..., 0], axis = 0), label = "DIG: No PageRank")
    axs[0,0].plot(x_values, np.mean(np.array(DIG_off_diagonal_data)[..., 0], axis = 0), label = "DIG: Off-diagonal PageRank")
    axs[0,0].plot(x_values, np.mean(np.array(SPUD_avg_data)[..., 0], axis = 0), label = "SPUD - Avg")
    axs[0,0].plot(x_values, np.mean(np.array(SPUD_abs_data)[..., 0], axis = 0), label = "SPUD - Abs")
    axs[0,0].legend()
    axs[0,0].set_xlabel("KNN percents (roughly)")
    axs[0,0].set_xlabel("FOSCTTM scores")

    #Bottom left wtill be CE scores
    axs[1,0].set_title("Cross Embedding KNN")
    axs[1,0].axhline(y = np.mean(np.array(NAMA_data)[..., 1]), label = "NAMA", c = "black")
    axs[1,0].plot(x_values, np.mean(np.array(DTA_data)[..., 1], axis = 0), label = "DTA")
    axs[1,0].plot(x_values, np.mean(np.array(SSMA_data)[..., 1], axis = 0), label = "SSMA")
    axs[1,0].plot(x_values, np.mean(np.array(DIG_full_data)[..., 1], axis = 0), label = "DIG: Full PageRank")
    axs[1,0].plot(x_values, np.mean(np.array(DIG_none_data)[..., 1], axis = 0), label = "DIG: No PageRank")
    axs[1,0].plot(x_values, np.mean(np.array(DIG_off_diagonal_data)[..., 1], axis = 0), label = "DIG: Off-diagonal PageRank")
    axs[1,0].plot(x_values, np.mean(np.array(SPUD_avg_data)[..., 1], axis = 0), label = "SPUD - Avg")
    axs[1,0].plot(x_values, np.mean(np.array(SPUD_abs_data)[..., 1], axis = 0), label = "SPUD - Abs")
    axs[1,0].legend()
    axs[1,0].set_xlabel("KNN percents (roughly)")
    axs[1,0].set_xlabel("CE scores")

    #Top Right wtill be Combined scores
    axs[0,1].set_title("Cross Embedding - FOSCTTM")
    axs[0,1].axhline(y = (np.mean(np.array(NAMA_data)[..., 1]) - np.mean(np.array(NAMA_data)[..., 0])), label = "NAMA", c = "black")
    axs[0,1].plot(x_values, (np.mean(np.array(DTA_data)[..., 1], axis = 0) - np.mean(np.array(DTA_data)[..., 0], axis = 0)), label = "DTA")
    axs[0,1].plot(x_values, (np.mean(np.array(SSMA_data)[..., 1], axis = 0) - np.mean(np.array(SSMA_data)[..., 0], axis = 0)), label = "SSMA")
    axs[0,1].plot(x_values, (np.mean(np.array(DIG_full_data)[..., 1], axis = 0) - np.mean(np.array(DIG_full_data)[..., 0], axis = 0)), label = "DIG: Full PageRank")
    axs[0,1].plot(x_values, (np.mean(np.array(DIG_none_data)[..., 1], axis = 0) - np.mean(np.array(DIG_none_data)[..., 0], axis = 0)), label = "DIG: No PageRank")
    axs[0,1].plot(x_values, (np.mean(np.array(DIG_off_diagonal_data)[..., 1], axis = 0) - np.mean(np.array(DIG_off_diagonal_data)[..., 0], axis = 0)), label = "DIG: Off-diagonal PageRank")
    axs[0,1].plot(x_values, (np.mean(np.array(SPUD_avg_data)[..., 1], axis = 0) - np.mean(np.array(SPUD_avg_data)[..., 0], axis = 0)), label = "SPUD - Avg")
    axs[0,1].plot(x_values, (np.mean(np.array(SPUD_abs_data)[..., 1], axis = 0)  - np.mean(np.array(SPUD_abs_data)[..., 0], axis = 0)), label = "SPUD - Abs")
    axs[0,1].legend()
    axs[0,1].set_xlabel("KNN percents (roughly)")
    axs[0,1].set_xlabel("CE scores")



    #Bottom Right will be MAE predict values
    axs[1,1].set_title("Amputation problem")
    axs[1,1].set_xlabel("Knn percents (roughly)")
    axs[1,1].set_ylabel("MAE")
    try: #This may fail if we did let predict = True
        axs[1,1].plot(x_values, np.mean(np.array(DIG_full_data)[..., 2], axis = 0), label = "DIG: Full PageRank")
        axs[1,1].plot(x_values, np.mean(np.array(DIG_none_data)[..., 2], axis = 0), label = "DIG: No PageRank")
        axs[1,1].plot(x_values, np.mean(np.array(DIG_off_diagonal_data)[..., 2], axis = 0), label = "DIG: Off-diagonal PageRank")
    except:
        print("We do not have available the needed predict values")
    
    axs[1,1].legend()


    plt.tight_layout()
    plt.show()


   


#Later -> Add a main function to call the function. Now, we can just call it



"""Practice Tests to Run"""
#test = test_manifold_algorithms("iris.csv", split = "even")
#print(f"Anchors : {test.anchors}")
#print(f"KNN range {test.knn_range}")
#print(f"Features split A {test.split_A.shape}")
#print(f"Features split B {test.split_B.shape}")
#test.run_SPUD_tests(kind = ["distance"])
#test.run_DIG_tests(predict = True)
#test.run_NAMA_tests()
#test.run_DTA_tests()
#test.run_SSMA_tests()

"""Testing All functions"""
class_instances = run_all_tests(csv_files = ["iris.csv"], #["iris.csv", "seeds.csv", "ecoli_5.csv"], 
                                split = "random", verbose = 0, #Init Key arguments
                                run_DIG = True, page_ranks = ("None", "off-diagonal", "full"), predict = True, #DIG key arguments
                                run_DTA = False,
                                run_NAMA = False,
                                run_SSMA = False,
                                run_SPUD = False, kind = ["distance"]) #SPUD key arguments

"""Visualization"""
#visualize_results(file_names = "all")
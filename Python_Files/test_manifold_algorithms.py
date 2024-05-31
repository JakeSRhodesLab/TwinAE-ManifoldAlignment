#This tests all of our models against each other

"""
Questions:
1. What do I do with two FOSCTTMS? Currently I chose to average them... (These are always returning extremely low values -- it may not be right)
- > Work with the prof from the conference?

General Notes:
1. Distance with SPUD seems to be arbitrarly better than the other arguments -> See the Pandas table
2. With random splits, MAGAN preformance can vary dramatically within the same dataset based on the seed

Changes Log:
1. Added New MAGAN Correspondonce method
2. Added Time Logs
3. Added Split_A and Split_B classification Baselines to compare our models too
 - > This is super helpful when looking at the rankings. It tells us how much better each model does compared to the others
4. Added predict anchors function to DIG -> (Super genius I think)
5. Added User verification to the clear all files method. Also added file selection logic. Proceeded to delete all erronious MAGAN Files


FUTURE IDEAS:
1. If kind = Distance is preforming arbitrarly the best, delete the other kind functions
2. Make it so we can have incomplete splits --> or use splits with not all of the data. Its possible that some features may actually hinder the process (Similar to Doctor's being overloaded with information)
3. We could have the algorithm discover "new anchors", and repeat the process with the anchors it guesses are real



TASKS:
0. Time each method :) DONE
1. Figure out MAGAN's Correspondences - Recalculate MAGAN -- Overwrite feature? ... Maybe Parrelize the plot embedding function --> In Process
2. Determine KNN model values for each split -> Nothing Fancy (Like a base case -- No methodology) >>>> Add Baseline File Readings DONE >>>> -> In Process
3. We could have the algorithm discover "new anchors", and repeat the process with the anchors it guesses are real --- Use "hold-out" anchors

----------------------------------------------------------     Helpful Information      ----------------------------------------------------------
Supercomputers Access: carter, collings, cox, hilton, rencher, and tukey
Resource Monitor Websitee: http://statrm.byu.edu/

To Set up Git:
  git config --global user.email "rustadadam@gmail.com"
  git config --global user.name "rustadadam"

Tmux Cheatsheat:
https://gist.github.com/andreyvit/2921703

Tmux Zombies
12. evens on Hilton -- All of the bigest data files (8 days in)
13. all on carter -- RUNNING ALL COMBINATIONS --> (a week in)
14. time on collings -- Running all timing tests (3 days in)
15 MagBig on Tukey -- Running Big MAGAN  (1 day in)
17. base on Rencher -> Running small big (1 day in)
18. magHuge on Hilton -> Runngin the MAssive Magan files (1 day in)


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
import seaborn as sns
import MAGAN
import timeit

#Simply, for my sanity
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Logic to ensure the right directory is always used /yunity/arusty/Graph-Manifold-Alignment/Python_Files
if os.getcwd()[-12:] == "Python_Files":
    CURR_DIR = os.getcwd()[:-13]
else:
    CURR_DIR = os.getcwd()
#Directory Constant
MANIFOLD_DATA_DIR = CURR_DIR + "/ManifoldData/"

#Create function to do everything
class test_manifold_algorithms():
    def __init__(self, csv_file, split = "random", percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3],  verbose = 0, random_state = 42):
        """csv_file should be the name of the csv file. If set to 'S-curve' or "blobs", it will create a toy data set. 
        
        split can be 'skewed' (for the features to be split by more important and less important),
        or 'random' for the split to be completely random, or 'even' for each split to have both 
        important or unimportant features. If split = "distort", then it will create a second dataset, 
        with the features distorted in the second one. 
        
        Verbose has different levels. 0 includes no additional prints. 1 prints a little bit more, and
        2 everything."""

        self.verbose = verbose

        if self.verbose > 0:
            print(f"\n \n \n---------------------------       Initalizing class with {csv_file} data       ---------------------------\n")

        self.random_state = random_state
        random.seed(self.random_state)

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

        #Create anchors
        self.anchors = self.create_anchors()

        #Testing the amount of anchors
        self.percent_of_anchors = percent_of_anchors

        #Set our KNN range dependent on the amount of values in the dataset
        self.knn_range = tuple(self.find_knn_range())
        if verbose > 1:
            print(f"The knn values are: {self.knn_range}")

        #Create file directory to store the information
        self.base_directory = MANIFOLD_DATA_DIR + csv_file[:-4] + "/"
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory) 

    """EVALUATION FUNCTIONS"""
    def cross_embedding_knn(self, embedding, Y, knn_args = {'n_neighbors': 4}):
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
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            # Training the RandomForest Classifier
            clf = RandomForestClassifier(random_state=self.random_state) #NOTE: this might take forever based on this algorithm 
            clf.fit(X_train, y_train)

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

        return split_a, split_b

    def create_blobs(self): #TODO: FINISH DAKINE
        """Creates 3 blobs for each split"""
        self.split_A, self.labels = utils.make_multivariate_data_set(amount=100)
        self.split_B, labels2 = utils.make_multivariate_data_set(amount = 100, adjust=5)

        #Use both labels
        self.labels_doubled = np.concatenate((self.labels, labels2))

        #Create the mds
        self.mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = self.random_state, n_components = 2)

        self.split = "None"

    # TODO: May want to add path as an argument, with the current data path as default
    def prep_data(self, csv_file):
        #Read in file and seperate feautres and labels
        df = pd.read_csv(CURR_DIR + "/CSV Files/" + csv_file)
        features, self.labels = utils.dataprep(df, label_col_idx=0)

        #Ensure that labels are continuous
        unique_values, inverse = np.unique(self.labels, return_inverse=True)
        self.labels = inverse + 1

        #Split the features, and prep data
        self.split_A, self.split_B = self.split_features(features, self.labels)
        self.labels_doubled = np.concatenate((self.labels, self.labels))

        #We just assume they want the components to be the number of features
        n_comp = max(min(len(self.split_B[1]), len(self.split_A[1])), 2)
        self.mds = MDS(metric=True, dissimilarity = 'precomputed', random_state = self.random_state, n_components = n_comp)

        if self.verbose > 1:
            print(f"MDS initialized with {n_comp} components")

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
        filename = method + '(' + self.split[0] + str(self.random_state) + ')'

        #Loop though the Different opperations
        for key in kwargs:
            filename += "_" + str(key)[:3] + "(" + kwargs[key] + ')'

        #Add in the Anchors
        filename += "_AP(" #Short for Anchor Percent
        for name in self.percent_of_anchors:
            filename += str(name) + "-"

        #Add the last index for knn range so we know what knn values were used
        filename = filename[:-1] + ")_" + str(self.knn_range[-1])

        #Finish file name
        filename = self.base_directory + filename + ".npy"

        return filename

    """RUN TESTS FUNCTIONS"""
    def run_SPUD_tests(self, operations = ("average", "abs"), kind = ("distance", "pure", "similarity")): #NOTE: After lots of tests, Distance seems to always be the best in every scenario
        """Operations should be a tuple of the different operations wanted to run. All are included by default. 
        
        Kind should be a tuple of the different opperations wanted to run. All are included by default."""

        #We are going to run test with every variation
        print(f"\n-------------------------------------    SPUD Tests " + self.base_directory[52:-1] + "   -------------------------------------\n")
        for operation in operations:
            print(f"Operation {operation}")
            for type in kind:
                print(f"    Kind {type}")
                
                #Create files and store data
                filename = self.create_filename("SPUD", Operation = operation, Kind = type)

                #If file aready exists, then we are done :)
                if os.path.exists(filename):
                    print(f"        <><><><><>    File {filename} already exists   <><><><><>")
                    continue

                #Store the data in a numpy array
                spud_scores = np.zeros((len(self.knn_range), len(self.percent_of_anchors), 2))

                for k, knn in enumerate(self.knn_range):
                    print(f"        KNN {knn}")
                    for l, anchor_percent in enumerate(self.percent_of_anchors):
                        print(f"            Percent of Anchors {anchor_percent}")

                        try:
                            #Create the class with all the arguments
                            spud_class = SPUD(self.split_A, self.split_B, known_anchors=self.anchors[:int(len(self.anchors) * anchor_percent)], knn = knn, operation = operation, kind = type)
                        except Exception as e:
                            print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e} TEST FAILED   <><><><><><>")
                            spud_scores[k, l, 0] = np.NaN
                            spud_scores[k, l, 1] = np.NaN
                            continue

                        #FOSCTTM METRICS
                        try:
                            spud_FOSCTTM = self.FOSCTTM(spud_class.matrix_AB)
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
    def run_DIG_tests(self, page_ranks = ("None", "off-diagonal", "full"), predict = False):  #TODO: Add a predict features evaluation 
        """page_ranks should be whether or not we want to test the page_ranks
        
        predict should be a Boolean value and decide whether we want to test the amputation features. 
        NOTE: This assumes a 1 to 1 correspondance with the variables. ThE MAE doesn't make sense if they aren't the same"""

        #Run through the tests with every variatioin
        print("\n-------------------------------------   DIG TESTS " + self.base_directory[52:-1] + "   -------------------------------------\n")
        for link in page_ranks:
            print(f"Page rank applied: {link}")

            #Create the filename
            filename = self.create_filename("DIG", PageRanks = link)

            #If file aready exists, then we are done :)
            if os.path.exists(filename):
                print(f"    <><><><><>    File {filename} already exists   <><><><><>")
                return True
            
            #Store the data in a numpy array
            DIG_scores = np.zeros((len(self.knn_range), len(self.percent_of_anchors), 2 + predict))

            for j, knn in enumerate(self.knn_range):
                print(f"    KNN {knn}")
                for k, anchor_percent in enumerate(self.percent_of_anchors):
                    print(f"        Percent of Anchors {anchor_percent}")

                    try:
                        #Create our class to run the tests
                        DIG_class = DIG(self.split_A, self.split_B, known_anchors = self.anchors[:int(len(self.anchors) * anchor_percent)], t = -1, knn = knn, link = link)
                    except Exception as e:
                        print(f"<><><><><><>   UNABLE TO CREATE CLASS BECAUSE {e}  <><><><><><>")
                        DIG_scores[j, k, 0] = np.NaN
                        DIG_scores[j, k, 1] = np.NaN

                        #If we are using predict, this must also be NaN
                        if predict:
                            DIG_scores[j, k, 2] = np.NaN
                        continue

                    #FOSCTTM Evaluation Metrics
                    try:
                        DIG_FOSCTTM = self.FOSCTTM(DIG_class.sim_diffusion_matrix[DIG_class.len_A:, :DIG_class.len_A])
                        print(f"            FOSCTTM Score: {DIG_FOSCTTM}")
                    except Exception as e:
                        print(f"            FOSCTTM exception occured: {e}")
                        DIG_FOSCTTM = np.NaN

                    DIG_scores[j, k, 0] = DIG_FOSCTTM

                    #Cross Embedding Evaluation Metric
                    try:
                        emb = self.mds.fit_transform(DIG_class.sim_diffusion_matrix)
                        DIG_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
                        print(f"            CE Score: {DIG_CE}")
                    except Exception as e:
                        print(f"            Cross Embedding exception occured: {e}")
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
                        print(f"            Predicted MAE {DIG_MAE}") #NOTE: this is all scaled 0-1

                #Save the numpy array
                np.save(filename, DIG_scores)

        #Run successful
        return True

    def run_NAMA_tests(self):
        """Needs no additional parameters"""

        #Create file name
        filename = self.create_filename("NAMA")

        #If file aready exists, then we are done :)
        if os.path.exists(filename):
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True

        #Store the results in an array
        NAMA_scores = np.zeros((len(self.percent_of_anchors), 2))

        #Create the Nama object on the dataset
        nama = NAMA(ot_reg = 0.001)

        print("\n-------------------------------------   NAMA TESTS  " + self.base_directory[52:-1] + "  -------------------------------------\n")
        
        for i, anchor_percent in enumerate(self.percent_of_anchors):
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

        #Create file name
        filename = self.create_filename("DTA")

        #If file aready exists, then we are done :)
        if os.path.exists(filename):
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Store the results in an array
        DTA_scores = np.zeros((len(self.knn_range), len(self.percent_of_anchors), 2))

        print("\n--------------------------------------   DTA TESTS " + self.base_directory[52:-1] + "   --------------------------------------\n")

        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")

            #Initialize the class with the correct KNN
            DTA_class = DTA(knn = knn, entR=0.001, verbose = 0)

            #Loop through each anchor. 
            for j, anchor_percent in enumerate(self.percent_of_anchors):
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
    
    def run_SSMA_tests(self):
        """ No Additional arguments needed"""

        #Add the last index for knn range so we know what knn values were used
        filename = self.create_filename("SSMA")

        #If file aready exists, then we are done :)
        if os.path.exists(filename):
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Create an array to store the important data in 
        SSMA_scores = np.zeros((len(self.knn_range), len(self.percent_of_anchors), 2))

        print("\n--------------------------------------   SSMA TESTS " + self.base_directory[52:-1] + "   --------------------------------------\n")
        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")
            #Initialize the class with the correct KNN
            SSMA_class = ssma(knn = knn, verbose = 0, r = 2) #R can also be = to this: (self.split_A.shape[1] + self.split_B.shape[1])

            #Loop through each anchor. 
            for j, anchor_percent in enumerate(self.percent_of_anchors):
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

        #Create file name
        filename = self.create_filename("MAGAN")

        #If file aready exists, then we are done :)
        """
        if os.path.exists(filename):
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        """

        #Store the results in an array
        MAGAN_scores = np.zeros((2))

        print("\n-------------------------------------   MAGAN TESTS  " + self.base_directory[52:-1] + "  -------------------------------------\n")

        #Run Magan and tests
        domain_a, domain_b, domain_ab, domain_ba = MAGAN.run_MAGAN(self.split_A, self.split_B, self.anchors)

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
        MAGAN_scores[0] = MAGAN_FOSCTTM
        
        #Get embedding for CE
        try:
            emb = self.mds.fit_transform(MAGAN_block)
            MAGAN_CE = self.cross_embedding_knn(emb, (self.labels, self.labels), knn_args = {'n_neighbors': 4})
            print(f"Cross Embedding: {MAGAN_CE}")
        except Exception as e:
            print(f"Cross Embedding exception occured: {e}")
            MAGAN_CE = np.NaN
        MAGAN_scores[1] = MAGAN_CE

        #Save the numpy array
        np.save(filename, MAGAN_scores)

        #Run successful
        return True

    def run_KNN_tests(self):
        """Needs no additional paramenters.
        
        Gets the baseline classification without doing any alignment for each data set"""

        #Create file name
        filename = self.create_filename("Base_Line_Scores")

        #If file aready exists, then we are done :)
        if os.path.exists(filename):
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        #Create an array to store the important data in 
        scores = np.zeros((len(self.knn_range), 2)) #Now both of these are classification scores -- one for Split A and one for Split B

        print("\n--------------------------------------   Base Line Tests " + self.base_directory[53:-1] + "   --------------------------------------\n")
        #Repeat through each knn value
        for i, knn in enumerate(self.knn_range):
            print(f"KNN {knn}")

            #Initilize model
            model = KNeighborsClassifier(n_neighbors = knn)
            
            #Split data and train for split A
            try:
                X_train, X_test, y_train, y_test = train_test_split(self.split_A, self.labels, test_size=0.3, random_state=self.random_state)
                model.fit(X_train, y_train)
                scores[i, 0] = model.score(X_test, y_test)
                print(f"    Classification Score A {scores[i, 0]}")
            except:
                scores[i, 0] = np.NaN
                print(f"    Classification Score A Failed")

            #Split data and train for split B
            try:
                X_train, X_test, y_train, y_test = train_test_split(self.split_B, self.labels, test_size=0.3, random_state=self.random_state)
                model.fit(X_train, y_train)
                scores[i, 1] = model.score(X_test, y_test)
                print(f"    Classification Score B {scores[i, 1]}")
            except:
                scores[1, 1] = np.NaN
                print(f"    Classification Score B Failed")

        #Save the numpy array
        np.save(filename, scores)

        #Run successful
        return True


    """Visualization"""
    def plot_embeddings(self, knn = "auto", anchor_percent = "auto", **kwargs):
        """Shows the embeddings of each graph in a plot"""

        #Set the Knn
        if knn == "auto":
            knn = self.knn_range[2] #This seems to generally be a good KNN percent -- its about 5% knn

        #Choose the most anchors given
        if anchor_percent == "auto":
            anchor_percent = self.percent_of_anchors[-1]

        #Print the Metrics
        if self.verbose > 0:
            print(f"Percent of anchors used: {anchor_percent}")
            print(f"The amount of Nearest Neighbors: {knn}")

        #Create Spuds embedding
        filtered_kwargs = {}
        if "operation" in kwargs:
            filtered_kwargs["operation"] = kwargs["operation"]
        if "kind" in kwargs:
            filtered_kwargs["kind"] = kwargs["kind"]

        spud_class = SPUD(self.split_A, self.split_B, known_anchors=self.anchors[:int(len(self.anchors) * anchor_percent)], knn = knn, **filtered_kwargs)
        SPUD_emb = self.mds.fit_transform(spud_class.block)

        #Create DIG embedding
        if "link" in kwargs:
            link =  kwargs["link"]
        else:
            link = "None"
        
        DIG_class = DIG(self.split_A, self.split_B, known_anchors = self.anchors[:int(len(self.anchors) * anchor_percent)], t = -1, knn = knn, link = link)
        DIG_emb = self.mds.fit_transform(DIG_class.sim_diffusion_matrix)

        #Create NAMA embedding
        nama = NAMA(ot_reg = 0.001)
        nama.fit(self.anchors[:int(len(self.anchors)*anchor_percent)], self.split_A, self.split_B)
        NAMA_emb = self.mds.fit_transform(nama.block)

        #Prep Shared Data points
        sharedD1 = self.split_A[self.anchors[:int(len(self.anchors)*anchor_percent)].T[0]] 
        sharedD2 = self.split_B[self.anchors[:int(len(self.anchors)*anchor_percent)].T[1]]
        labelsh1 = self.labels[self.anchors[:int(len(self.anchors)*anchor_percent)].T[0]] 
        labels_extended = np.concatenate((np.concatenate((self.labels, labelsh1)), np.concatenate((self.labels, labelsh1)))) #This is the extended labels (meaning the labels, and then the shared labels) multiplied by two

        #Create DTA embedding
        DTA_class = DTA(knn = knn, entR=0.001, verbose = 0)
        DTA_class.fit(self.split_A, self.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2)
        DTA_emb = self.mds.fit_transform(1 - self.normalize_0_to_1(DTA_class.W))

        #Create SSMA Embedding | uses the same labels as DTA
        SSMA_class = ssma(knn = knn, verbose = 0, r = 2) #R can also be = to this: (self.split_A.shape[1] + self.split_B.shape[1])
        SSMA_class.fit(self.split_A, self.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2)
        SSMA_emb = self.mds.fit_transform(1 - SSMA_class.W)

        #Create MAGAN Embedding
        domain_a, domain_b, domain_ab, domain_ba = MAGAN.run_MAGAN(self.split_A, self.split_B, self.anchors)
        domain_a, domain_b = MAGAN.get_pure_distance(domain_a, domain_b)
        domain_ab, domain_ba = MAGAN.get_pure_distance(domain_ab, domain_ba)
        magan_block = np.block([[domain_a, domain_ba],
                                [domain_ba, domain_b]])
        MAGAN_emb = self.mds.fit_transform(magan_block)


        """Now Plot the Embeddings"""
        #Create the figure and set titles
        fig, axes = plt.subplots(2, 3, figsize = (16, 10))
        axes[0,0].set_title("NAMA")
        axes[1,0].set_title("SPUD")
        axes[0,1].set_title("DIG")
        axes[0, 2].set_title("SSMA")
        axes[1,1].set_title("DTA")
        axes[1,2].set_title("MAGAN")

        #Create keywords for DIG, SPUD, NAMA
        keywords = {"markers" : {"Graph1": "^", "Graph2" : "o"},
                    "hue" : pd.Categorical(self.labels_doubled),
                    "style" : ['Graph1' if i < len(DIG_emb[:]) / 2 else 'Graph2' for i in range(len(DIG_emb[:]))]
        
        }

        #Now the plotting
        sns.scatterplot(x = NAMA_emb[:, 0], y = NAMA_emb[:, 1], ax = axes[0,0], **keywords)
        sns.scatterplot(x = SPUD_emb[:, 0], y = SPUD_emb[:, 1], ax = axes[1,0], **keywords)
        sns.scatterplot(x = DIG_emb[:, 0], y = DIG_emb[:, 1], ax = axes[0,1], **keywords)
        sns.scatterplot(x = MAGAN_emb[:, 0], y = MAGAN_emb[:, 1], ax = axes[1,2], **keywords)


        #Create keywords for DTA and SSMA
        keywords = {"markers" : {"Graph1": "^", "Graph2" : "o"},
                    "hue" : pd.Categorical(labels_extended),
                    "style" : ['Graph1' if i < len(DTA_emb[:]) / 2 else 'Graph2' for i in range(len(DTA_emb[:]))]
        
        }

        #Now the plotting
        sns.scatterplot(x = DTA_emb[:, 0], y = DTA_emb[:, 1], ax = axes[0,2], **keywords)
        sns.scatterplot(x = SSMA_emb[:, 0], y = SSMA_emb[:, 1], ax = axes[1,1], **keywords)

        plt.plot()

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

def clear_directory(text_curater = "all"):
    """CAREFUL. THIS WIPES THE MANIFOLD DATA DIRECTORY CLEAN"""

    #Use all of our files
    file_names = ["artificial_tree", "audiology", "balance_scale", "breast_cancer", "Cancer_Data", "car", "chess", 
                "crx", "diabetes", "ecoli_5", "flare1", "glass", "heart_disease", "heart_failure", "hepatitis",
                "hill_valley", "ionosphere", "iris", "Medicaldataset", "mnist_test", "optdigits", "parkinsons",
                "seeds", "segmentation", "tic-tac-toe", "titanic", "treeData", "water_potability", "waveform",
                "winequality-red", "zoo"]

    #Modify the file names to become directory names
    directories = [MANIFOLD_DATA_DIR + file_name for file_name in file_names]

    #Loop through each directory and file and get the file paths
    files = []
    for directory in directories:
        if os.path.isdir(directory): #Check to make sure directory exists
            files += [os.path.join(directory, file) for file in os.listdir(directory)]

    if text_curater != "all":
        curated_files = []
        curated_files += [file for file in files if text_curater in file]
    else:
        curated_files = files

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


def _upload_file(file):

    #Create DataFrame
    df = pd.DataFrame(columns= ["csv_file", "method", "seed", "split", "KNN",
                                "Percent_of_KNN", "Percent_of_Anchors", 
                                "Page_Rank", "Predicted_Feature_MAE",
                                "Operation", "SPUDS_Algorithm", 
                                "FOSCTTM", "Cross_Embedding_KNN"])
    
    #Create Base Line Data Frame
    base_df = pd.DataFrame(columns= ["csv_file", "method", "seed", "split", "KNN", "Percent_of_KNN", #Shared headers
                                    "A_Classification_Score", "B_Classification_Score"])

    #Load in the numpy array
    try:
        data = np.load(MANIFOLD_DATA_DIR + file) #allow_pickle=True
    except Exception as e:
        print(f"-------------------------------------------------------------------------------------------------------\nUnable to load {file}. \nError Caught: {e} \nContinuing without uploading file\n-------------------------------------------------------------------------------------------------------")
        pass

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
    
    #Split based on method
    if data_dict["method"] == "DIG":

        #Add the right Page Rank Argument
        if "None" in file:
            data_dict["Page_Rank"] = "None"
        elif "off-diagonal" in file:
            data_dict["Page_Rank"] = "off-diagonal"
        else: #Then it is full
            data_dict["Page_Rank"] = "full"

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
        if "average" in file:
            data_dict["Operation"] = "average"
        else: 
            data_dict["Operation"] = "abs" #There might be a more intuitive name for this 

        #Assign its Kind
        if "distance" in file:
            data_dict["SPUDS_Algorithm"] = "distance"
        elif "pure" in file:
            data_dict["SPUDS_Algorithm"] = "pure"
        else:
            data_dict["SPUDS_Algorithm"] = "similarity"
        
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
    
    elif data_dict["method"] == "MAGAN":
        #Now use are data array to grab the FOSCTTM and CE scores
        data_dict["FOSCTTM"] = data[0]
        data_dict["Cross_Embedding_KNN"] = data[1]

        #Create a new Data frame instance with all the asociated values
        df = df._append(data_dict, ignore_index=True)

    elif data_dict["method"] == "Base_Line_Scores":
        #Loop through each Knn
        for k in range(0, 10):
            knn = (k*knn_increment) + 2
            data_dict["KNN"] = knn

            #Now use are data array to grab the FOSCTTM and CE scores
            data_dict["A_Classification_Score"] = data[k, 0]
            data_dict["B_Classification_Score"] = data[k, 1]

        #Create a new Data frame instance with all the asociated values -- Attach to base_df instead of df
        base_df = base_df._append(data_dict, ignore_index=True)

    #METHOD DTA
    elif data_dict["method"] == "DTA":
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


    return (df, base_df)

def _run_time_trials(csv_file = "iris.csv"):
    #Create file path
    DIR = "/yunity/arusty/Graph-Manifold-Alignment/ManifoldData/Time_DataFrame.csv"

    #Test to see if data already exists
    if os.path.exists(DIR):
        old_data = pd.read_csv(DIR, index_col= None)

        #Check to see if we already have timing for the csv file
        if csv_file in old_data.columns:
            print("Test has already been calculated")
            return True

    #Preform the timing functions
    test = test_manifold_algorithms(csv_file=csv_file, split = "random", percent_of_anchors = [0.1], random_state=9876, verbose = 0)

    # Time the execution of the function -- 10 KNN + 3 page rank methods
    execution_time = {}
    execution_time["DIG"] =  timeit.timeit(test.run_DIG_tests, number=1)/30 #To account for the 3 Page Rank Methods

    #DTA
    execution_time["DTA"] = timeit.timeit(test.run_DTA_tests, number=1)/10 #To account for the 10 knn

    #SSMA
    execution_time["SSMA"] = timeit.timeit(test.run_SSMA_tests, number=1) / 10 #To account for the 10 knn

    #MAGAN
    execution_time["MAGAN"] = timeit.timeit(test.run_MAGAN_tests, number=1)

    #SPUD
    execution_time["SPUD"] = timeit.timeit(test.run_SPUD_tests, number = 1)/60 #To account for 10 knn, 2 operations and 3 algorithms to test

    #NAMA
    execution_time["NAMA"] = timeit.timeit(test.run_NAMA_tests,  number = 1)

    #Convert to DF
    df_executions = pd.DataFrame(list(execution_time.items()), columns = ["Methods", str(csv_file)])

    #Append to old data
    if os.path.exists(DIR):
        #We reget the old data so we always have the freshest version of the DF (meaning, if it was added to while the program ran)
        old_data = pd.read_csv(DIR, index_col= None)

        #Combine the two dataframes together
        df_executions = pd.concat([old_data, df_executions[str(csv_file)]], axis=1)

    #Save to file
    df_executions.to_csv(DIR,  index= False)

    #Print out the results
    print("------------------------------------------------------------------------------    Time Comparisions     ------------------------------------------------------------------------------")
    print(df_executions.sort_values(by = str(csv_file)))

    #Function completed
    return True

"""IMPORTANT FUNCTIONS"""
def time_all_files(csv_files = "all"):
    """Creates a dataframe that stores the time complexity for each method against 1 iteration. Also includes the calculation time
    to calculate FOSCTTM and CE"""

    #Use all of our files
    if csv_files == "all":
        csv_files = ["artificial_tree.csv", "audiology.csv", "balance_scale.csv", "breast_cancer.csv", "Cancer_Data.csv", "car.csv", "chess.csv", 
                    "crx.csv", "diabetes.csv", "ecoli_5.csv", "flare1.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "hepatitis.csv",
                    "hill_valley.csv", "ionosphere.csv", "iris.csv", "Medicaldataset.csv", "mnist_test.csv", "optdigits.csv", "parkinsons.csv",
                    "seeds.csv", "segmentation.csv", "tic-tac-toe.csv", "titanic.csv", "treeData.csv", "water_potability.csv", "waveform.csv",
                    "winequality-red.csv", "zoo.csv", 
                    "S-curve", "blobs"] #Toy data sets -- It will automatically create them

    Parallel(n_jobs=-3)(delayed(_run_time_trials)(csv_file) for csv_file in csv_files)

    return True

def run_all_tests(csv_files = "all", test_random = 1, run_DIG = True, run_SPUD = True, run_NAMA = True, run_DTA = True, run_SSMA = True, run_MAGAN = False, run_KNN_Tests = False, **kwargs):
    """Loops through the tests and files specified. If all csv_files want to be used, let it equal all. Else, 
    specify the csv file names in a list.

    test_random should be a positive integer greater than 1, and is the amount of random tests we want to do. It can also be a list of seeds. TODO: Make it so each random split only occurs once
    
    Returns a dictionary of test_manifold_algorithms class instances."""

    #Use all of our files
    if csv_files == "all":
        csv_files = ["artificial_tree.csv", "audiology.csv", "balance_scale.csv", "breast_cancer.csv", "Cancer_Data.csv", "car.csv", "chess.csv", 
                    "crx.csv", "diabetes.csv", "ecoli_5.csv", "flare1.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "hepatitis.csv",
                    "hill_valley.csv", "ionosphere.csv", "iris.csv", "Medicaldataset.csv", "mnist_test.csv", "optdigits.csv", "parkinsons.csv",
                    "seeds.csv", "segmentation.csv", "tic-tac-toe.csv", "titanic.csv", "treeData.csv", "water_potability.csv", "waveform.csv",
                    "winequality-red.csv", "zoo.csv", 
                    "S-curve", "blobs"] #Toy data sets -- It will automatically create them
        
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

    # Create an instance of TestManifoldAlgorithms for each CSV file.
    for csv_file in csv_files:

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

    
    """Preform parralell processing and run the tests"""
    if run_DIG:
        #Filter out the necessary Key word arguments for DIG - NOTE: This will need to be updated based on the KW wanted to be passed
        filtered_kwargs = {}
        if "page_ranks" in kwargs:
            filtered_kwargs["page_ranks"] = kwargs["page_ranks"]
        if "predict" in kwargs:
            filtered_kwargs["predict"] = kwargs["predict"]
    
        #Loop through each file (Using Parralel Processing) for DIG
        Parallel(n_jobs=-7)(delayed(instance.run_DIG_tests)(**filtered_kwargs) for instance in manifold_instances.values())


    if run_SPUD:
        #Filter out the necessary Key word arguments for SPUD - NOTE: This will need to be updated based on the KW wanted to be passed
        filtered_kwargs = {}
        if "operations" in kwargs:
            filtered_kwargs["operations"] = kwargs["operations"]
        if "kind" in kwargs:
            filtered_kwargs["kind"] = kwargs["kind"]

        #Loop through each file (Using Parralel Processing) for SPUD
        Parallel(n_jobs=-3)(delayed(instance.run_SPUD_tests)(**filtered_kwargs) for instance in manifold_instances.values())

    if run_NAMA:
        #Loop through each file (Using Parralel Processing) for NAMA
        Parallel(n_jobs=-7)(delayed(instance.run_NAMA_tests)() for instance in manifold_instances.values())
    
    if run_DTA:
        #Loop through each file (Using Parralel Processing) for DTA
        Parallel(n_jobs=-7)(delayed(instance.run_DTA_tests)() for instance in manifold_instances.values())

    if run_SSMA:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=-7)(delayed(instance.run_SSMA_tests)() for instance in manifold_instances.values())

    if run_MAGAN:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=-10)(delayed(instance.run_MAGAN_tests)() for instance in manifold_instances.values())

    #Now run Knn tests
    if run_KNN_Tests:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=-3)(delayed(instance.run_KNN_tests)() for instance in manifold_instances.values())


    return manifold_instances

def upload_to_DataFrame():
    """Returns a Panda's DataFrame from all the test data"""

    #Loop through each directory to get all the file names
    files = []
    for directory in os.listdir(MANIFOLD_DATA_DIR):
        if os.path.isdir(MANIFOLD_DATA_DIR + directory): #Check to make sure its a directory
            files += [os.path.join(directory, file) for file in os.listdir(MANIFOLD_DATA_DIR + directory)]

    #Use Parralel processing to upload lines to dataframe
    processed_files = Parallel(n_jobs=-5)(delayed(_upload_file)(file) for file in files)

    # Convert the list of dictionaries to a pandas DataFrame
    dataframes = [file[0] for file in processed_files]
    base_dataframes = [file[1] for file in processed_files]
    df = pd.concat(dataframes, ignore_index=True)
    base_df = pd.concat(base_dataframes, ignore_index = True)

    #Prep base_df for merging
    base_df = base_df.drop(columns=["method", "Percent_of_KNN"])

    #Merge the DataFrames together
    merged_df = pd.merge(df, base_df, on=["csv_file", "seed", "split", "KNN"], how="left")

    return merged_df.drop_duplicates()

def change_old_files_to_new():
    """Goes through all files and changes them to be consistent with the new file formatting"""
    #Loop through each directory to get all the file names
    files = []
    for directory in os.listdir(MANIFOLD_DATA_DIR):
        if os.path.isdir(MANIFOLD_DATA_DIR + directory): #Check to make sure its a directory
            files += [os.path.join(directory, file) for file in os.listdir(MANIFOLD_DATA_DIR + directory)]
    #Create DataFrame
    df = pd.DataFrame(columns= ["csv_file", "method", "seed", "split", "KNN", "Percent_of_Anchors", 
                                "FOSCTTM", "Cross_Embedding_KNN", "Page_Rank", "Predicted_Feature_MAE",
                                "Operation", "SPUDS_Algorithm"])
    #Sort through the Numpy arrays to get the data out
    for file in files:

        #Check to make sure we can load the file
        try:
            data = np.load(MANIFOLD_DATA_DIR + file) #allow_pickle=True
        except Exception as e:
            print(f"-------------------------------------------------------------------------------------------------------\nUnable to load {file}. \nError Caught: {e} \nContinuing Loop without uploading file\n-------------------------------------------------------------------------------------------------------")
            continue

        #Get the name of the csv_file and then cut the csv file out of the name
        csv_file_index = file.find('/')

        #Get the method out of the file
        method_index = file.find('(')
        method = file[csv_file_index + 1 : method_index]
        
        #Split based on method
        if method == "DIG":

            #Get Page Rank indicies
            PageRank_index = file.find('_PR(')
            end_PageRank_index = PageRank_index + file[PageRank_index:].find(')') + 1

            #Check to see if file is not already correct
            if PageRank_index == -1:
                #File is already new
                continue

            #Loop through each Page Rank Argument in order and add the values to corresponding list
            for i, link in enumerate(find_words_order(file, ("None", "off-diagonal", "full"))):
                if link == "None":
                    P_file = file[:PageRank_index] + '_Pag(None)' + file[end_PageRank_index:]
                elif link == "off-diagonal":
                    P_file = file[:PageRank_index] + '_Pag(off-diagonal)' + file[end_PageRank_index:]
                else: #Then it is full
                    P_file = file[:PageRank_index] + '_Pag(full)' + file[end_PageRank_index:]

                #If file aready exists, then we are done :)
                P_file = MANIFOLD_DATA_DIR + P_file
                if os.path.exists(P_file):
                    continue
                else:
                    #Save the numpy array
                    np.save(P_file, data[i])

            #Delete original file
            os.remove(MANIFOLD_DATA_DIR + file)

        #Method SPUD
        elif method == "SPUD":

            #Get operation indicies
            O_index = file.find('_O(')
            end_O_index = O_index + file[O_index:].find(')') + 1

            #Check to see if file is not already correct
            if O_index == -1:
                #File is already new
                continue

            #Loop through each word in order and add the values to corresponding list
            for i, operation in enumerate(find_words_order(file, ("average", "abs"))):
                if operation == "average":
                    O_file = file[:O_index] + '_Ope(average)' + file[end_O_index:]
                else: 
                    O_file = file[:O_index] + '_Ope(abs)' + file[end_O_index:]

                #Get Kind indicies
                K_index = O_file.find('_K(')
                end_K_index = K_index + O_file[K_index:].find(')') + 1

                #Loop through each kind
                for j, kind in enumerate(find_words_order(file, (("distance", "pure", "similarity")))):
                    if kind == "distance":
                        K_file = O_file[:K_index] + '_Kin(distance)' + O_file[end_K_index:]
                    elif kind == "pure": #We should touch up on the SPUD algorithm for this
                       K_file = O_file[:K_index] + '_Kin(pure)' + O_file[end_K_index:]
                    else:
                        K_file = O_file[:K_index] + '_Kin(similarity)' + O_file[end_K_index:]

                    #If file aready exists, then we are done :)
                    K_file = MANIFOLD_DATA_DIR + K_file
                    if os.path.exists(K_file):
                        continue
                    else:
                        #Save the numpy array
                        np.save(K_file, data[i, j])
            
            #Delete original filw
            os.remove(MANIFOLD_DATA_DIR + file)
    
    print("<><><><><><><><><><><><><><><><><><><><><><><>     Updates completed     <><><><><><><><><><><><><><><><><><><><><><><>")
    return True

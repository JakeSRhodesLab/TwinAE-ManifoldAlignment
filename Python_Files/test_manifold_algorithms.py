#This tests all of our models against each other

"""
Questions:
1. Ask Professor Rhodes to share with me one of his presentations
2. Ask why TMUX is so slow


General Notes:
1. Distance with SPUD seems to be arbitrarly better than the other arguments -> See the Pandas table

Changes Log:
1. Added S-curve as a valid csv file option
2. Added the "distort" split method
3. Added "Percent of KNN" to make it easy to graph
4. Added embedding veiwing :)
5. Added Blobs data set as a valid csv option
6. Added the Turn split method

Future Ideas:
1. If kind = Distance is preforming arbitrarly the best, delete the other kind functions
2. Make it so we can have incomplete splits --> or use splits with not all of the data. Its possible that some features may actually hinder the process (Similar to Doctor's being overloaded with information)



TASKS:
0. Run tests wtih different SPUD methods -- Done
1. Implement old datasets (S-curve and things) - DONE
2. Create a High Level presentation -> Copy from the slides to make a presentation
3. Last try efforts for MAGAN TF 2. If can't, refactor code to be compatible to python 2, and run original MAGAN code and have tests be seperate
4. Create a function that visualizes all the embeddings -- For Presentation DONE
5. Apply ranking CSV file visual
6. Go through TODOS
7. Apply rotation tests

----------------------------------------------------------     Helpful Information      ----------------------------------------------------------
Supercomputers Access: carter, collings, cox, hilton, rencher, and tukey
Resource Monitor Websitee: http://statrm.byu.edu/

To Set up Git:
  git config --global user.email "rustadadam@gmail.com"
  git config --global user.name "rustadadam"

Tmux Cheatsheat:
https://gist.github.com/andreyvit/2921703

Tmux Zombies
1. distort on Hilton (Still running - 2 day)
2. even on tukey (Still running - 2 days)
3. tSPUD on Carter (Still running - 1 day)
5. tDIG on Collings (Still running - 1 day)
6. tDTAs (SSMA too) on Hilton (Still running - 1 day)
7. tNAMA on Tukey (Still running - 1 day)
8. dDTA_NAMA on Rencher
9. dS on Rencher

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
    def __init__(self, csv_file, split = "skewed", percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3],  verbose = 0, random_state = 42):
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

        """Now Plot the Embeddings"""
        #Create the figure and set titles
        fig, axes = plt.subplots(2, 3, figsize = (16, 10))
        axes[0,0].set_title("NAMA")
        axes[1,0].set_title("SPUD")
        axes[0,1].set_title("DIG")
        axes[0, 2].set_title("SSMA")
        axes[1,1].set_title("DTA")

        #Create keywords for DIG, SPUD, NAMA
        keywords = {"markers" : {"Graph1": "^", "Graph2" : "o"},
                    "hue" : pd.Categorical(self.labels_doubled),
                    "style" : ['Graph1' if i < len(DIG_emb[:]) / 2 else 'Graph2' for i in range(len(DIG_emb[:]))]
        
        }

        #Now the plotting
        sns.scatterplot(x = NAMA_emb[:, 0], y = NAMA_emb[:, 1], ax = axes[0,0], **keywords)
        sns.scatterplot(x = SPUD_emb[:, 0], y = SPUD_emb[:, 1], ax = axes[1,0], **keywords)
        sns.scatterplot(x = DIG_emb[:, 0], y = DIG_emb[:, 1], ax = axes[0,1], **keywords)

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

def clear_directory():
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
    
    #Finally delete all files
    for file in files:
        os.remove(file)

    return True

def _upload_file(file):

    #Create DataFrame
    df = pd.DataFrame(columns= ["csv_file", "method", "seed", "split", "KNN",
                                "Percent_of_KNN", "Percent_of_Anchors", 
                                "Page_Rank", "Predicted_Feature_MAE",
                                "Operation", "SPUDS_Algorithm", 
                                "FOSCTTM", "Cross_Embedding_KNN"])

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

    return df

"""IMPORTANT FUNCTIONS"""
def run_all_tests(csv_files = "all", test_random = 1, run_DIG = True, run_SPUD = True, run_NAMA = True, run_DTA = True, run_SSMA = True, **kwargs):
    """Loops through the tests and files specified. If all csv_files want to be used, let it equal all. Else, 
    specify the csv file names in a list.

    test_random should be a positive integer greater than 1, and is the amount of random tests we want to do. TODO: Make it so each random split only occurs once
    
    Returns a dictionary of test_manifold_algorithms class instances."""

    #Use all of our files
    if csv_files == "all":
        csv_files = ["artificial_tree.csv", "audiology.csv", "balance_scale.csv", "breast_cancer.csv", "Cancer_Data.csv", "car.csv", "chess.csv", 
                    "crx.csv", "diabetes.csv", "ecoli_5.csv", "flare1.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "hepatitis.csv",
                    "hill_valley.csv", "ionosphere.csv", "iris.csv", "Medicaldataset.csv", "mnist_test.csv", "optdigits.csv", "parkinsons.csv",
                    "seeds.csv", "segmentation.csv", "tic-tac-toe.csv", "titanic.csv", "treeData.csv", "water_potability.csv", "waveform.csv",
                    "winequality-red.csv", "zoo.csv", 
                    "S-curve"] #Toy data sets -- It will automatically create them
        
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
        for i in range(0, test_random):
            random_seed = random.randint(1, 10000)

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
        Parallel(n_jobs=-7)(delayed(instance.run_SPUD_tests)(**filtered_kwargs) for instance in manifold_instances.values())

    if run_NAMA:
        #Loop through each file (Using Parralel Processing) for NAMA
        Parallel(n_jobs=-7)(delayed(instance.run_NAMA_tests)() for instance in manifold_instances.values())
    
    if run_DTA:
        #Loop through each file (Using Parralel Processing) for DTA
        Parallel(n_jobs=-7)(delayed(instance.run_DTA_tests)() for instance in manifold_instances.values())

    if run_SSMA:
        #Loop through each file (Using Parralel Processing) for SSMA
        Parallel(n_jobs=-7)(delayed(instance.run_SSMA_tests)() for instance in manifold_instances.values())

    return manifold_instances

def upload_to_DataFrame():
    """Returns a Panda's DataFrame from all the test data"""

    #Loop through each directory to get all the file names
    files = []
    for directory in os.listdir(MANIFOLD_DATA_DIR):
        if os.path.isdir(MANIFOLD_DATA_DIR + directory): #Check to make sure its a directory
            files += [os.path.join(directory, file) for file in os.listdir(MANIFOLD_DATA_DIR + directory)]

    #Use Parralel processing to upload lines to dataframe
    processed_files = Parallel(n_jobs=-3)(delayed(_upload_file)(file) for file in files)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.concat(processed_files, ignore_index=True)

    return df

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

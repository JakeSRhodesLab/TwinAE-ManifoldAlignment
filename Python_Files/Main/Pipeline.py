# Testing Pipeline 
from Main.test_manifold_algorithms import test_manifold_algorithms as tma
import numpy as np
import os
import json
from joblib import Parallel, delayed
from glob import glob
import logging
from Helpers.Pipeline_Helpers import *


#Start Logging:
logging.basicConfig(filename='/yunity/arusty/Graph-Manifold-Alignment/Resources/Pipeline.log',
                     level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger('Pipe')

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

class pipe():
    """
    A class on initalize runs tests
    """

    def __init__(self, method, csv_files, parallel_factor = 5, 
                splits = ["random", "distort", "turn", "even", "skewed"],
                percent_of_anchors = [0.05, 0.15, 0.3],
                overide_defaults = None,
                **parameters):
        """On init, creates test and methods, and runs tests"""
        
        if overide_defaults is None:
            overide_defaults = {} #the parameters to overide

        self.method_data = method_dict[method]
        self.parallel_factor = parallel_factor
        self.splits = splits
        self.percent_of_anchors = percent_of_anchors
        self.parameters = parameters #The parameters to test
        self.overide_defaults = overide_defaults #the parameters to overide
        self.seed = 42
        self.param_std_dict = {}

        #Check to make sure parameters line up
        self.defaults = get_default_parameters(self.method_data["Model"])


        for param in list(self.parameters.keys()) + list(self.overide_defaults.keys()):
            if param not in self.defaults.keys():
                raise RuntimeError(f"Parameter {param} not valid for {method} class")
            
        #Involve random state
        self.overide_defaults["random_state"] = self.seed  

        #Loop for each split
        for split in splits:
            
            #Loop through each csv_file
            for csv_file in csv_files:

                self.csv_file = csv_file

                print(f"---------------------------------------------      {self.csv_file}     ---------------------------------------------")
                self.tma = tma(csv_file = self.csv_file, split = split, percent_of_anchors = self.percent_of_anchors, random_state=self.seed, verbose = 0)

                #loop for each anchor 
                
                #For Looping each anchor percent
                #Parallel(n_jobs=min(self.parallel_factor, len(self.percent_of_anchors)))(delayed(self.save_tests)(anchor_percent) for anchor_percent in self.percent_of_anchors)
                
                #Normal looping -- Not parrelized
                for anchor_percent in self.percent_of_anchors:
                    self.save_tests(anchor_percent)

    def get_parameter_std(self, parameter, results):
        """Finds the std from within the differing parameter tests"""

        #Only save results for our own methods
        if self.method_data["Name"] not in ["MASH-", "MASH", "SPUD", "NAMA", "RF-MASH-", "RF-MASH"]:
            return False #We don't need to run this
        
        result_new = np.array([result[:2] for result in results])
        
        #Get the combined score of F_score and C_score
        results = result_new[:, 1] - result_new[:, 0]

        #Save the std
        self.param_std_dict[parameter] = np.std(results)

        return True #We ran through it
        
    def run_single_test(self, anchor_percent, test_parameters, tma = None):
        """
        Function to run a single test for a given combination of parameters.
        This will be executed in parallel using ProcessPoolExecutor.
        """

        if tma is None:
            tma = self.tma

        anchor_amount = int(len(tma.anchors) * anchor_percent)       

        try:
            method_class = self.method_data["Model"](**self.overide_defaults, **test_parameters)
            method_class = self.method_data["Fit"](method_class, tma, anchor_amount) #This usually just returns self, except with MAGAN

            # FOSCTTM Evaluation Metrics
            f_score = self.method_data["FOSCTTM"](method_class)

            # Embedding Evaluation Metrics
            emb = tma.mds.fit_transform(self.method_data["Block"](method_class))
            c_score = tma.cross_embedding_knn(emb, (tma.labels, tma.labels), knn_args={'n_neighbors': 4})

            print(f"Results with {self.method_data['Name']} with parameters: {test_parameters}")
            print(f"                FOSCTTM {f_score}")
            print(f"                CE Score {c_score}")

            
            return f_score, c_score, emb

        except Exception as e:
            print(f"<><><>      Tests failed for: {test_parameters}. Why {e}        <><><>")
            logger.warning(f"Name: {self.method_data['Name']}. CSV: {self.csv_file}. Parameters: {test_parameters}. Error: {e}")
            return (np.NaN, np.NaN, np.NaN, np.NaN)
            
    def get_validation_scores(self, emb, tma):

        rf_oob_score = get_RF_score(emb, tma.labels_doubled)
        knn_score = get_KNN_score(emb, tma.labels_doubled)

        print(f"                Random Forest out of bag score {rf_oob_score}")
        print(f"                CE Score {knn_score}")

        return rf_oob_score, knn_score

    def run_tests(self, anchor_percent):
        """
        Run tests over all parameter combinations and find the best ones.
        The KNN parameter is tested first, followed by others sequentially, but parallelized within each step.
        """
        best_fit = {}
        best_f_score = np.NaN
        best_c_score = np.NaN
        best_rf_oob_score= np.NaN
        best_knn_score = np.NaN

        # Step 1: Test the KNN parameter first
        if self.method_data["KNN"]:
            knn_configs = [(anchor_percent, {"knn": knn_value}) for knn_value in self.tma.knn_range]
            knn_results = Parallel(n_jobs=min(self.parallel_factor, 10))(delayed(self.run_single_test)(ap, params) for ap, params in knn_configs)

            self.get_parameter_std("knn", knn_results)

            # Process the results to find the best KNN value
            for (f_score, c_score, emb), (_, params) in zip(knn_results, knn_configs):
                if np.isnan(best_c_score) or (c_score - f_score >  best_c_score - best_f_score):
                    best_f_score = f_score
                    best_c_score = c_score
                    best_fit["knn"] = params["knn"]
                    best_emb = emb


            print(f"----------------------------------------------->     Best KNN Value: {best_fit['knn']}")

        # Step 2: Sequentially test other parameters, while using the best KNN value found
        for i, (parameter, values) in enumerate(self.parameters.items()):
            # Check if it's the last iteration
            if i == len(self.parameters) - 1:
                last_iteration = True
            else:
                last_iteration = False

            #Create flag for default being the best
            is_default_best = True

            #Get all the text cases except when it equals the default
            param_configs = [(anchor_percent, {**best_fit, parameter: value}) for value in values if value != self.defaults[parameter]]
            param_results = Parallel(n_jobs=min(self.parallel_factor, len(param_configs)))(delayed(self.run_single_test)(ap, params) for ap, params in param_configs)

            self.get_parameter_std(parameter, param_results)

            # Process the results to find the best value for the current parameter
            for (f_score, c_score, emb), (_, params) in zip(param_results, param_configs):
                if np.isnan(best_c_score) or (c_score - f_score >  best_c_score - best_f_score): #Simply comparing if all of them together is better
                    best_f_score = f_score
                    best_c_score = c_score
                    best_fit[parameter] = params[parameter]
                    is_default_best = False
                    best_emb = emb

            #Set best_fit to default if necessary
            if is_default_best:
                best_fit[parameter] = self.defaults[parameter]
                best_emb = emb

            print(f"----------------------------------------------->     Best value for {parameter}: {best_fit[parameter]}")
                
        best_rf_oob_score, best_knn_score = self.get_validation_scores(best_emb, self.tma)

        print(f"\n------> Best Parameters: {best_fit}")
        print(f"------------------> Best CE score {best_c_score}")
        print(f"-----------------------------> Best FOSCTTM score {best_f_score}")
        print(f"----------------------------------------> Best Random Forest score {best_rf_oob_score}")
        print(f"---------------------------------------------------> Best Nearest Neighbor score {best_knn_score}")


        C_scores = {42 : best_c_score}
        F_scores = {42 : best_f_score}
        RF_oob_score = {42 : best_rf_oob_score}
        KNN_scores = {42 : best_knn_score}

        #Step 3: Repeat the process with different seeds
        if self.tma.split in ["random", "turn", "distort"]:
            #Delete the seed overide
            self.overide_defaults.pop("random_state")

            #Get all the text cases except when it equals the default
            param_configs = [(anchor_percent, {**best_fit, "random_state": value}, tma(csv_file = self.csv_file, split = self.tma.split, percent_of_anchors = self.percent_of_anchors, random_state=value, verbose = 0)) for value in [1738, 5271, 9209, 1316]]
            param_results = Parallel(n_jobs=min(self.parallel_factor, len(param_configs)))(delayed(self.run_single_test)(ap, params, tma) for ap, params, tma in param_configs)

            # Add the results
            for (f_score, c_score, emb), (_, params, _) in zip(param_results, param_configs):
                seed = params["random_state"]
                C_scores[seed] = c_score
                F_scores[seed] = f_score
                RF_oob_score[seed], KNN_scores[seed] = self.get_validation_scores(emb, self.tma)
            
            #Reset seed default
            self.overide_defaults["random_state"] = self.seed  

        return best_fit, C_scores, F_scores, RF_oob_score, KNN_scores

    def save_tests(self, anchor_percent):

        #Reset the tma anchor percent values to be right
        self.tma.percent_of_anchors = [anchor_percent]

        #Create file name
        filename, AP_values = self.tma.create_filename(self.method_data["Name"], **self.overide_defaults)

        #Remove .npy and replace with json
        filename = filename[:-4] + ".json"

        # #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1: # NOTE: For MASH, we can instead check if MASH- has been run. If it hasn't, we can force it to run MASH- first, and then have a different off-shoot for MASH to run
            print(f"<><><><><>    File {filename} already exists   <><><><><>")
            return True
        
        if self.method_data["Name"] in ["MASH", "RF-MASH"]:
            best_fit, c_score, f_score, rf_oob_score, knn_score =self.run_MASH(anchor_percent, filename)
        else:
            best_fit, c_score, f_score, rf_oob_score, knn_score = self.run_tests(anchor_percent)

        #Ensure the validation scores are not cheating when using RF methods by ensure no labels used twice
        if self.method_data["Name"][:2] == "RF":
            rf_base_score, knn_score = self.get_validation_scores_RF_similarity(best_fit) #Do we even want to get the KNN_score when we used the labels?


        # Combine them into a single dictionary
        combined_data = {
            "method" : self.method_data["Name"],
            "csv_file" : self.csv_file[:-4],
            "split" : self.tma.split,
            "Percent_of_Anchors" : anchor_percent,
            "Best_Params": best_fit,
            "CE": c_score,
            "FOSCTTM": f_score,
            "Random Forrest": rf_oob_score,
            "Nearest Neighbor": knn_score,
            "Parameter STD": self.param_std_dict
        }

        # # Write the combined dictionary to a JSON file
        with open(filename, 'w') as json_file:
            json.dump(combined_data, json_file, indent=4)

        print(f"-      -     -   -  -  -  -  -  - - Data has been saved to {filename} - -  -   -    -     -      -      -")

    def run_MASH(self, anchor_percent, filename):

        #Ensure that MASH- has run
        filename = filename.replace("MASH", "MASH-")

        #Find matching files
        filename = glob(filename[:filename.find("AP")] + "*" + ".json")[0]

        # #If file doesn't exsist, return false
        if not os.path.exists(filename): 
            print("MASH- must be run first")
            return False

        #Read and get the best fit
        with open(filename, 'r') as f:
            data = json.load(f)
        
        best_fit = data["Best_Params"]
        
        #Refit with best found parameters
        anchor_amount = int((len(self.tma.anchors) * anchor_percent)/2)       
        method_class = self.method_data["Model"](**self.overide_defaults, **best_fit)
        method_class.fit(self.tma.split_A, self.tma.split_B, known_anchors=self.tma.anchors[:anchor_amount])

        #Set score
        best_c_score = np.NaN
        best_f_score = np.NaN
        best_rf_oob_score = np.NaN
        best_knn_score = np.NaN


        for parameter, values in {"connection_limit": ["auto", 1000, 5000, None], "threshold" : [0.2, 0.5, 0.8, 1], "epochs" : [3, 10, 200]}.items():
            
            #Create flag for default being the best
            is_default_best = True

            #Get all the text cases except when it equals the default
            param_configs = [{"hold_out_anchors" : np.array(self.tma.anchors[:int(len(self.tma.anchors) * anchor_percent)]), parameter: value} for value in values]

            #NOTE: Memory scare posibilty here: This is because we have to do a deep copy of the class
            param_results = Parallel(n_jobs=min(self.parallel_factor, len(param_configs)))(delayed(get_mash_score_connected)(method_class, self.tma, **best_fit, **params) for params in param_configs)

            self.get_parameter_std(parameter, param_results)

            # Process the results to find the best value for the current parameter
            for (c_score, f_score, rf_oob_score, knn_score), dictionary in zip(param_results, param_configs):
                if np.isnan(best_c_score) or (c_score - f_score >  best_c_score - best_f_score):
                    best_f_score = f_score
                    best_c_score = c_score
                    best_rf_oob_score = rf_oob_score
                    best_knn_score = knn_score
                    best_fit[parameter] = dictionary[parameter]
                    is_default_best = False

            #Set best_fit to default if necessary
            if is_default_best:
                best_fit[parameter] = {"epochs" : 100, "threshold" : "auto", "connection_limit" : "auto"}[parameter]

            print(f"----------------------------------------------->     Best value for {parameter}: {best_fit[parameter]}")

        C_scores = {42 : best_c_score}
        F_scores = {42 : best_f_score}
        RF_oob_score = {42 : best_rf_oob_score}
        KNN_scores = {42 : best_knn_score}

        #Step 3: Repeat the process with different seeds # RETURN WORKING HERE
        if self.tma.split in ["random", "turn", "distort"]:
            #Delete the seed overide
            self.overide_defaults.pop("random_state")

            #Get all the text cases except when it equals the default
            param_configs = [{**best_fit, "hold_out_anchors" : np.array(self.tma.anchors[:int(len(self.tma.anchors) * anchor_percent)]), "random_state": value} for value in [1738, 5271, 9209, 1316]]
            tma_configs = [tma(csv_file = self.csv_file, split = self.tma.split, percent_of_anchors = self.percent_of_anchors, random_state=value, verbose = 0) for value in [1738, 5271, 9209, 1316]]
            param_results = Parallel(n_jobs=min(self.parallel_factor, len(param_configs)))(delayed(get_mash_score_connected)(method_class, tma, **params) for params, tma in zip(param_configs, tma_configs))

            # Add the results
            for (f_score, c_score, rf_oob_score, knn_score), params in zip(param_results, param_configs):
                seed = params["random_state"]
                C_scores[seed] = c_score
                F_scores[seed] = f_score
                RF_oob_score[seed] = rf_oob_score
                KNN_scores[seed] = knn_score
            
            #Reset seed default
            self.overide_defaults["random_state"] = self.seed  

        return best_fit, C_scores, F_scores, RF_oob_score, KNN_scores
    
    def get_validation_scores_RF_similarity(best_fit):
        """ We will need a special embedding formation, and keep track of the labels and things. """
        return None, None
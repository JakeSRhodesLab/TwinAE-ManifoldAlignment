# Testing Pipeline 
from Main.test_manifold_algorithms import test_manifold_algorithms as tma
import numpy as np
import os
import json
from joblib import Parallel, delayed
from glob import glob
import logging
from Helpers.Pipeline_Helpers import *
from sklearn.model_selection import train_test_split
from Helpers.Grae import *

"""
Return to editing MASH with the get_calidation scores
-> I think this file could be simplified more
"""

#Start Logging:
logging.basicConfig(filename='/yunity/arusty/Graph-Manifold-Alignment/Resources/Pipeline.log',
                     level=logging.DEBUG, format='%(asctime)s  -> %(levelname)s: %(message)s')
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

        try:
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
        
        except Exception as e:
            logger.warning(f"Unexpected Failure with {self.method_data['Name']}. Error: {e}")
            raise Exception(e)

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

        anchors = self.tma.anchors[:int(len(tma.anchors) * anchor_percent)]       

        try:
            method_class = self.method_data["Model"](**self.overide_defaults, **test_parameters)
            method_class = self.method_data["Fit"](method_class, tma, anchors) #This usually just returns self, except with MAGAN

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
            return (np.NaN, np.NaN, np.NaN)
            
    def get_validation_scores(self, emb, tma, seed, best_fit):

        #Update tma labels if needed for Andres fit methods
        if len(emb) != len(tma.labels_doubled):
            labelsh1 = tma.labels[tma.anchors[:int(len(tma.anchors) * tma.percent_of_anchors[0])].T[0]]
            tma.labels = np.concatenate((tma.labels, labelsh1))
            tma.labels_doubled = np.concatenate((tma.labels, tma.labels))

        rf_oob_score = get_RF_score(emb, tma.labels_doubled, seed)

        if self.method_data["Name"][:2] == "RF":
            #To avoid it changing outside of the class
            from copy import deepcopy
            best_fit = deepcopy(best_fit)

            #To avoid using the data on the text and Train, we will need to split it
            X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(tma.split_A, tma.labels, test_size=0.2, random_state=seed)
            X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(tma.split_B, tma.labels, test_size=0.2, random_state=seed)

            #Because of RF MASH, we need to specialize the initilization so we can call optimize on it later
            if self.method_data["Name"] == "RF-MASH":
                anchors = tma.anchors[:int(len(tma.anchors) * tma.percent_of_anchors[0]/2)]
                optimize_dict = {"hold_out_anchors": tma.anchors[:int(len(tma.anchors) * tma.percent_of_anchors[0])]}
                for param in ["connection_limit", "threshold", "epochs"]:
                    optimize_dict[param] = best_fit[param]
                    del best_fit[param]

                #Delete hold out anchors
                if "hold_out_anchors" in best_fit.keys():
                    del best_fit["hold_out_anchors"]
                
            else:
                anchors = tma.anchors[:int(len(tma.anchors) * tma.percent_of_anchors[0])]

            #Create model
            rf_method_class = self.method_data["Model"](**self.overide_defaults, **best_fit)

            #Fit it. Should work for all that can use the Rhodes Test Fit Model
            rf_method_class = Rhodes_test_fit(rf_method_class, (X_A_train, X_A_test, y_A_train), (X_B_train, X_B_test, y_B_train), anchors) #This works because we garuntee the tests are the same size

            #Calculate GRAE variant

            if self.method_data["Name"] == "RF-MASH":
                rf_method_class.optimize_by_creating_connections(**optimize_dict)

            emb = tma.mds.fit_transform(self.method_data["Block"](rf_method_class))
        
        else:
            #We want to test 80/20 still, but it doesn't matter which part because we didn't use any labels this time
            X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(emb[:int(len(emb)/2)], tma.labels, test_size=0.2, random_state=seed)
            X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(emb[int(len(emb)/2):], tma.labels, test_size=0.2, random_state=seed)
            
            #Resticth the embedding 
            emb = np.vstack((X_A_train, X_A_test, X_B_train, X_B_test))

        #Get scores
        knn_score, rf_score, knn_metric, rf_metric = get_embedding_scores(emb, seed, (y_A_train, y_A_test, y_B_train, y_B_test))
        
        print(f"                KNN Score {knn_score}")
        print(f"                RF on embedding Score {rf_score}")
        print(f"                Random Forest out of bag score {rf_oob_score}")
        print(f"                KNN's f1 or Root mean square error score {rf_score}")
        print(f"                Random Forest f1 or Root mean square error score {rf_oob_score}")

        return rf_oob_score, knn_score, rf_score, knn_metric, rf_metric

    def get_GRAE_validation_scores(self, emb, tma, seed, best_fit):
        """
        Get the GRAE version of the validation metrics. We use the emb only to compare if we need to change tma sizes
        """

    
        if not self.method_data["Name"][:2] == "RF":
            #No need to calculate GRAE
            return None, None, None, None, None

        #To avoid it changing outside of the class
        from copy import deepcopy
        best_fit = deepcopy(best_fit)

        #To avoid using the data on the text and Train, we will need to split it
        X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(tma.split_A, tma.labels, test_size=0.2, random_state=seed)
        X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(tma.split_B, tma.labels, test_size=0.2, random_state=seed)

        #Because of RF MASH, we need to specialize the initilization so we can call optimize on it later
        if self.method_data["Name"] == "RF-MASH":

            optimize_dict = {"hold_out_anchors": create_unique_pairs(len(X_A_train), int(len(tma.anchors) * tma.percent_of_anchors[0]))}

            #Select half of the anchors for training
            anchors = optimize_dict["hold_out_anchors"][:int(len(tma.anchors) * tma.percent_of_anchors[0]/2)]
            
            for param in ["connection_limit", "threshold", "epochs"]:
                optimize_dict[param] = best_fit[param]
                del best_fit[param]

            #Delete hold out anchors
            if "hold_out_anchors" in best_fit.keys():
                del best_fit["hold_out_anchors"]
            
        else:
            anchors = create_unique_pairs(len(X_A_train), int(len(tma.anchors) * tma.percent_of_anchors[0]))

        #Create model
        rf_method_class = self.method_data["Model"](**self.overide_defaults, **best_fit)

        #Fit using x train
        tma.split_A = X_A_train
        tma.split_B = X_B_train
        tma.labels = y_A_train # y_A_train will equal y_B_train
        
        rf_method_class = self.method_data["Fit"](rf_method_class, tma, anchors)

        #Optimize for RF-MASH
        if self.method_data["Name"] == "RF-MASH":
            rf_method_class.optimize_by_creating_connections(**optimize_dict)

        #Get the embedding
        emb = tma.mds.fit_transform(self.method_data["Block"](rf_method_class))

        #GRAE on domain A
        myGrae = GRAEBase()
        split_A = BaseDataset(x = X_A_train, y = y_A_train, split_ratio = 0.8, random_state = seed, split = "none")
        myGrae.fit(split_A, emb=emb)
        testA = BaseDataset(x = X_A_test, y = y_A_test, split_ratio = 0.8, random_state = seed, split = "none")
        pred_A = myGrae.score(testA)

        #Grae on domain B 
        split_B = BaseDataset(x = X_B_train, y = y_B_train, split_ratio = 0.8, random_state = seed, split = "none")
        myGrae.fit(split_B, emb=emb)
        testB = BaseDataset(x = X_B_test, y = y_B_test, split_ratio = 0.8, random_state = seed, split = "none")
        pred_B = myGrae.score(testB)
        
        #Grab the scores
        emb = np.vstack([X_A_train, pred_A, X_B_train, pred_B])
        knn_score, rf_score, knn_metric, rf_metric = get_embedding_scores(emb, seed, (y_A_train, y_A_test, y_B_train, y_B_test))
        rf_oob_score = get_RF_score(emb, tma.labels_doubled, seed)

        print(f"                GRAE KNN Score {knn_score}")
        print(f"                GRAE RF on embedding Score {rf_score}")
        print(f"                GRAE Random Forest out of bag score {rf_oob_score}")
        print(f"                GRAE KNN's f1 or Root mean square error score {rf_score}")
        print(f"                GRAE Random Forest f1 or Root mean square error score {rf_oob_score}")

        return rf_oob_score, knn_score, rf_score, knn_metric, rf_metric

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
                
        best_rf_oob_score, best_knn_score, best_rf_score, best_knn_metric, best_rf_metric = self.get_validation_scores(best_emb, self.tma, 42, best_fit)

        print(f"\n------> Best Parameters: {best_fit}")
        print(f"------------------> Best CE score {best_c_score}")
        print(f"-----------------------------> Best FOSCTTM score {best_f_score}")
        print(f"----------------------------------------> Best Random Forest score {best_rf_oob_score}")
        print(f"---------------------------------------------------> Best Nearest Neighbor score {best_knn_score}")
        print(f"--------------------------------------------------------------> Best KNN metric score {best_knn_metric}")
        print(f"-----------------------------------------------------------------------> Best random Forest score {best_rf_metric}")

        grae_rf_oob_score, grae_knn_score, grae_rf_score, grae_knn_metric, grae_rf_metric = self.get_GRAE_validation_scores(best_emb, self.tma, 42, best_fit)


        C_scores = {42 : best_c_score}
        F_scores = {42 : best_f_score}
        RF_oob_score = {42 : best_rf_oob_score}
        KNN_scores = {42 : best_knn_score}
        RF_score = {42: best_rf_score}
        KNN_metric = {42: best_knn_metric}
        RF_metric = {42: best_rf_metric}
        GRAE_results = { 42 : {"RF-OOB" : grae_rf_oob_score,
                        "KNN" :grae_knn_score, 
                        "RF" : grae_rf_score,
                        "KNN-metric" : grae_knn_metric,
                        "RF-metric": grae_rf_metric}
        }

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
                RF_oob_score[seed], KNN_scores[seed], RF_score[seed], KNN_metric[seed], RF_metric[seed] = self.get_validation_scores(emb, self.tma, seed, best_fit)
                grae_rf_oob_score, grae_knn_score, grae_rf_score, grae_knn_metric, grae_rf_metric = self.get_GRAE_validation_scores(best_emb, self.tma, 42, best_fit)
                GRAE_results[seed] = {"RF-OOB" : grae_rf_oob_score,
                        "KNN" :grae_knn_score, 
                        "RF" : grae_rf_score,
                        "KNN-metric" : grae_knn_metric,
                        "RF-metric": grae_rf_metric}
                        
            
            #Reset seed default
            self.overide_defaults["random_state"] = self.seed  

        return best_fit, C_scores, F_scores, RF_oob_score, KNN_scores, RF_score, KNN_metric, RF_metric, GRAE_results

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
            best_fit, c_score, f_score, rf_oob_score, knn_score, rf_emb_score, knn_metric, rf_metric =self.run_MASH(anchor_percent, filename)
        else:
            best_fit, c_score, f_score, rf_oob_score, knn_score, rf_emb_score, knn_metric, rf_metric, grae_results = self.run_tests(anchor_percent)

        # Combine them into a single dictionary
        combined_data = {
            "method" : self.method_data["Name"],
            "csv_file" : self.csv_file[:-4],
            "split" : self.tma.split,
            "Percent_of_Anchors" : anchor_percent,
            "Best_Params": best_fit,
            "CE": c_score,
            "FOSCTTM": f_score,
            "Random Forest OOB": rf_oob_score,
            "Random Forest Emb": rf_emb_score,
            "Nearest Neighbor": knn_score,
            "Nearest Neighbor (F1 score or RMSE)": knn_metric,
            "Random Forest (F1 score or RMSE)": rf_metric,
            "GRAE" : grae_results,
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
        anchors = self.tma.anchors[:int((len(self.tma.anchors) * anchor_percent)/2)]     
        method_class = self.method_data["Model"](**self.overide_defaults, **best_fit)
        method_class.fit(self.tma.split_A, self.tma.split_B, known_anchors= anchors)

        #Set score
        best_c_score = np.NaN
        best_f_score = np.NaN
        best_emb = np.NaN

        for parameter, values in {"connection_limit": ["auto", 1000, 5000, None], "threshold" : [0.2, 0.5, 0.8, 1], "epochs" : [3, 10, 200]}.items():
            
            #Create flag for default being the best
            is_default_best = True

            #Get all the text cases except when it equals the default
            param_configs = [{"hold_out_anchors" : np.array(self.tma.anchors[:int(len(self.tma.anchors) * anchor_percent)]), parameter: value} for value in values]

            #NOTE: Memory scare posibilty here: This is because we have to do a deep copy of the class
            param_results = Parallel(n_jobs=min(self.parallel_factor, len(param_configs)))(delayed(get_mash_score_connected)(method_class, self.tma, **best_fit, **params) for params in param_configs)

            self.get_parameter_std(parameter, param_results)

            # Process the results to find the best value for the current parameter
            for (f_score, c_score, emb), dictionary in zip(param_results, param_configs):
                if np.isnan(best_c_score) or (c_score - f_score >  best_c_score - best_f_score):
                    best_f_score = f_score
                    best_c_score = c_score
                    best_emb = emb
                    best_fit[parameter] = dictionary[parameter]
                    is_default_best = False

            #Set best_fit to default if necessary
            if is_default_best:
                best_fit[parameter] = {"epochs" : 100, "threshold" : "auto", "connection_limit" : "auto"}[parameter]

            print(f"----------------------------------------------->     Best value for {parameter}: {best_fit[parameter]}")

        rf_oob_score, knn_score, rf_score, knn_metric, rf_metric = self.get_validation_scores(best_emb, self.tma, 42, best_fit)

        C_scores = {42 : best_c_score}
        F_scores = {42 : best_f_score}
        RF_oob_score = {42 : rf_oob_score}
        KNN_scores = {42 : knn_score}
        RF_score = {42: rf_score}
        KNN_metric = {42: knn_metric}
        RF_metric = {42: rf_metric}

        grae_rf_oob_score, grae_knn_score, grae_rf_score, grae_knn_metric, grae_rf_metric = self.get_GRAE_validation_scores(best_emb, self.tma, 42, best_fit)
        GRAE_results = {42: {"RF-OOB" : grae_rf_oob_score,
                        "KNN" :grae_knn_score, 
                        "RF" : grae_rf_score,
                        "KNN-metric" : grae_knn_metric,
                        "RF-metric": grae_rf_metric}}

        #Step 3: Repeat the process with different seeds # RETURN WORKING HERE
        if self.tma.split in ["random", "turn", "distort"]:
            #Delete the seed overide
            self.overide_defaults.pop("random_state")

            #Get all the text cases except when it equals the default
            param_configs = [{**best_fit, "hold_out_anchors" : np.array(self.tma.anchors[:int(len(self.tma.anchors) * anchor_percent)]), "random_state": value} for value in [1738, 5271, 9209, 1316]]
            tma_configs = [tma(csv_file = self.csv_file, split = self.tma.split, percent_of_anchors = self.percent_of_anchors, random_state=value, verbose = 0) for value in [1738, 5271, 9209, 1316]]
            param_results = Parallel(n_jobs=min(self.parallel_factor, len(param_configs)))(delayed(get_mash_score_connected)(method_class, tma, **params) for params, tma in zip(param_configs, tma_configs))

            # Add the results
            for (f_score, c_score, emb), params in zip(param_results, param_configs):
                seed = params["random_state"]
                C_scores[seed] = c_score
                F_scores[seed] = f_score

                rf_oob_score, knn_score, rf_score, knn_metric, rf_metric = self.get_validation_scores(emb, self.tma, seed, params)

                RF_oob_score[seed] = rf_oob_score
                KNN_scores[seed] = knn_score
                RF_score[seed] = rf_score
                KNN_metric[seed] = knn_metric
                RF_metric[seed] = rf_metric

                grae_rf_oob_score, grae_knn_score, grae_rf_score, grae_knn_metric, grae_rf_metric = self.get_GRAE_validation_scores(best_emb, self.tma, 42, best_fit)
                GRAE_results[seed] = {"RF-OOB" : grae_rf_oob_score,
                        "KNN" :grae_knn_score, 
                        "RF" : grae_rf_score,
                        "KNN-metric" : grae_knn_metric,
                        "RF-metric": grae_rf_metric}
            
            #Reset seed default
            self.overide_defaults["random_state"] = self.seed  

        return best_fit, C_scores, F_scores, RF_oob_score, KNN_scores, RF_score, KNN_metric, RF_metric
    
# Testing Pipeline 
from test_manifold_algorithms import test_manifold_algorithms as tma
import numpy as np
from mashspud import MASH, SPUD
import os
import json
from joblib import Parallel, delayed
import inspect
from DTA_andres import DTA


# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")


"""To do:
>Test the force parameters method"""

"""
Idea Notes:
parallelize the inner_function (computing the knn). Its less memory intensive, and so we only have to 
load one dataset at a time. 

"""

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

def get_mash_score_connected(self, tma, **kwargs):
    
    import copy

    #We need to copy the class as it get changed and will be parralized
    self = copy.deepcopy(self)
    self.optimize_by_creating_connections(**kwargs)

    # Cross Embedding Evaluation Metric
    emb = tma.mds.fit_transform(self.int_diff_dist)
    c_score = tma.cross_embedding_knn(emb, (tma.labels, tma.labels), knn_args={'n_neighbors': 4})
    f_score = np.mean([self.FOSCTTM(self.int_diff_dist[:self.len_A, self.len_A:]), self.FOSCTTM(self.int_diff_dist[self.len_A:, :self.len_A])])

    print(f"                FOSCTTM {f_score}")
    print(f"                CE Sore {c_score}")

    #Return FOSCTTM score
    return c_score, f_score

def Rustad_fit(self, tma, anchor_amount):
    self.fit(tma.split_A, tma.split_B, known_anchors=tma.anchors[:anchor_amount])

def Andres_fit(self, tma, anchor_amount):
    #Reformat the anchors 
    sharedD1 = tma.split_A[tma.anchors[:anchor_amount].T[0]] 
    sharedD2 = tma.split_B[tma.anchors[:anchor_amount].T[1]]
    labelsh1 = tma.labels[tma.anchors[:anchor_amount].T[0]]
    
    #We only need to overide the labels once otherwise it will mess up the CE score
    if len(tma.labels) != (len(labelsh1) + len(tma.split_A)):
        tma.labels = np.concatenate((tma.labels, labelsh1))

    self.fit(tma.split_A, tma.split_B, sharedD1 = sharedD1, sharedD2 = sharedD2)
    
#Create dictionaries for the different classes
method_dict = {
     "MASH-" : {"Name": "MASH-", "Model": MASH, "KNN" : True, "Seed" : "random_state", "Block" : lambda mash: mash.int_diff_dist, "FOSCTTM" : mash_foscttm, "Fit" : Rustad_fit},
     "MASH" : {"Name": "MASH", "Model": MASH, "KNN" : True, "Seed" : "random_state", "Block" : lambda mash: mash.int_diff_dist, "FOSCTTM" : mash_foscttm, "Fit" : Rustad_fit},
     "SPUD" : {"Name": "SPUD", "Model": SPUD, "KNN" : True, "Seed" : "random_state", "Block" : lambda spud: spud.block, "FOSCTTM" : spud_foscttm, "Fit" : Rustad_fit},
     "NAMA" : {"Name": "NAMA", "Model": SPUD, "KNN" : False, "Seed" : "random_state", "Block" : lambda spud: spud.block, "FOSCTTM" : spud_foscttm, "Fit" : Rustad_fit},
     
     #NOTE: adopted fit below
     "DTA" : {"Name": "DTA", "Model": DTA, "KNN" : True, "Seed" : "random_state", "Block" : lambda dta: 1 - tma.normalize_0_to_1(None, dta.W), "FOSCTTM" : lambda dta : tma.FOSCTTM(None, 1 - tma.normalize_0_to_1(None, dta.W12)), "Fit": Andres_fit}


 }

class pipe():
    """
    A class on initalize runs tests
    """

    def __init__(self, method, csv_files, parallel_factor = 5, 
                splits = ["random", "distort", "turn", "even", "skewed"],
                percent_of_anchors = [0.05, 0.15, 0.3],
                overide_defaults = {},
                **parameters):
        """On init, creates test and methods, and runs tests"""
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
            
        #Involve random state if it exists
        if False != self.method_data["Seed"]:  
            self.overide_defaults[self.method_data["Seed"]] = self.seed  


        #Preform tests

        #Loop for each split
        for split in splits:
            
            #Loop through each csv_file
            for csv_file in csv_files:

                print(f"---------------------------------------------      {csv_file}     ---------------------------------------------")
                self.tma = tma(csv_file = csv_file, split = split, percent_of_anchors = self.percent_of_anchors, random_state=self.seed, verbose = 0)

                #loop for each anchor 
                
                #For Looping each anchor percent
                #Parallel(n_jobs=min(self.parallel_factor, len(self.percent_of_anchors)))(delayed(self.save_tests)(anchor_percent) for anchor_percent in self.percent_of_anchors)
                
                #Normal looping -- Not parrelized
                for anchor_percent in self.percent_of_anchors:
                    self.save_tests(anchor_percent, csv_file)

    def get_parameter_std(self, parameter, results):
        """Finds the std from within the differing parameter tests"""

        #Only save results for our own methods
        if self.method_data["Name"] not in ["MASH-", "MASH", "SPUD", "NAMA"]:
            return False #We don't need to run this
        
        #Get the combined score of F_score and C_score
        results = np.array(results)
        results = results[:, 1] - results[:, 0] 

        #Save the std
        self.param_std_dict[parameter] = np.std(results)

        return True #We ran through it
        
    def run_single_test(self, anchor_percent, test_parameters):
        """
        Function to run a single test for a given combination of parameters.
        This will be executed in parallel using ProcessPoolExecutor.
        """

        anchor_amount = int(len(self.tma.anchors) * anchor_percent)       

        try:
            method_class = self.method_data["Model"](**self.overide_defaults, **test_parameters)
            self.method_data["Fit"](method_class, self.tma, anchor_amount)

            # FOSCTTM Evaluation Metrics
            f_score = self.method_data["FOSCTTM"](method_class)

            # Cross Embedding Evaluation Metric
            emb = self.tma.mds.fit_transform(self.method_data["Block"](method_class))
            c_score = self.tma.cross_embedding_knn(emb, (self.tma.labels, self.tma.labels), knn_args={'n_neighbors': 4})
           
            print(f"Results with {self.method_data['Name']} with parameters: {test_parameters}")
            print(f"                FOSCTTM {f_score}")
            print(f"                CE Sore {c_score}")

        except Exception as e:
            print(f"<><><>      Tests failed for: {test_parameters}. Why {e}        <><><>")
            return (100, 0)
            
        return f_score, c_score

    def run_tests(self, anchor_percent):
        """
        Run tests over all parameter combinations and find the best ones.
        The KNN parameter is tested first, followed by others sequentially, but parallelized within each step.
        """
        best_fit = {}
        best_f_score = 100
        best_c_score = 0

        # Step 1: Test the KNN parameter first
        if self.method_data["KNN"]:
            knn_configs = [(anchor_percent, {"knn": knn_value}) for knn_value in self.tma.knn_range]
            knn_results = Parallel(n_jobs=min(self.parallel_factor, 10))(delayed(self.run_single_test)(ap, params) for ap, params in knn_configs)

            self.get_parameter_std("knn", knn_results)

            # Process the results to find the best KNN value
            for (f_score, c_score), (_, params) in zip(knn_results, knn_configs):
                if best_c_score - best_f_score < c_score - f_score:
                    best_f_score = f_score
                    best_c_score = c_score
                    best_fit["knn"] = params["knn"]

            print(f"----------------------------------------------->     Best KNN Value: {best_fit['knn']}")

        # Step 2: Sequentially test other parameters, while using the best KNN value found
        for parameter, values in self.parameters.items():

            #Create flag for default being the best
            is_default_best = True

            #Get all the text cases except when it equals the default
            param_configs = [(anchor_percent, {**best_fit, parameter: value}) for value in values if value != self.defaults[parameter]]
            param_results = Parallel(n_jobs=min(self.parallel_factor, len(param_configs)))(delayed(self.run_single_test)(ap, params) for ap, params in param_configs)

            self.get_parameter_std(parameter, param_results)

            # Process the results to find the best value for the current parameter
            for (f_score, c_score), (_, params) in zip(param_results, param_configs):
                if best_c_score - best_f_score < c_score - f_score:
                    best_f_score = f_score
                    best_c_score = c_score
                    best_fit[parameter] = params[parameter]
                    is_default_best = False

            #Set best_fit to default if necessary
            if is_default_best:
                best_fit[parameter] = self.defaults[parameter]

            print(f"----------------------------------------------->     Best value for {parameter}: {best_fit[parameter]}")

        #Step 3: Run last -- Only for run MASH optimzation
        if self.method_data["Name"] == "MASH":

            #Refit with best found parameters
            anchor_amount = int((len(self.tma.anchors) * anchor_percent)/2)       
            method_class = self.method_data["Model"](**self.overide_defaults, **best_fit)
            method_class.fit(self.tma.split_A, self.tma.split_B, known_anchors=self.tma.anchors[:anchor_amount])

            for parameter, values in {"connection_limit": ["auto", 1000, 5000, None], "threshold" : [0.2, 0.5, 0.8, 1], "epochs" : [3, 10, 200]}.items():
                
                #Create flag for default being the best
                is_default_best = True

                #Get all the text cases except when it equals the default
                param_configs = [{"hold_out_anchors" : np.array(self.tma.anchors[:int(len(self.tma.anchors) * anchor_percent)]), parameter: value} for value in values]

                #NOTE: Memory scare posibilty here: This is because we have to do a deep copy of the class
                param_results = Parallel(n_jobs=min(self.parallel_factor, len(param_configs)))(delayed(get_mash_score_connected)(method_class, self.tma, **params) for params in param_configs)

                self.get_parameter_std(parameter, param_results)

                # Process the results to find the best value for the current parameter
                for (c_score, f_score), dictionary in zip(param_results, param_configs):
                    if best_c_score - best_f_score < c_score - f_score:
                        best_f_score = f_score
                        best_c_score = c_score
                        best_fit[parameter] = dictionary[parameter]
                        is_default_best = False

                #Set best_fit to default if necessary
                if is_default_best:
                    best_fit[parameter] = {"epochs" : 100, "threshold" : "auto", "connection_limit" : "auto"}[parameter]

                print(f"----------------------------------------------->     Best value for {parameter}: {best_fit[parameter]}")

                
        print(f"\n------> Best Parameters: {best_fit}")
        print(f"------------------> Best CE score {best_c_score}")
        print(f"-----------------------------> Best FOSCTTM score {best_f_score}")

        #Set the scores to be dictionaries. This is so we know what random seed lead to what score
        C_scores = {42 : best_c_score}
        F_scores = {42 : best_f_score}

        #Step 4: Repeat the process with different seeds
        if self.method_data["Seed"] != False:
            #Delete the seed overide
            self.overide_defaults.pop(self.method_data["Seed"])

            #Get all the text cases except when it equals the default
            param_configs = [(anchor_percent, {**best_fit, self.method_data["Seed"]: value}) for value in [1738, 5271, 9209, 1316]]
            param_results = Parallel(n_jobs=min(self.parallel_factor, len(param_configs)))(delayed(self.run_single_test)(ap, params) for ap, params in param_configs)

            # Add the results
            for (f_score, c_score), (_, params) in zip(param_results, param_configs):
                seed = params[self.method_data["Seed"]]
                C_scores[seed] = c_score
                F_scores[seed] = f_score
            
            #Reset seed default
            self.overide_defaults[self.method_data["Seed"]] = self.seed  

        return best_fit, C_scores, F_scores

    def save_tests(self, anchor_percent, csv_file):

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
        
        best_fit, c_score, f_score = self.run_tests(anchor_percent)


        # Combine them into a single dictionary
        combined_data = {
            "Best_Params": best_fit,
            "CE": c_score,
            "FOSCTTM": f_score,
            "Percent_of_Anchors" : anchor_percent,
            "split" : self.tma.split,
            "csv_file" : csv_file,
            "method" : self.method_data["Name"],
            "Parameter STD": self.param_std_dict
        }

        # # Write the combined dictionary to a JSON file
        with open(filename, 'w') as json_file:
            json.dump(combined_data, json_file, indent=4)

        print(f"-      -     -   -  -  -  -  -  - - Data has been saved to {filename} - -  -   -    -     -      -      -")
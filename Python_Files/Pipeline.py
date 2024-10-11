# Testing Pipeline 
from test_manifold_algorithms import test_manifold_algorithms as tma
import numpy as np
from mashspud import MASH
import os
import json
from joblib import Parallel, delayed
import inspect

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

def get_mash_block(self):
    return self.int_diff_dist

#Create dictionaries for the different classes
method_dict = {
     "MASH" : {"Name": "MASH", "Model": MASH, "KNN" : True, "Block" : get_mash_block, "FOSCTTM" : mash_foscttm}

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

        #Check to make sure parameters line up
        self.defaults = get_default_parameters(self.method_data["Model"])

        for param in list(self.parameters.keys()) + list(self.overide_defaults.keys()):
            if param not in self.defaults.keys():
                raise RuntimeError(f"Parameter {param} not valid for {method} class")

        #Preform tests

        #Loop for each split
        for split in splits:
            
            #Loop through each csv_file
            for csv_file in csv_files:

                print(f"---------------------------------------------      {csv_file}     ---------------------------------------------")
                self.tma = tma(csv_file = csv_file, split = split, percent_of_anchors = [], verbose = 0)

                #loop for each anchor 
                
                #Parallel(n_jobs=min(self.parallel_factor, len(self.percent_of_anchors)))(delayed(self.save_tests)(anchor_percent) for anchor_percent in self.percent_of_anchors)
                
                for anchor_percent in self.percent_of_anchors:
                    self.save_tests(anchor_percent)

    def run_single_test(self, anchor_percent, test_parameters):
        """
        Function to run a single test for a given combination of parameters.
        This will be executed in parallel using ProcessPoolExecutor.
        """
        anchor_amount = int((len(self.tma.anchors) * anchor_percent) / 2)       

        try:
            method_class = self.method_data["Model"](**self.overide_defaults, **test_parameters)
            method_class.fit(self.tma.split_A, self.tma.split_B, known_anchors=self.tma.anchors[:anchor_amount])

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
            knn_results = Parallel(n_jobs=min(self.parallel_factor, 5))(delayed(self.run_single_test)(ap, params) for ap, params in knn_configs)

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

        print(f"\n------> Best Parameters: {best_fit}")
        print(f"------------------> Best CE score {best_c_score}")
        print(f"-----------------------------> Best FOSCTTM score {best_f_score}")

        return best_fit, best_c_score, best_f_score

    def save_tests(self, anchor_percent):

        #Create file name
        filename, AP_values = self.tma.create_filename(self.method_data["Name"])

        #Remove .npy and replace with json
        filename = filename[:-4] + ".json"

        # #If file aready exists, then we are done :)
        if os.path.exists(filename) or len(AP_values) < 1:
             print(f"<><><><><>    File {filename} already exists   <><><><><>")
             #return True
        
        best_fit, c_score, f_score = self.run_tests(anchor_percent)


        # Combine them into a single dictionary
        combined_data = {
            "Best_Params": best_fit,
            "CE": c_score,
            "FOSCTTM": f_score
        }

        # # Write the combined dictionary to a JSON file
        # with open(filename, 'w') as json_file:
        #     json.dump(combined_data, json_file, indent=4)

        print(f"-      -     -   -  -  -  -  -  - - Data has been saved to {filename} - -  -   -    -     -      -      -")
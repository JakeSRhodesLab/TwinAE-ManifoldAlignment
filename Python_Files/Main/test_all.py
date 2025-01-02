print("Process Running...\n")

import os
# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")
#import Main.test_manifold_algorithms as tma
from Main.Pipeline import pipe


# file_names = ["artificial_tree", "audiology", "balance_scale", "breast_cancer", "Cancer_Data", "car", "chess", 
#                 "crx", "diabetes", "ecoli_5", "flare1", "glass", "heart_disease", "heart_failure", "hepatitis",
#                 "hill_valley", "ionosphere", "iris", "Medicaldataset", "optdigits", "parkinsons",
#                 "seeds", "segmentation", "tic-tac-toe", "titanic", "treeData", "water_potability", "waveform",
#                 "winequality-red", "zoo"]



# csv_files = [f"{file_name}.csv" for file_name in file_names]

csv_files = [
             "zoo.csv", "hepatitis.csv", "iris.csv", "audiology.csv", "parkinsons.csv", "seeds.csv", 
              "segmentation.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "flare1.csv", 
              "ecoli_5.csv", "ionosphere.csv", "Cancer_Data.csv", "hill_valley.csv", "balance_scale.csv",
             "S-curve", "blobs", 'winequality-red.csv', 'car.csv',
            "crx.csv", "breast_cancer.csv", "titanic.csv", 
              "diabetes.csv", "tic-tac-toe.csv",
              'Medicaldataset.csv', "water_potability.csv", 
             'treeData.csv', 
              #"optdigits.csv", "waveform.csv", "chess.csv", "artificial_tree.csv"
             ]

reg_files = [ #REGRESSION 
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
 "SML2010.csv"
]

"""
<><><><><<><><><><><><><><><><><><><><><>   Testing All functions      <><><><><><><><><><><><><><><><><><><><>><><><><><><

state = False
nope = not state

#Skewed


tma.run_all_tests(csv_files = reg_files, test_random = [1825],#, 2830, 2969],# 3407, 3430, 5198], #General function arguments: 1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515
                            split = "skewed", verbose = 0, percent_of_anchors = [0.3], #Init Key arguments
                            run_DIG = state, 
                            run_DTA = state,
                            run_NAMA = state,
                            run_SSMA = state,
                            run_MAGAN= state,
                            run_JLMA = state,
                            run_KNN_Tests= nope,
                            run_PCR = state,
                            run_MALI=state, #graph_distances = ["rf_gap"],
                            run_RF_MASH= state,
                            run_RF_SPUD= state,
                            run_KEMA = state,
                            run_RF_BL_tests = state,
                            run_CSPUD = state) #SPUD key arguments

#Even
tma.run_all_tests(csv_files = reg_files, test_random = [1738],#, 1825, 2830],# 3407, 3430, 5198, 7667, 9515], #General function arguments: 1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515
                            split = "even", verbose = 0, percent_of_anchors = [0.3], #Init Key arguments
                            run_DIG = state, 
                            run_DTA = state,
                            run_NAMA = state,
                            run_SSMA = state,
                            run_MAGAN= state,
                            run_JLMA = state,
                            run_KNN_Tests=nope,
                            run_PCR = state,
                            run_MALI= state, #graph_distances = ["rf_gap"],
                            run_RF_MASH= state,
                            run_RF_SPUD= state,
                            run_KEMA = state,
                            run_RF_BL_tests = state,
                            run_CSPUD = state) #SPUD key arguments

#turn
tma.run_all_tests(csv_files = reg_files, test_random = [1738],#, 9515, 1825], #General function arguments: 1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515
                            split = "turn", verbose = 0, percent_of_anchors = [0.3], #Init Key arguments
                            run_DIG = state, 
                            run_DTA = state,
                            run_NAMA = state,
                            run_SSMA = state,
                            run_MAGAN= state,
                            run_JLMA = state,
                            run_KNN_Tests= nope,
                            run_PCR = state,
                            run_MALI= state, #graph_distances = ["rf_gap"],
                            run_RF_MASH= state,
                            run_RF_SPUD= state,
                            run_KEMA = state,
                            run_RF_BL_tests = state,
                            run_CSPUD = state) #SPUD key arguments

#Random
tma.run_all_tests(csv_files = reg_files, test_random = [1738],#, 1825, 2830],#, 3407, 3430, 5198, 7667, 9515], #General function arguments: 1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515
                            split = "random", verbose = 0, percent_of_anchors = [0.3], #Init Key arguments
                            run_DIG = state, 
                            run_DTA = state,
                            run_NAMA = state,
                            run_SSMA = state,
                            run_MAGAN= state,
                            run_JLMA = state,
                            run_KNN_Tests=nope,
                            run_PCR = state,
                            run_MALI=state, #graph_distances = ["rf_gap"],
                            run_RF_MASH= state,
                            run_RF_SPUD=state,
                            run_KEMA = state,
                            run_RF_BL_tests = state,
                            run_CSPUD = state) #SPUD key arguments

#distort
tma.run_all_tests(csv_files = reg_files, test_random =  [1738],#, 5198, 7667],# 9515], #General function arguments: 1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515
                            split = "distort", verbose = 0, percent_of_anchors = [0.3], #Init Key arguments
                            run_DIG = state, 
                            run_DTA = state,
                            run_NAMA = state,
                            run_SSMA = state,
                            run_MAGAN= state,
                            run_JLMA = state,
                            run_KNN_Tests= nope,
                            run_PCR = state,
                            run_MALI=state, #graph_distances = ["rf_gap"],
                            run_RF_MASH= state, DTM = ("log"),
                            run_RF_SPUD= state,
                            run_KEMA = state,
                            run_RF_BL_tests = state, 
                            run_CSPUD = state) #SPUD key argument
"""

# #Pipeline Tests
from Pipeline import pipe

SPLITS = ["distort", "even", "random", "skewed", "turn"]
PF = 1

"""
Files 1-3 ran. 

Files 3-6 crashed on MASH- due to excess memory requirements.

File 6 crashed on MASH due to excess memory requirements.

Files 7-10 still on Parkinsons with MASH-.





RF Methods Below -> \/


pipe("RF-SPUD", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        overide_defaults= {"overide_method" : "none"},
        OD_method = ["default", "absolute_distance", "mean"],  agg_method = ['sqrt', 'log', 0.5, 'None'])
pipe("RF-SPUD", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        overide_defaults= {"overide_method" : "Jaccard"},
        OD_method = ["default", "absolute_distance", "mean"],  agg_method = ['sqrt', 'log', 0.5, 'None'])
pipe("RF-SPUD", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        overide_defaults= {"overide_method" : "similarities"},
        OD_method = ["default", "absolute_distance", "mean"],  agg_method = ['sqrt', 'log', 0.5, 'None'])
pipe("RF-NAMA", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        overide_defaults= {"overide_method" : "NAMA"},
        OD_method = ["absolute_distance", "mean"],  agg_method = ['sqrt', 'log', 0.5, 'None'])


Our methods below -> \/



#We sorted out the overide methods between each of the spuds
pipe("SPUD", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        overide_defaults= {"overide_method" : "none"},
        OD_method = ["default", "absolute_distance", "mean"],  agg_method = ['sqrt', 'log', 0.5, 'None'])
pipe("SPUD", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        overide_defaults= {"overide_method" : "Jaccard"},
        OD_method = ["default", "absolute_distance", "mean"],  agg_method = ['sqrt', 'log', 0.5, 'None'])
pipe("SPUD", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        overide_defaults= {"overide_method" : "similarities"},
        OD_method = ["default", "absolute_distance", "mean"],  agg_method = ['sqrt', 'log', 0.5, 'None'])
pipe("NAMA", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        overide_defaults= {"overide_method" : "NAMA"},
        OD_method = ["absolute_distance", "mean"],  agg_method = ['sqrt', 'log', 0.5, 'None'])



Other methods below -> \/


pipe("JLMA", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
     normalized_laplacian = [True, False], d = [1, 2, 3, 4, 5, PF], mu = [0.01, 0.5, 1, 2])

pipe("DTA", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        distances = ["DPT", "Not_DPT"])

pipe("SSMA", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        Uincluded = [True, False], Dincluded = [True, False])

pipe("MAPA", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        r = [2,5,10,20, 50, 100, 1000])

pipe("MALI", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        mu = [0.01, 0.1, 0.3, 0.5, 0.75, 0.99], t = ["auto", "auto-I", "DPT", "DPT-I", 3, 5, 30], transition_only = [True, False],
        ot = [True, False], normalize_M = [True, False])

pipe("RF-MALI", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
        mu = [0.01, 0.1, 0.3, 0.5, 0.75, 0.99], t = ["auto", "auto-I", "DPT", "DPT-I", 3, 5, 30], transition_only = [True, False],
        ot = [True, False], normalize_M = [True, False], interclass_distance = ["rfgap", "cosine"],
        overide_defaults= {"graph_distance" : "rfgap"}) 

pipe("MAGAN", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
     learning_rate = [0.01, 0.005, 0.001])


Mash Methods Below \/
"""

pipe("MASH-", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
    page_rank = ["None", "off-diagonal", "full"],  DTM = ["hellinger", "kl", "log"], density_normalization = [True, False])

# pipe("MASH", csv_files=reg_files, splits = SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
#     page_rank = ["None", "off-diagonal", "full"],  DTM = ["hellinger", "kl", "log"], density_normalization = [True, False])

# pipe("RF-MASH-", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
#     page_rank = ["None", "off-diagonal", "full"],  DTM = ["hellinger", "kl", "log"], density_normalization = [True, False])

# pipe("RF-MASH", csv_files=reg_files, splits =  SPLITS, percent_of_anchors=[0.3], parallel_factor = PF,
#     page_rank = ["None", "off-diagonal", "full"],  DTM = ["hellinger", "kl", "log"], density_normalization = [True, False])


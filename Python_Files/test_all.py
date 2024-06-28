# Start by reviewing the DataVisualizationTests.ipynb; review and make any necesar changes there first

#Import everything
import test_manifold_algorithms as tma


# file_names = ["artificial_tree", "audiology", "balance_scale", "breast_cancer", "Cancer_Data", "car", "chess", 
#                 "crx", "diabetes", "ecoli_5", "flare1", "glass", "heart_disease", "heart_failure", "hepatitis",
#                 "hill_valley", "ionosphere", "iris", "Medicaldataset", "optdigits", "parkinsons",
#                 "seeds", "segmentation", "tic-tac-toe", "titanic", "treeData", "water_potability", "waveform",
#                 "winequality-red", "zoo"]



# csv_files = [f"{file_name}.csv" for file_name in file_names]

csv_files = ["zoo.csv", "hepatitis.csv", "iris.csv", "audiology.csv", "parkinsons.csv", "seeds.csv", 
             "segmentation.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "flare1.csv", 
             "ecoli_5.csv", "ionosphere.csv", "Cancer_Data.csv", "hill_valley.csv", "balance_scale.csv",
             #"S-curve", "blobs",
             "crx.csv", "breast_cancer.csv", "titanic.csv", "diabetes.csv", "tic-tac-toe.csv",
             'Medicaldataset.csv', "water_potability.csv",
             'treeData.csv', 'winequality-red.csv', 'car.csv'
             ]

"""
<><><><><><><><><><><><><><><><><><><><><>    Timing all functions  <><><><><><><<><><><><>><<><><><><><>><><<>><<><><><><><><>
"""

#tma.time_all_files("all")


"""
<><><><><<><><><><><><><><><><><><><><><>   Testing All functions      <><><><><><><><><><><><><><><><><><><><>><><><><><><
"""


#for split_type in ["skewed", "even", "distort", "random", "turn"]:

#Even
tma.run_all_tests(csv_files = csv_files, test_random =  [1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515], #General function arguments: 1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515
                            split = "even", verbose = 0, percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5], #Init Key arguments
                            run_DIG = False, page_ranks = ["None"], predict = True, #DIG key arguments
                            run_CwDIG=False, connection_limit = (0.1, 0.2, 1, 10, None), #CwDIG key arguments in addition to DIG's arguments
                            run_DTA = False,
                            run_NAMA = False,
                            run_SSMA = False,
                            run_MAGAN= False,
                            run_JLMA = False,
                            run_KNN_Tests=False,
                            run_PCR = False,
                            run_CSPUD = True,
                            run_SPUD = False, operations = ("average", "abs", "normalize")) #SPUD key arguments

#Skewed
tma.run_all_tests(csv_files = csv_files, test_random =  [1825, 2830, 2969, 3407, 3430, 5198], #General function arguments: 1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515
                            split = "skewed", verbose = 0, percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5], #Init Key arguments
                            run_DIG = False, page_ranks = ["None"], predict = True, #DIG key arguments
                            run_CwDIG=False, connection_limit = (0.1, 0.2, 1, 10, None), #CwDIG key arguments in addition to DIG's arguments
                            run_DTA = False,
                            run_NAMA = False,
                            run_SSMA = False,
                            run_MAGAN= False,
                            run_JLMA = False,
                            run_KNN_Tests=False,
                            run_PCR = False,
                            run_CSPUD = True,
                            run_SPUD = False, operations = ("average", "abs", "normalize")) #SPUD key arguments

#Random
tma.run_all_tests(csv_files = csv_files, test_random =  [1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515], #General function arguments: 1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515
                            split = "random", verbose = 0, percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5], #Init Key arguments
                            run_DIG = False, page_ranks = ["None"], predict = True, #DIG key arguments
                            run_CwDIG=False, connection_limit = (0.1, 0.2, 1, 10, None), #CwDIG key arguments in addition to DIG's arguments
                            run_DTA = False,
                            run_NAMA = False,
                            run_SSMA = False,
                            run_MAGAN= False,
                            run_JLMA = False,
                            run_KNN_Tests=False,
                            run_PCR = False,
                            run_CSPUD = True,
                            run_SPUD = False, operations = ("average", "abs", "normalize")) #SPUD key arguments

#distort
tma.run_all_tests(csv_files = csv_files, test_random =  [1738, 5198, 7667, 9515], #General function arguments: 1738, 1825, 2830, 3407, 3430, 5198, 7667, 9515
                            split = "distort", verbose = 0, percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5], #Init Key arguments
                            run_DIG = False, page_ranks = ["None"], predict = True, #DIG key arguments
                            run_CwDIG=False, connection_limit = (0.1, 0.2, 1, 10, None), #CwDIG key arguments in addition to DIG's arguments
                            run_DTA = False,
                            run_NAMA = False,
                            run_SSMA = False,
                            run_MAGAN= False,
                            run_JLMA = False,
                            run_KNN_Tests=False,
                            run_PCR = False,
                            run_CSPUD = True,
                            run_SPUD = False, operations = ("average", "abs", "normalize")) #SPUD key arguments
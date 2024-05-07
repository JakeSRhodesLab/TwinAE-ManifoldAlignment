# Start by reviewing the DataVisualizationTests.ipynb; review and make any necesar changes there first

#Import everything
import test_manifold_algorithms as tma
from test_manifold_algorithms import clear_directory


file_names = ["artificial_tree", "audiology", "balance_scale", "breast_cancer", "Cancer_Data", "car", "chess", 
                "crx", "diabetes", "ecoli_5", "flare1", "glass", "heart_disease", "heart_failure", "hepatitis",
                "hill_valley", "ionosphere", "iris", "Medicaldataset", "optdigits", "parkinsons",
                "seeds", "segmentation", "tic-tac-toe", "titanic", "treeData", "water_potability", "waveform",
                "winequality-red", "zoo"]



csv_files = [f"{file_name}.csv" for file_name in file_names]

# csv_files = ["zoo.csv", "hepatitis.csv", "iris.csv", "audiology.csv", "parkinsons.csv", "seeds.csv", 
#              "segmentation.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "flare1.csv", 
#              "ecoli_5.csv", "ionosphere.csv", "Cancer_Data.csv", "hill_valley.csv", "balance_scale.csv", 
#              "crx.csv", "breast_cancer.csv", "titanic.csv", "diabetes.csv", "tic-tac-toe.csv"]


# CHECK WHICH METHODS TESTED
"""Testing All functions"""
class_instances = tma.run_all_tests(csv_files = csv_files, test_random = 2, #General function arguments
                                split = "random", verbose = 2, percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3], #Init Key arguments
                                run_DIG = True, page_ranks = ("None", "off-diagonal", "full"), predict = True, #DIG key arguments
                                run_DTA = True,
                                run_NAMA = False,
                                run_SSMA = True,
                                run_SPUD = True, kind = ["distance"]) #SPUD key arguments


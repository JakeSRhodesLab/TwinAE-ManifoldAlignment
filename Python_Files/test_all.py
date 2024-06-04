# Start by reviewing the DataVisualizationTests.ipynb; review and make any necesar changes there first

#Import everything
import test_manifold_algorithms as tma


# file_names = ["artificial_tree", "audiology", "balance_scale", "breast_cancer", "Cancer_Data", "car", "chess", 
#                 "crx", "diabetes", "ecoli_5", "flare1", "glass", "heart_disease", "heart_failure", "hepatitis",
#                 "hill_valley", "ionosphere", "iris", "Medicaldataset", "optdigits", "parkinsons",
#                 "seeds", "segmentation", "tic-tac-toe", "titanic", "treeData", "water_potability", "waveform",
#                 "winequality-red", "zoo"]



# csv_files = [f"{file_name}.csv" for file_name in file_names]

csv_files = [#"zoo.csv", "hepatitis.csv", "iris.csv", "audiology.csv", "parkinsons.csv", "seeds.csv", 
             #"segmentation.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "flare1.csv", 
             #"ecoli_5.csv", "ionosphere.csv", "Cancer_Data.csv", "hill_valley.csv", "balance_scale.csv",
             #"S-curve", "blobs",
             #"crx.csv", "breast_cancer.csv", "titanic.csv", "diabetes.csv", "tic-tac-toe.csv",
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

for split_type in ["turn", "skewed", "even", "distort", "random"]:

    tma.run_all_tests(csv_files = csv_files, test_random = [1738, 7667, 2969, 9515, 42, 3407, 3430, 5198, 5259, 2830], #General function arguments
                            split = split_type, verbose = 0, percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5], #Init Key arguments
                            run_DIG = False, page_ranks = ("None", "off-diagonal", "full"), predict = True, #DIG key arguments
                            run_DTA = False,
                            run_NAMA = False,
                            run_SSMA = False,
                            run_MAGAN= True,
                            run_JLMA = False,
                            run_KNN_Tests=False,
                            run_SPUD = False, operations = ("average", "abs"), kind = ("pure", "similarity", "distance")) #SPUD key arguments

# Start by reviewing the DataVisualizationTests.ipynb; review and make any necesar changes there first

import test_manifold_algorithms as tma

csv_files = ['audiology.csv', 'parkinsons.csv', 'heart_failure.csv', 'ecoli_5.csv',
            'ionosphere.csv', 'segmentation.csv', 'seeds.csv', 'audiology.csv', 'flare1.csv', 'heart_disease.csv', 'crx.csv', 'hill_valley.csv',
            'breast_cancer.csv', 'diabetes.csv', 'titanic.csv', 'car.csv', 'chess.csv', 'tic-tac-toe.csv', 'optdigits.csv', 'waveform.csv',
            'segmentation.csv']



"""Testing All functions"""
class_instances = tma.run_all_tests(csv_files = csv_files, test_random = 2, #General function arguments
                                split = "random", verbose = 2, percent_of_anchors = [0.05, 0.1, 0.15, 0.2, 0.3], #Init Key arguments
                                run_DIG = True, page_ranks = ("None", "off-diagonal", "full"), predict = True, #DIG key arguments
                                run_DTA = True,
                                run_NAMA = True,
                                run_SSMA = True,
                                run_SPUD = True, kind = ["distance"]) #SPUD key arguments


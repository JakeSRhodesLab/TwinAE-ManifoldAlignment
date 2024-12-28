"""
This File is to run basic baseline tests. 


To remain accurate against tests and how we score the models we are going to implement the same testing methodology as given in the Pipline file. 

Methodology: 
1. Create Train test splits with 20 % test size with the following random seeds: 1738, 5271, 9209, 1316, 42. 
2. Use get_RF_score from Pipeline_Helpers.py to get the rf scores. 
3. Use get_embedding_scores as same as above. 
4. Save it as a CSV. 
"""

#Imports
from Helpers.Pipeline_Helpers import get_RF_score, get_embedding_scores
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Data sets below.
classification_csv = [
    "zoo.csv", "hepatitis.csv", "iris.csv", "audiology.csv", "parkinsons.csv", "seeds.csv", "segmentation.csv", "glass.csv", "heart_disease.csv", "heart_failure.csv", "flare1.csv", "ecoli_5.csv", "ionosphere.csv",
    "Cancer_Data.csv", "hill_valley.csv", "balance_scale.csv", "S-curve", "blobs", 'winequality-red.csv', 'car.csv', "crx.csv", "breast_cancer.csv", "titanic.csv", "diabetes.csv", "tic-tac-toe.csv",
    'Medicaldataset.csv', "water_potability.csv", 'treeData.csv', "optdigits.csv", "waveform.csv", "chess.csv", "artificial_tree.csv"
    ]

regression_csv = [
    "EnergyEfficiency.csv", "Hydrodynamics.csv", "OpticalNetwork.csv","AirfoilSelfNoise.csv","AutoMPG.csv","ComputerHardware.csv","CommunityCrime.csv",
    "ConcreteSlumpTest.csv", "FacebookMetrics.csv", "Parkinsons.csv", "IstanbulStock.csv", "Automobile.csv", "ConcreteCompressiveStrength.csv", "SML2010.csv"
    ]


"""
Designed to test memory usages of MASH

Run with following command:
python -m memory_profiler /yunity/arusty/Graph-Manifold-Alignment/Python_Files/Test_Files/MemoryTests.py
"""

print("Process running...")

#Imports 
from Main.test_manifold_algorithms import test_manifold_algorithms as tma
from mashspud.MASH import MASH

#Create Data class to fit the data later
dc = tma("waveform.csv", split = "distort", verbose = 1)

try: 
    #Fit the data
    optimized = MASH(DTM = "kl_optimized", verbose = 4, random_state = 42, chunk_size= 100)
    optimized.fit(dc.split_A, dc.split_B, dc.anchors[:40])
    print("\nFinished fitting the optimized...")
    print(f"Optimized scores {optimized.get_scores(n_jobs = 5, n_init = 5)} \n\n")

except Exception as e:
    print("Error" + str(e))

print("\n<><><><><><><><><>>>>><><><><><><><><><><><><><><<<<<<<><><><><><><><><><><><>\n")

try:
    #Do this for comparison
    natural = MASH(DTM = "kl", verbose = 4, random_state = 42)
    natural.fit(dc.split_A, dc.split_B, dc.anchors[:40])
    print("\nFinished fitting the natural kl...")
    print(f"Optimized scores {natural.get_scores(n_jobs = 5, n_init = 5)} \n\n")

except Exception as e:
    print("Error" + str(e))

